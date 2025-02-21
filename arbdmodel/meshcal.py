import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
from .grid import writeDx

class MeshProcessor:
    """Process gmsh files to calculate inertia and generate potential fields"""
    
    # Conversion factor from microns to angstroms
    MICRON_TO_ANGSTROM = 10000  
    
    def __init__(self, mesh_file, density=1.0, unit_scale=MICRON_TO_ANGSTROM):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in mass units per volume unit
            unit_scale: Conversion factor from input units to angstroms
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        
        # Initialize gmsh and read mesh
        gmsh.initialize()
        gmsh.open(str(self.mesh_file))
        
        # Get nodes and elements
        self.nodes = self._get_nodes()
        self.elements = self._get_elements()
        
        # Calculate basic properties before alignment
        self.volume = self._calculate_volume()
        self.mass = self.volume * self.density
        
        # Align mesh to center of mass and principal axes
        self._align_mesh()
        
        # Calculate final properties after alignment
        self.inertia_tensor = self._calculate_inertia_tensor()
        self.principal_moments = np.diag(self.inertia_tensor)
        
        gmsh.finalize()

    def _get_nodes(self):
        """Get mesh nodes with unit conversion"""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        return coords * self.unit_scale  # Convert to angstroms
        
    def _get_elements(self):
        """Get triangular elements from the surface mesh"""
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
        if not element_types:  # No surface elements found
            raise ValueError("No surface elements found in mesh")
        return np.array(node_tags[0]).reshape(-1, 3) - 1  # Convert to 0-based indexing

    def _calculate_volume(self):
        """Calculate volume of the mesh"""
        volume = 0
        for element in self.elements:
            triangle = self.nodes[element]
            v1, v2, v3 = triangle
            volume += abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        return volume

    def _calculate_center_of_mass(self):
        """Calculate center of mass of the mesh"""
        com = np.zeros(3)
        total_volume = 0
        
        for element in self.elements:
            triangle = self.nodes[element]
            v1, v2, v3 = triangle
            tet_volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            centroid = (v1 + v2 + v3) / 4.0  # Tetrahedron centroid
            com += centroid * tet_volume
            total_volume += tet_volume
            
        return com / total_volume

    def _calculate_inertia_tensor(self):
        """Calculate inertia tensor about center of mass"""
        inertia = np.zeros((3, 3))
        
        for element in self.elements:
            triangle = self.nodes[element]
            v1, v2, v3 = triangle
            
            # Calculate contribution to inertia tensor from tetrahedron
            tet_volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            
            for i in range(3):
                for j in range(3):
                    if i == j:
                        # Diagonal terms
                        term = (v1[(i+1)%3]**2 + v1[(i+2)%3]**2 +
                               v2[(i+1)%3]**2 + v2[(i+2)%3]**2 +
                               v3[(i+1)%3]**2 + v3[(i+2)%3]**2) / 20.0
                    else:
                        # Off-diagonal terms
                        term = -(v1[i]*v1[j] + v2[i]*v2[j] + v3[i]*v3[j]) / 20.0
                    
                    inertia[i,j] += self.density * tet_volume * term
        
        return inertia

    def _align_mesh(self):
        """Align mesh to center of mass and principal axes"""
        # First center the mesh
        com = self._calculate_center_of_mass()
        self.nodes -= com
        
        # Calculate and diagonalize inertia tensor
        inertia = self._calculate_inertia_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Sort by eigenvalues to get consistent orientation
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
            
        # Rotate mesh to align with principal axes
        self.nodes = self.nodes @ eigenvectors
        
        # Store transformation
        self.rotation_matrix = eigenvectors
        self.translation = com

    def generate_potential_grid(self, spacing=2.0, buffer=20.0, k=1.0, cutoff=10.0, max_potential=1000.0):
        """
        Generate potential grid for ARBD
        
        Args:
            spacing: Grid spacing in angstroms
            buffer: Extra space around mesh bounds in angstroms
            k: Steepness parameter for sigmoid potential
            cutoff: Distance cutoff for potential in angstroms
            max_potential: Maximum potential value
            
        Returns:
            Tuple of (grid data, origin coordinates, delta spacing)
        """
        # Calculate grid bounds with buffer
        bounds_min = np.min(self.nodes, axis=0) - buffer
        bounds_max = np.max(self.nodes, axis=0) + buffer
        
        # Create grid points
        npts = np.ceil((bounds_max - bounds_min) / spacing).astype(int)
        x = np.linspace(bounds_min[0], bounds_max[0], npts[0])
        y = np.linspace(bounds_min[1], bounds_max[1], npts[1])
        z = np.linspace(bounds_min[2], bounds_max[2], npts[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create KD-tree for fast distance calculations
        tree = KDTree(self.nodes)
        
        # Calculate distances to nearest surface points
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        distances, _ = tree.query(grid_points)
        distances = distances.reshape(X.shape)
        
        # Generate potential
        potential = np.zeros_like(distances)
        
        # Inside cutoff distance
        mask = distances <= cutoff
        potential[mask] = max_potential * (1 - 1/(1 + np.exp(-k * distances[mask])))
        
        return potential, bounds_min, spacing * np.ones(3)
        
    def write_potential_dx(self, output_file, **kwargs):
        """
        Generate and write potential field to DX file
        
        Args:
            output_file: Path to output .dx file
            **kwargs: Arguments passed to generate_potential_grid
        """
        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)

    def save_aligned_mesh(self, output_file):
        """
        Save the aligned mesh to a new .msh file
        
        Args:
            output_file: Path to output .msh file
        """
        gmsh.initialize()
        
        # Create new model
        gmsh.model.add("aligned_mesh")
        
        # Add nodes
        node_tags = []
        node_coords = []
        for i, node in enumerate(self.nodes):
            tag = i + 1
            node_tags.append(tag)
            node_coords.extend(node / self.unit_scale)  # Convert back to microns
            
        gmsh.model.mesh.addNodes(2, 1, node_tags, node_coords)
        
        # Add elements
        element_tags = []
        element_nodes = []
        for i, element in enumerate(self.elements):
            tag = i + 1
            element_tags.append(tag)
            element_nodes.extend([x + 1 for x in element])  # Convert to 1-based indexing
            
        gmsh.model.mesh.addElements(2, 1, [2], [element_tags], [element_nodes])
        
        # Write mesh
        gmsh.write(str(output_file))
        gmsh.finalize()

def process_mesh_file(mesh_file, density=1.0, output_dx=None, output_mesh=None, **kwargs):
    """
    Convenience function to process mesh file and optionally generate potential
    
    Args:
        mesh_file: Path to .msh file
        density: Material density
        output_dx: Optional path to output potential DX file
        output_mesh: Optional path to save aligned mesh
        **kwargs: Additional arguments for potential generation
        
    Returns:
        MeshProcessor instance
    """
    processor = MeshProcessor(mesh_file, density)
    
    print(f"Mass: {processor.mass:.3f}")
    print(f"Volume: {processor.volume:.3f}")
    print("\nPrincipal moments of inertia:")
    print(processor.principal_moments)
    print("\nRotation matrix to principal axes:")
    print(processor.rotation_matrix)
    print("\nTranslation to center:")
    print(processor.translation)
    
    if output_dx:
        processor.write_potential_dx(output_dx, **kwargs)
        
    if output_mesh:
        processor.save_aligned_mesh(output_mesh)
        
    return processor

if __name__ == "__main__":
    mesh_file="r10ar5.msh"
    process_mesh_file(mesh_file, density=1.0, output_dx=None, output_mesh=None)
