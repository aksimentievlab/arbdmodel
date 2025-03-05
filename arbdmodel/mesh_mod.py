import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import os
from scipy.linalg import eigh
from .grid import writeDx
from . import get_resource_path


"""Process 3D meshes to calculate inertia, diffusion coefficients and generate potential fields."""

class MeshProcessor:
    """Process gmsh files to calculate inertia, diffusion coefficients and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=1.0, temperature=295, viscosity=0.01, 
                 solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM,
                 extract_surface=False):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in mass units per volume unit in g/cm^3
            temperature: Temperature in Kelvin
            viscosity: Solvent viscosity in poise (convert to Pa·s for calculations)
            solvent_density: Solvent density in g/cm3
            unit_scale: Conversion factor from input units to angstroms
            extract_surface: If True, extract surface mesh from 3D volumetric mesh
        """
        self.mesh_file = Path(mesh_file)
        self.density = density* 0.6022  # g/cm^3 to amu/AA^3
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity * 0.1  # Convert poise to Pa·s
        self.solvent_density = solvent_density

        # Initialize gmsh and read mesh
        gmsh.initialize()
        gmsh.open(str(self.mesh_file))
        
        # Get nodes and elements
        if extract_surface:
            self.nodes, self.elements = self._extract_surface_mesh()
        else:
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
        
        # Calculate diffusion coefficients 
        self._calculate_diffusion()
        
        gmsh.finalize()

    def _get_nodes(self):
        """Get mesh nodes with unit conversion"""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        return coords * self.unit_scale  # Convert to angstroms
        
    def _get_elements(self):
        """Get tetrahedral elements from the volume mesh"""
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        if not element_types:  # No volume elements found
            raise ValueError("No tetrahedral elements found in mesh")
            
        # Find tetrahedral elements (type 4 in gmsh)
        tet_idx = None
        for i, type_num in enumerate(element_types):
            if type_num == 4:  # Tetrahedral elements
                tet_idx = i
                break
                
        if tet_idx is None:
            raise ValueError("No tetrahedral elements found in mesh")
            
        # Convert to 0-based indexing and reshape to Nx4 array
        elements = np.array(node_tags[tet_idx]).reshape(-1, 4) - 1
        return elements
        
    def _extract_surface_mesh(self):
        """Extract surface mesh from a 3D volumetric mesh"""
        # Get all surface elements
        gmsh.model.mesh.createTopology()
        
        # Get surface elements
        surface_dimtags = gmsh.model.getBoundary([(3, tag) for tag in gmsh.model.getEntities(3)], 
                                                combined=False, oriented=False)
        
        # Create a new physical group for the surface
        surface_tag = gmsh.model.addPhysicalGroup(2, [tag for dim, tag in surface_dimtags])
        
        # Get nodes associated with surface elements
        all_surface_nodes = set()
        surface_elements = []
        
        for dim, tag in surface_dimtags:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
            
            # We only want triangular elements (type 2)
            tri_idx = None
            for i, type_num in enumerate(element_types):
                if type_num == 2:  # Triangle elements
                    tri_idx = i
                    break
                    
            if tri_idx is not None:
                node_tags_array = np.array(node_tags[tri_idx]).reshape(-1, 3) - 1  # Convert to 0-based
                surface_elements.extend(node_tags_array)
                all_surface_nodes.update(node_tags_array.flatten())
        
        # Get coordinates of surface nodes
        surface_nodes = []
        node_index_map = {}  # Map old indices to new ones
        
        for i, old_idx in enumerate(sorted(all_surface_nodes)):
            coords = gmsh.model.mesh.getNode(old_idx + 1)[0]  # +1 because gmsh uses 1-based indexing
            surface_nodes.append(coords)
            node_index_map[old_idx] = i
            
        # Remap element indices
        remapped_elements = []
        for element in surface_elements:
            remapped_elements.append([node_index_map[idx] for idx in element])
            
        # Convert to numpy arrays and apply unit conversion
        nodes = np.array(surface_nodes) * self.unit_scale
        elements = np.array(remapped_elements)
        
        return nodes, elements

    def _calculate_volume(self):
        """Calculate volume of the tetrahedral mesh"""
        volume = 0
        for element in self.elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.nodes[element]
            # Calculate volume using determinant method
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            volume += abs(np.dot(v01, np.cross(v02, v03))) / 6.0
        return volume

    def _calculate_center_of_mass(self):
        """Calculate center of mass of the tetrahedral mesh"""
        com = np.zeros(3)
        total_volume = 0
        
        for element in self.elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.nodes[element]
            # Calculate volume
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            tet_volume = abs(np.dot(v01, np.cross(v02, v03))) / 6.0
            # Tetrahedron centroid is average of vertices
            centroid = (v0 + v1 + v2 + v3) / 4.0
            com += centroid * tet_volume
            total_volume += tet_volume
            
        return com / total_volume

    def _calculate_inertia_tensor(self):
        """Calculate inertia tensor about center of mass for tetrahedral mesh"""
        inertia = np.zeros((3, 3))
        
        for element in self.elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.nodes[element]
            
            # Calculate volume
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            tet_volume = abs(np.dot(v01, np.cross(v02, v03))) / 6.0
            
            # Vertices contribution to inertia tensor
            vertices = np.vstack([v0, v1, v2, v3])
            
            for i in range(3):
                for j in range(3):
                    if i == j:
                        # Diagonal terms
                        term = np.sum(vertices[:, (i+1)%3]**2 + vertices[:, (i+2)%3]**2) / 10.0
                    else:
                        # Off-diagonal terms
                        term = -np.sum(vertices[:, i] * vertices[:, j]) / 10.0
                        
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

    def _calculate_diffusion(self):
        """Calculate diffusion and damping coefficients based on mesh geometry"""
        # Calculate hydrodynamic radius from gyration tensor
        gyration_tensor = np.zeros((3, 3))
        for v in self.nodes:  # Nodes are already centered at CoM during alignment
            gyration_tensor += np.outer(v, v)
        gyration_tensor /= len(self.nodes)
        
        # Get radius of gyration from the trace of gyration tensor
        Rg_squared = np.trace(gyration_tensor)
        Rg = np.sqrt(Rg_squared)
        
        # Approximation for hydrodynamic radius
        Rh = Rg * (5/3)**0.5  # in Angstroms
        
        # Convert to meters for damping calculation
        Rh_m = Rh * 1e-10  # Å to m
        
        # Calculate translational damping using Stokes' law
        # gamma = 6 * pi * eta * Rh
        gamma_trans = 6 * np.pi * self.viscosity * Rh_m  # kg/s
        
        # Convert to ARBD units (1/ns)
        # 1/ns = 1e9/s
        # kg/s * (1/amu) * (1e9/s) = 1e9/(amu*s) = 1/(amu*ns)
        amu = 1.66054e-27  # kg
        gamma_trans_arbd = gamma_trans / (self.mass * amu) * 1e9
        
        # Apply to all 3 axes (isotropic translational damping)
        self.damping_coefficient = np.array([gamma_trans_arbd, gamma_trans_arbd, gamma_trans_arbd])
        
        # For rotational damping
        # Approximation for rotational damping using sphere model first
        gamma_rot_sphere = 8 * np.pi * self.viscosity * Rh_m**3  # kg·m²/s
        
        # Convert principal moments from mesh units to SI
        principal_moments_si = self.principal_moments * 1.66e-47  # from amu/AA^2 Convert to kg·m²
        
        # Calculate rotational damping coefficient = gamma_rot / I
        # Units: kg·m²/s / (kg·m²) * 1e9 = 1e9/s = 1/ns
        gamma_rot_arbd = gamma_rot_sphere / principal_moments_si * 1e9
        
        # Refine for non-spherical objects using eigenvalues ratio
        # Sort eigenvalues for consistency
        evs = np.sort(np.diag(self.inertia_tensor))
        
        # For a more accurate treatment of non-spherical objects
        # Approximate correction factors based on axial ratios
        if evs[2] / evs[0] > 2.0:  # Significantly non-spherical
            # Adjust based on the axial ratio (simplified approximation)
            axial_ratio = np.sqrt(evs[2] / evs[0])
            correction = np.array([
                1.0,  # Smallest axis (fastest rotation)
                np.sqrt(axial_ratio),  # Middle axis
                axial_ratio  # Largest axis (slowest rotation)
            ])
            gamma_rot_arbd *= correction
            
        self.rotdamp = gamma_rot_arbd
        self.transdamp=gamma_trans_arbd 
        # Also store diffusion coefficients (not used directly in ARBD)
        # D = kB*T/gamma
        # This is just for reference/output
        kB = 1.380649e-23  # J/K
        self.D_trans = kB * self.temperature / gamma_trans  # m²/s
        self.D_rot = kB * self.temperature / gamma_rot_sphere  # rad²/s
        

    def generate_potential_grid(self, spacing=2.0, buffer=20.0, k=1.0, cutoff=10.0, max_potential=1000.0):
        """Generate potential grid for ARBD"""
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
        mask = distances <= cutoff
        potential[mask] = max_potential * (1 - 1/(1 + np.exp(-k * distances[mask])))
        
        return potential, bounds_min, spacing * np.ones(3)
        
    def write_potential_dx(self, output_file, **kwargs):
        """Generate and write potential field to DX file"""
        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)

    def save_aligned_mesh(self, output_file):
        """Save the aligned mesh to a new .msh file"""
        gmsh.initialize()
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

def process_mesh_file(mesh_file, density=1.0, temperature=295, output_dx="pod.dx", 
                     output_mesh="rod.msh",  **kwargs):
    """
    Process mesh file and calculate all properties
    
    Args:
        mesh_file: Path to .msh file
        density: Material density
        temperature: Temperature in Kelvin
        output_dx: Optional path to output potential DX file
        output_mesh: Optional path to save aligned mesh
        **kwargs: Additional arguments for potential generation
    """
    processor = MeshProcessor(mesh_file, density, temperature=temperature)
    
    print(f"Mass: {processor.mass:.3f} amu")
    print(f"Volume: {processor.volume:.3f} Å³")
    print("\nPrincipal moments of inertia:")
    print(processor.principal_moments)
    print("\nTranslational damping coefficients [1/ns]:")
    print(processor.transdamp)
    print("\nRotational damping coefficients [1/ns]:")
    print(processor.rotdamp)
    print(processor.D_trans, processor.D_rot)
    
    if output_dx:
        processor.write_potential_dx(output_dx, **kwargs)
        
    if output_mesh:
        processor.save_aligned_mesh(output_mesh)
        
    return processor

if __name__=="__main__":
    process_mesh_file("3drod.msh")
