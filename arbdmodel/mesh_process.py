import numpy as np
import gmsh
from scipy.spatial import KDTree, ConvexHull
from pathlib import Path
import subprocess
import os
import platform
from .grid import writeDx
#from .runner import HydroProRunner, write_hydropro_config
from . import get_resource_path

class MeshProcessor:
    """Process gmsh files to calculate inertia, hydrodynamics and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=1.0, temperature=295, viscosity=0.01, 
                 solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM,
                 extract_surface=True, max_beads=1500,binary_path=None, **kwargs):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in mass units per volume unit in g/cm^3
            temperature: Temperature in Kelvin
            viscosity: Solvent viscosity in poise
            solvent_density: Solvent density in g/cm3
            unit_scale: Conversion factor from input units to angstroms
            binary_path: Path to HYDROPRO executable
            extract_surface: If True, extract surface mesh from 3D volumetric mesh
            max_beads: Maximum number of beads to place on surface
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.binary_path = binary_path
        self.max_beads = max_beads

        # Initialize gmsh and read mesh
        gmsh.initialize()
        gmsh.open(str(self.mesh_file))
        

        self.nodes = self._get_nodes()
        self.elements = self._get_elements()
        # Calculate basic properties before alignment
        self.volume = self._calculate_volume()
        self.mass = self.volume * self.density * 0.6022  # g/cm^3 to amu/AA^3
                # Get nodes and elements
        if extract_surface:
            self.nodes, self.elements = self._extract_surface_mesh()
            
        self.surface_area = self._calculate_surface_area()
        
        # Calculate optimal bead radius based on surface area and max_beads
        self.bead_radius = self._calculate_optimal_bead_radius()
        
        # Align mesh to center of mass and principal axes
        self._align_mesh()
        
        # Calculate final properties after alignment
        self.inertia_tensor = self._calculate_inertia_tensor()
        self.principal_moments = np.diag(self.inertia_tensor)
        
        # Generate surface beads
        self.surface_beads = self._place_beads_on_surface()
        self._write_bead_pdb(output_path=f"{self.mesh_file.stem}_beads.pdb")
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
        surface_dimtags = gmsh.model.getBoundary([tag for tag in gmsh.model.getEntities(3)], 
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
            coords = list(gmsh.model.mesh.getNode(int(old_idx + 1)))[0]  # +1 because gmsh uses 1-based indexing
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

    def _calculate_surface_area(self):
        """Calculate surface area of the mesh"""
        area = 0
        for element in self.elements:
            # For tetrahedral mesh, we need to consider all faces
            if len(element) == 4:  # Tetrahedron
                # Each face is a triangle
                faces = [
                    [element[0], element[1], element[2]],
                    [element[0], element[1], element[3]],
                    [element[0], element[2], element[3]],
                    [element[1], element[2], element[3]]
                ]
                for face in faces:
                    v0, v1, v2 = self.nodes[face]
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                    area += face_area
            elif len(element) == 3:  # Triangle (surface mesh)
                v0, v1, v2 = self.nodes[element]
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                area += face_area
        
        return area

    def _calculate_optimal_bead_radius(self):
        """Calculate optimal bead radius based on surface area and max bead count"""
        # Estimate optimal bead radius based on surface area and max beads
        # Each bead covers approximately 4*pi*r^2 area when projected on the surface
        # But we need some overlap, so we use 2*pi*r^2 as effective area per bead
        effective_area_per_bead = self.surface_area / self.max_beads
        
        # Solve for r: 2*pi*r^2 = effective_area_per_bead
        bead_radius = np.sqrt(effective_area_per_bead / (2 * np.pi))
        
        # Calculate bounds of the mesh to ensure radius is reasonable
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        max_dim = np.max(bounds_max - bounds_min)
        
        # Make sure bead radius is reasonable (between 1% and 5% of max dimension)
        min_radius = max_dim * 0.01
        max_radius = max_dim * 0.05
        
        # Constrain radius to reasonable range
        bead_radius = max(min_radius, min(bead_radius, max_radius))
        
        print(f"Calculated optimal bead radius: {bead_radius:.2f} Å")
        print(f"Surface area: {self.surface_area:.2f} Å², Maximum beads: {self.max_beads}")
        
        return bead_radius

    def _calculate_center_of_mass(self):
        """Calculate center of mass of the tetrahedral mesh"""
        com = np.zeros(3)
        total_volume = 0
        
        for element in self.elements:
            # Get vertices of element
            if len(element) == 4:  # Tetrahedron
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
            elif len(element) == 3:  # Triangle (surface mesh)
                v0, v1, v2 = self.nodes[element]
                # For surface mesh, we approximate using triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                # Triangle centroid is average of vertices
                centroid = (v0 + v1 + v2) / 3.0
                com += centroid * area
                total_volume += area
            
        return com / total_volume

    def _calculate_inertia_tensor(self):
        """Calculate inertia tensor about center of mass for mesh"""
        inertia = np.zeros((3, 3))
        
        for element in self.elements:
            # Handle both tetrahedral and triangular elements
            if len(element) == 4:  # Tetrahedron
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
            
            elif len(element) == 3:  # Triangle (surface mesh)
                # For surface mesh, we use a thin shell approximation
                v0, v1, v2 = self.nodes[element]
                
                # Calculate area
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                
                # Thin shell approximation
                centroid = (v0 + v1 + v2) / 3.0
                
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            inertia[i,i] += self.density * area * (
                                (v0[(i+1)%3]**2 + v0[(i+2)%3]**2) +
                                (v1[(i+1)%3]**2 + v1[(i+2)%3]**2) +
                                (v2[(i+1)%3]**2 + v2[(i+2)%3]**2)
                            ) / 12.0
                        else:
                            inertia[i,j] -= self.density * area * (
                                (v0[i]*v0[j] + v1[i]*v1[j] + v2[i]*v2[j])
                            ) / 12.0
        
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

    def _place_beads_on_surface(self):
        """Place beads on the surface with uniform distribution"""
        # Calculate surface normals for each element
        normals = {}
        for i, element in enumerate(self.elements):
            if len(element) == 3:  # Triangle
                v0, v1, v2 = self.nodes[element]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Store normal for each vertex
                for idx in element:
                    if idx not in normals:
                        normals[idx] = []
                    normals[idx].append(normal)
        
        # Average normals at vertices
        for idx in normals:
            if normals[idx]:
                avg_normal = np.mean(normals[idx], axis=0)
                normals[idx] = avg_normal / np.linalg.norm(avg_normal)
        
        # First approach: try uniform sampling from the convex hull
        try:
            # Calculate a convex hull of the surface
            hull = ConvexHull(self.nodes)
            
            # Sample points on the convex hull
            bead_positions = []
            for simplex in hull.simplices:
                v0, v1, v2 = self.nodes[simplex]
                
                # Calculate triangle properties
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Number of points to sample from this triangle proportional to area
                num_points = max(1, int(area / self.surface_area * self.max_beads))
                
                # Generate random points on triangle
                for _ in range(num_points):
                    # Random barycentric coordinates
                    r1, r2 = np.random.random(2)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                    r0 = 1 - r1 - r2
                    
                    # Calculate point
                    point = r0 * v0 + r1 * v1 + r2 * v2
                    
                    # Shrink inward by bead radius
                    point = point - self.bead_radius * normal
                    
                    bead_positions.append(point)
            
            # Limit to max_beads
            if len(bead_positions) > self.max_beads:
                # Randomly select max_beads
                indices = np.random.choice(len(bead_positions), self.max_beads, replace=False)
                bead_positions = [bead_positions[i] for i in indices]
        
        except Exception as e:
            print(f"Convex hull approach failed: {e}")
            print("Falling back to vertex-based bead placement")
            
            # Fallback: use mesh vertices as bead centers
            bead_positions = []
            for idx, node in enumerate(self.nodes):
                if idx in normals:
                    # Shrink inward by bead radius along normal
                    bead_pos = node - self.bead_radius * normals[idx]
                    bead_positions.append(bead_pos)
            
            # Limit to max_beads if needed
            if len(bead_positions) > self.max_beads:
                # Randomly select max_beads
                indices = np.random.choice(len(bead_positions), self.max_beads, replace=False)
                bead_positions = [bead_positions[i] for i in indices]
        
        # Make sure we have beads
        if not bead_positions:
            raise ValueError("Failed to place beads on the surface")
            
        print(f"Generated {len(bead_positions)} beads for hydrodynamic calculation")
        return np.array(bead_positions)

    def _write_bead_pdb(self, output_path):
        """Write PDB file with bead positions for HYDROPRO"""
        with open(output_path, 'w') as f:
            for i, position in enumerate(self.surface_beads):
                x, y, z = position / 10  # Convert to nm for HYDROPRO
                # Using TER marks to help HYDROPRO recognize the shell
                if i > 0 and i % 500 == 0:
                    f.write("TER\n")
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  {self.bead_radius/10:.2f}\n")
            f.write("END\n")
            
        return output_path
        
    def generate_potential_grid(self, spacing=2.0, buffer=20.0, cutoff=10.0, max_potential=1000.0):
        """Generate potential grid for ARBD simulation"""
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
        
        # Generate potential based on bead radius
        potential = np.zeros_like(distances)
        
        # Create a soft boundary that starts at cutoff + bead_radius
        effective_cutoff = cutoff + self.bead_radius
        mask = distances <= effective_cutoff
        
        # Softer potential function that increases with distance from surface
        k = 1.0  # Steepness parameter
        potential[mask] = max_potential * (1 - 1/(1 + np.exp(-k * (distances[mask] - self.bead_radius))))
        
        return potential, bounds_min, spacing * np.ones(3)
        
    def write_potential_dx(self, output_file, **kwargs):
        """Generate and write potential field to DX file"""
        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)

    def export_aligned_mesh(self, output_path):
        """Save the aligned surface mesh as a PDB file"""
        with open(output_path, 'w') as f:
            for i, node in enumerate(self.nodes):
                x, y, z = node
                f.write(f"ATOM  {i+1:5d}  C   ALA A{(i//3)+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                if i % 3 == 2:  # Add TER after each triangle
                    f.write("TER\n")
            f.write("END\n")
        
        print(f"Exported aligned mesh to {output_path}")
        return output_path

def process_mesh_file(mesh_file, output_prefix=None, density=1.0, temperature=295, 
                    max_beads=1500, **kwargs):
    """
    Process mesh file and calculate all properties
    
    Args:
        mesh_file: Path to .msh file
        output_prefix: Prefix for output files (defaults to mesh filename)
        density: Material density in g/cm^3
        temperature: Temperature in Kelvin
        max_beads: Maximum number of beads for surface representation
        **kwargs: Additional arguments for MeshProcessor
    
    Returns:
        MeshProcessor instance
    """
    if output_prefix is None:
        output_prefix = Path(mesh_file).stem
        
    # Create processor
    processor = MeshProcessor(mesh_file, density=density, temperature=temperature, 
                            max_beads=max_beads, **kwargs)
    
    # Calculate hydrodynamics
    #processor.calculate_hydrodynamics(work_dir=f"{output_prefix}_hydro")
    
    # Generate potential field
    processor.write_potential_dx(f"{output_prefix}_potential.dx")
    
    # Export aligned mesh
    processor.export_aligned_mesh(f"{output_prefix}_aligned.pdb")
    
    # Print summary
    print("\n=== Mesh Processing Results ===")
    print(f"Mass: {processor.mass:.3f} AMU")
    print(f"Volume: {processor.volume:.3f} Å³")
    print(f"Surface area: {processor.surface_area:.3f} Å²")
    print(f"Bead radius: {processor.bead_radius:.3f} Å")
    print(f"Number of beads: {len(processor.surface_beads)}")
    print("\nPrincipal moments of inertia:")
    print(processor.principal_moments)
    print("\nTranslational damping coefficients [1/ns]:")
    print(processor.translation_damping)
    print("\nRotational damping coefficients [1/ns]:")
    print(processor.rotation_damping)
    
    return processor

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
        process_mesh_file(mesh_file)
    else:
        print("Usage: python -m mesh_processor.py mesh_file.msh")

"""
   def calculate_hydrodynamics(self, work_dir="hydrocal"):
        #Calculate hydrodynamic properties using HYDROPRO with surface beads
        work_dir = Path(work_dir)
        if not work_dir.exists():
            work_dir.mkdir(parents=True)
            
        # Create PDB file with bead positions
        pdb_path = work_dir / "hydrocal.pdb"
        self._write_bead_pdb(pdb_path)
        
        # Create HydroPro configuration
        config = {
            'structure_name': 'hydrocal',
            'temperature': self.temperature,
            'viscosity': self.viscosity, 
            'solvent_density': self.solvent_density,
            'mass': self.mass,
            'model_type': 'mesh',  # Use mesh model type for bead model
            'inertia': self.principal_moments.tolist()
        }
        
        # Write HydroPro config file in the work directory
        os.chdir(work_dir)
        write_hydropro_config(config)
        
        # Run HydroPro
        runner = HydroProRunner(self.mass, config)
        results = runner.run_calculation(work_dir=work_dir)
        
        # Store results
        self.translation_damping = results['translation_damping']
        self.rotation_damping = results['rotation_damping']
        
        print(f"Translation damping coefficients [1/ns]: {self.translation_damping}")
        print(f"Rotation damping coefficients [1/ns]: {self.rotation_damping}")
        
        return results
"""