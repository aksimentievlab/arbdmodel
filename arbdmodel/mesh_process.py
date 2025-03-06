import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import subprocess
from .grid import writeDx
from .runner import HydroProRunner
from . import get_resource_path
import platform
import os


"""input: 3d mesh in .msh, density of object. Output: no-entering potential, transdamp, rotdamping."""

class MeshProcessor:
    """Process gmsh files to calculate inertia, hydrodynamics and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=19.3, temperature=295, viscosity=0.01, 
                 solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM,
                 binary_path=None, extract_surface=False):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in mass units per volume unit in g/cm^3
            temperature: Temperature in Kelvin
            viscosity: Solvent viscosity in poise
            solvent_density: Solvent density in g/cm3
            unit_scale: Conversion factor from input units to angstroms
            hydropro_path: Path to HYDROPRO executable
            extract_surface: If True, extract surface mesh from 3D volumetric mesh
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.binary_path=binary_path


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
        self.mass = self.volume * self.density*0.6022 #g/cm^3 to amu/AA^3
        
        # Align mesh to center of mass and principal axes
        self._align_mesh()
        
        # Calculate final properties after alignment
        self.inertia_tensor = self._calculate_inertia_tensor()
        self.principal_moments = np.diag(self.inertia_tensor)
        
        # Calculate hydrodynamic properties
        self.runner=HydroProRunner(self.mass,temperature=temperature,viscosity=viscosity, binary_path=binary_path)
        self._get_damping()
        print(self.transdamp,self.rotational_damping_coefficient)
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

    def _get_damping(self, work_dir="hydrocalc"):
        """Write input files for HYDROPRO"""
        # First calculate bounding box to estimate atomic element radius
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        max_dim = np.max(bounds_max - bounds_min)
        atomic_radius = max_dim / 50  # Rule of thumb for initial radius
        # Write PDB file with surface nodes as pseudo-atoms

        try:
            os.listdir(work_dir)
        except:
            os.mkdir(work_dir)

        work_dir=Path(work_dir)
        base_path = work_dir / "hydrocal"
        self._save_aligned_mesh_both_formats(base_path)
        
        # Write HYDROPRO config
        # Using HYDROPRO's shell model mode (NMC=1) with automated radius calculation
            
        # Run HydroPro to get hydrodynamic properties
        results = self.runner.run_calculation(
            work_dir=work_dir
        )
        
        self.damping_coefficient = results['translation_damping']
        self.rotational_damping_coefficient = results['rotation_damping']
        pdb_path=str(base_path)+".pdb"
        return pdb_path


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
        surface_tag = gmsh.model.addDiscreteEntity(2)
        # Add nodes
        node_tags = []
        node_coords = []
        for i, node in enumerate(self.nodes):
            tag = i + 1
            node_tags.append(tag)
            node_coords.extend(node / self.unit_scale)  # Convert back to mesh units
            
        gmsh.model.mesh.addNodes(2, 1, node_tags, node_coords)
        
        # Add elements
        element_tags = []
        element_nodes = []
        for i, element in enumerate(self.elements):
            tag = i + 1
            element_tags.append(tag)
            element_nodes.extend([int(x + 1) for x in element])  # Convert to 1-based indexing
            
        gmsh.model.mesh.addElements(2, 1, [2], [element_tags], [element_nodes])
        
        # Write mesh
        gmsh.write(str(output_file))
        gmsh.finalize()
            
    def save_as_pdb(self, output_file):
        """Save the aligned mesh as a PDB file (coordinates in Å)"""
        # PDB format specifications
        HEADER = "HEADER    ALIGNED MESH                           "
        ATOM_FORMAT = "ATOM  {:5d} {:4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
        
        with open(output_file, 'w') as f:
            # Write header
            from datetime import datetime
            date_str = datetime.now().strftime("%d-%b-%y")
            f.write(f"{HEADER}{date_str}   XXXX\n")
            
            # Calculate box dimensions for CRYST record
            bounds_min = np.min(self.nodes, axis=0)
            bounds_max = np.max(self.nodes, axis=0)
            box_dimensions = bounds_max - bounds_min
            
            # Write crystallographic information
            f.write(f"CRYST1{box_dimensions[0]:9.3f}{box_dimensions[1]:9.3f}{box_dimensions[2]:9.3f}  90.00  90.00  90.00 P 1           1\n")
            
            # Write atoms (nodes from the mesh)
            for i, node in enumerate(self.nodes):
                # PDB uses 1-indexed atom numbers
                atom_num = i + 1
                # Use CA (alpha carbon) atom type for visibility in viewers
                atom_name = " CA "
                # Use residue name MES for mesh
                res_name = "MES"
                # Use chain ID A
                chain_id = "A"
                # Residue number = atom number for simplicity
                res_num = atom_num
                # X, Y, Z coordinates in Ångströms (already in the correct units)
                x, y, z = node
                # Use 1.0 for occupancy and 0.0 for temperature factor
                occupancy = 1.0
                temp_factor = 0.0
                
                f.write(f"{ATOM_FORMAT.format(atom_num, atom_name, res_name, chain_id, res_num, x, y, z, occupancy, temp_factor)}\n")
            
            # Write connectivity (the mesh elements) as CONECT records
            for element in self.elements:
                # PDB CONECT records use 1-indexed atom numbers
                atom_indices = [idx + 1 for idx in element]
                
                # Create a CONECT record for each edge of the triangle
                edges = [(atom_indices[0], atom_indices[1]), 
                         (atom_indices[1], atom_indices[2]), 
                         (atom_indices[2], atom_indices[0])]
                
                for a1, a2 in edges:
                    f.write(f"CONECT{a1:5d}{a2:5d}\n")
            
            # Write end of file
            f.write("END\n")
        
        print(f"Saved aligned mesh as PDB: {output_file}")
    
    def _save_aligned_mesh_both_formats(self, base_filename):
        """Save the aligned mesh in both MSH and PDB formats"""
        # Save as MSH (in microns)
        msh_filename = f"{base_filename}.msh"
        self.save_aligned_mesh(msh_filename)
        print(f"Saved aligned mesh as MSH: {msh_filename}")
        
        # Save as PDB (in Ångströms)
        pdb_filename = f"{base_filename}.pdb"
        self.save_as_pdb(pdb_filename)
        print(f"Saved aligned mesh as PDB: {pdb_filename}")
        
        # If we have components, also save those
        if hasattr(self, 'component_indices'):
            comp_filename = f"{base_filename}_components.pdb"
            self.visualize_components(comp_filename)
            print(f"Saved component visualization: {comp_filename}")


def process_mesh_file(mesh_file, temperature=295, output_dx="pod.dx", 
                     output_mesh="rod.msh",  **kwargs):
    """
    Process mesh file and calculate all properties
    
    Args:
        mesh_file: Path to .msh file
        density: Material density
        temperature: Temperature in Kelvin
        output_dx: Optional path to output potential DX file
        output_mesh: Optional path to save aligned mesh
        hydropro_path: Path to HYDROPRO executable
        **kwargs: Additional arguments for potential generation
    """
    processor = MeshProcessor(mesh_file, temperature=temperature,)
    
    print(f"Mass: {processor.mass:.3f}")
    print(f"Volume: {processor.volume:.3f}")
    print("\nPrincipal moments of inertia:")
    print(processor.principal_moments)
    print("\nTranslational damping coefficients [1/ns]:")
    print(processor.translation_damping)
    print("\nRotational damping coefficients [1/ns]:")
    print(processor.rotation_damping)
    
    if output_dx:
        processor.write_potential_dx(output_dx, **kwargs)
        
    if output_mesh:
        processor.save_aligned_mesh(output_mesh)
        
    return processor
if __name__=="__main__":
    process_mesh_file("3drod.msh")
