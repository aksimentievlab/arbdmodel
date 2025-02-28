import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import subprocess
from .grid import writeDx


class MeshProcessor:
    """Unified processor for mesh files with VMD integration and hydrodynamic calculations"""
    
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=1.0, temperature=295, viscosity=0.01,
                 solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM,
                 hydropro_path=None, vmd_path=None, extract_surface=False):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in mass units per volume unit
            temperature: Temperature in Kelvin
            viscosity: Solvent viscosity in poise
            solvent_density: Solvent density in g/cm3
            unit_scale: Conversion factor from input units to angstroms
            hydropro_path: Path to HYDROPRO executable
            vmd_path: Path to VMD executable
            extract_surface: If True, extract surface mesh from 3D volumetric mesh
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.hydropro_path = hydropro_path or "hydropro"
        self.vmd_path = vmd_path or "vmd"
        
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
        
        # Calculate hydrodynamic properties if requested
        if hydropro_path:
            self._calculate_hydro_properties()
            
        gmsh.finalize()

    def _get_nodes(self):
        """Get mesh nodes with unit conversion"""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        return coords * self.unit_scale

    def _get_elements(self):
        """Get tetrahedral elements from the volume mesh"""
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        if not element_types:  # No volume elements found
            # Try surface elements instead
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
            if not element_types:
                raise ValueError("No tetrahedral or triangular elements found in mesh")
                
        if 4 in element_types:  # Tetrahedral elements
            tet_idx = element_types.index(4)
            elements = np.array(node_tags[tet_idx]).reshape(-1, 4) - 1
        else:  # Triangle elements
            tri_idx = element_types.index(2)
            elements = np.array(node_tags[tri_idx]).reshape(-1, 3) - 1
            
        return elements

    def _extract_surface_mesh(self):
        """Extract surface mesh from a 3D volumetric mesh"""
        # Get all surface elements
        gmsh.model.mesh.createTopology()
        surface_dimtags = gmsh.model.getBoundary([(3, tag) for tag in gmsh.model.getEntities(3)],
                                                combined=False, oriented=False)
        
        # Process surface elements
        all_surface_nodes = set()
        surface_elements = []
        
        for dim, tag in surface_dimtags:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
            
            # Find triangular elements (type 2)
            tri_idx = None
            for i, type_num in enumerate(element_types):
                if type_num == 2:
                    tri_idx = i
                    break
                    
            if tri_idx is not None:
                node_tags_array = np.array(node_tags[tri_idx]).reshape(-1, 3) - 1
                surface_elements.extend(node_tags_array)
                all_surface_nodes.update(node_tags_array.flatten())
        
        # Get coordinates and create mapping
        surface_nodes = []
        node_index_map = {}
        
        for i, old_idx in enumerate(sorted(all_surface_nodes)):
            coords = gmsh.model.mesh.getNode(old_idx + 1)[0]
            surface_nodes.append(coords)
            node_index_map[old_idx] = i
            
        # Remap element indices
        remapped_elements = []
        for element in surface_elements:
            remapped_elements.append([node_index_map[idx] for idx in element])
            
        nodes = np.array(surface_nodes) * self.unit_scale
        elements = np.array(remapped_elements)
        
        return nodes, elements

    def write_vmd_files(self, output_prefix):
        """Write mesh as PDB/PSF files for VMD visualization"""
        base_name = Path(output_prefix)
        
        # Write PDB file
        with open(f"{base_name}.pdb", 'w') as f:
            f.write("REMARK   Generated from mesh file\n")
            for i, node in enumerate(self.nodes):
                x, y, z = node / 10  # Convert to nm for PDB format
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
                if i > 0 and i % 1000 == 0:
                    f.write("TER\n")
            f.write("END\n")
            
        # Write PSF file with connectivity
        with open(f"{base_name}.psf", 'w') as f:
            f.write("PSF\n\n")
            f.write(f"{len(self.nodes):8d} !NATOM\n")
            for i in range(len(self.nodes)):
                f.write(f"{i+1:8d} PROT 1    ALA  CA   CA    0.000000    12.0110           0\n")
                
            # Write bonds based on mesh elements
            bonds = set()
            for element in self.elements:
                for i in range(len(element)):
                    for j in range(i+1, len(element)):
                        bond = tuple(sorted([element[i], element[j]]))
                        bonds.add(bond)
                        
            f.write(f"\n{len(bonds):8d} !NBOND\n")
            for i, (a1, a2) in enumerate(sorted(bonds)):
                f.write(f"{a1+1:8d}{a2+1:8d}")
                if i % 4 == 3:
                    f.write("\n")
            if len(bonds) % 4 != 0:
                f.write("\n")


    def _calculate_volume(self):
        """Calculate volume of the mesh"""
        volume = 0
        for element in self.elements:
            if len(element) == 4:  # Tetrahedral element
                v0, v1, v2, v3 = self.nodes[element]
                v01 = v1 - v0
                v02 = v2 - v0
                v03 = v3 - v0
                volume += abs(np.dot(v01, np.cross(v02, v03))) / 6.0
            else:  # Triangle element (surface)
                v0, v1, v2 = self.nodes[element]
                volume += abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        return volume

    def _calculate_center_of_mass(self):
        """Calculate center of mass"""
        com = np.zeros(3)
        total_volume = 0
        
        for element in self.elements:
            if len(element) == 4:  # Tetrahedral
                v0, v1, v2, v3 = self.nodes[element]
                v01 = v1 - v0
                v02 = v2 - v0
                v03 = v3 - v0
                volume = abs(np.dot(v01, np.cross(v02, v03))) / 6.0
                centroid = (v0 + v1 + v2 + v3) / 4.0
            else:  # Triangle
                v0, v1, v2 = self.nodes[element]
                volume = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
                centroid = (v0 + v1 + v2) / 3.0
                
            com += centroid * volume
            total_volume += volume
            
        return com / total_volume

    def _calculate_inertia_tensor(self):
        """Calculate inertia tensor about center of mass"""
        inertia = np.zeros((3, 3))
        
        for element in self.elements:
            vertices = self.nodes[element]
            
            if len(element) == 4:  # Tetrahedral
                volume = abs(np.dot(vertices[1] - vertices[0],
                                  np.cross(vertices[2] - vertices[0],
                                         vertices[3] - vertices[0]))) / 6.0
                denominator = 10.0
            else:  # Triangle
                volume = abs(np.dot(vertices[0],
                                  np.cross(vertices[1], vertices[2]))) / 6.0
                denominator = 20.0
                
            for i in range(3):
                for j in range(3):
                    if i == j:
                        term = np.sum(vertices[:, (i+1)%3]**2 +
                                    vertices[:, (i+2)%3]**2) / denominator
                    else:
                        term = -np.sum(vertices[:, i] * vertices[:, j]) / denominator
                        
                    inertia[i,j] += self.density * volume * term
        
        return inertia

    def _align_mesh(self):
        """Align mesh to center of mass and principal axes"""
        # Center the mesh
        com = self._calculate_center_of_mass()
        self.nodes -= com
        
        # Calculate and diagonalize inertia tensor
        inertia = self._calculate_inertia_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Sort by eigenvalues
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
            
        # Rotate mesh
        self.nodes = self.nodes @ eigenvectors
        
        # Store transformation
        self.rotation_matrix = eigenvectors
        self.translation = com

    def _calculate_hydro_properties(self):
        """Calculate hydrodynamic properties using HYDROPRO"""
        from .runner import HydroProRunner
        
        # Create HydroProRunner instance with same parameters
        runner = HydroProRunner(
            binary_path=self.hydropro_path,
            temperature=self.temperature,
            viscosity=self.viscosity,
            solvent_density=self.solvent_density
        )
        
        # Create working directory
        work_dir = Path.cwd() / "hydro_calc"
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Write PDB file
            pdb_path = work_dir / "structure.pdb"
            with open(pdb_path, 'w') as f:
                for i, node in enumerate(self.nodes):
                    x, y, z = node / 10  # Convert to nm for PDB format
                    if i > 0 and i % 1000 == 0:
                        f.write("TER\n")
                    f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
                f.write("END\n")
            
            # Run calculation
            results = runner.run_calculation(
                structure_name="structure",
                mass=self.mass,
                work_dir=str(work_dir)
            )
            
            # Store results
            self.translation_damping = results['translation_damping']
            self.rotation_damping = results['rotation_damping']
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)

    def generate_potential_grid(self, spacing=2.0, buffer=20.0, k=1.0, 
                              cutoff=10.0, max_potential=1000.0):
        """Generate potential grid"""
        bounds_min = np.min(self.nodes, axis=0) - buffer
        bounds_max = np.max(self.nodes, axis=0) + buffer
        
        npts = np.ceil((bounds_max - bounds_min) / spacing).astype(int)
        x = np.linspace(bounds_min[0], bounds_max[0], npts[0])
        y = np.linspace(bounds_min[1], bounds_max[1], npts[1])
        z = np.linspace(bounds_min[2], bounds_max[2], npts[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        tree = KDTree(self.nodes)
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        distances, _ = tree.query(grid_points)
        distances = distances.reshape(X.shape)
        
        potential = np.zeros_like(distances)
        mask = distances <= cutoff
        potential[mask] = max_potential * (1 - 1/(1 + np.exp(-k * distances[mask])))
        
        return potential, bounds_min, spacing * np.ones(3)

    def write_potential_dx(self, output_file, **kwargs):
        """Write potential field to DX file"""
        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)
