import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import subprocess
from .grid import writeDx

"""input: 3d mesh in .msh, density of object. Output: no-entering potential, transdamp, rotdamping."""

class MeshProcessor:
    """Process gmsh files to calculate inertia, hydrodynamics and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=1.0, temperature=295, viscosity=0.01, 
                 solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM,
                 hydropro_path=None, extract_surface=False):
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
            extract_surface: If True, extract surface mesh from 3D volumetric mesh
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.hydropro_path = hydropro_path or "hydropro"
        
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
        
        # Calculate hydrodynamic properties
        self._calculate_hydro_properties()
        
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

    def _write_hydropro_input(self, work_dir):
        """Write input files for HYDROPRO"""
        # First calculate bounding box to estimate atomic element radius
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        max_dim = np.max(bounds_max - bounds_min)
        atomic_radius = max_dim / 50  # Rule of thumb for initial radius
        
        # Write PDB file with surface nodes as pseudo-atoms
        pdb_path = work_dir / "hydro.pdb"
        with open(pdb_path, 'w') as f:
            for i, node in enumerate(self.nodes):
                x, y, z = node / 10  # Convert to nm for HYDROPRO
                # Using TER marks to help HYDROPRO recognize the shell
                if i > 0 and i % 1000 == 0:
                    f.write("TER\n")
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write("END\n")
        
        # Write HYDROPRO config
        # Using HYDROPRO's shell model mode (NMC=1) with automated radius calculation
        temperature_c = self.temperature - 273.15  # Convert K to C
        config_path = work_dir / "hydropro.dat"
        with open(config_path, 'w') as f:
            f.write(f"""hydro                          ! Project title
hydro-res.txt                   ! Output file
hydro.pdb                      ! Input PDB file
1                             ! NMC (calculation mode)- shell model
{atomic_radius:.1f}            ! AER (atomic element radius in nm)
6                             ! NSIG (number of values for interpolation)
1.2                           ! SIGMIN (minimum radius of for shell calculation in nm)
3.0                           ! SIGMAX (maximum radius for shell calculation in nm)
{temperature_c}                ! TEMP (temperature in Celsius)
{self.viscosity}              ! ETA (viscosity in poise)
{self.mass}                   ! RHOPR (protein density in g/cm^3)
1.0                           ! RHOSO (solvent density in g/cm^3)
{self.solvent_density}        ! ETASO (solvent viscosity in poise)
-1                           ! IUSEP (use P or not)
-1                           ! IUSM (use M or not)
0                            ! IBEG
1                            ! IEND
*""")
        
        return pdb_path, config_path

    def _calculate_hydro_properties(self):
        """Calculate hydrodynamic properties using HYDROPRO"""
        # Create working directory
        work_dir = Path.cwd() / "hydro_calc"
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Write input files
            pdb_path, config_path = self._write_hydropro_input(work_dir)
            
            # Run HYDROPRO
            subprocess.run([self.hydropro_path], 
                         cwd=work_dir,
                         check=True,
                         capture_output=True)
            
            # Parse results file
            results_file = work_dir / "hydro-res.txt"
            lineNum = 1
            with open(results_file) as f:
                # Skip header
                while lineNum <= 48:
                    f.readline()
                    lineNum += 1
                
                # Read translational coefficients
                Dx = float(f.readline().split()[0])
                Dy = float(f.readline().split()[1])
                Dz = float(f.readline().split()[2])
                
                # Skip two lines
                f.readline()
                f.readline()
                
                # Read rotational coefficients
                Rx = float(f.readline().split()[3])
                Ry = float(f.readline().split()[4])
                Rz = float(f.readline().split()[5])
            
            # Convert units
            # Translation: "(295 k K) / (( cm^2/s) *  amu)" "1/ns"
            self.translation_damping = [24.527692/(x*self.mass) for x in [Dx,Dy,Dz]]
            
            # Rotation: "(295 k K) / ((1 /s) *  amu AA^2)" "1/ns"
            self.rotation_damping = [2.4527692e+17 / (x*self.mass) for x in [Rx,Ry,Rz]]
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)

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

def process_mesh_file(mesh_file, density=1.0, temperature=295, output_dx=None, 
                     output_mesh=None, hydropro_path=None, **kwargs):
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
    processor = MeshProcessor(mesh_file, density, temperature=temperature,
                            hydropro_path=hydropro_path)
    
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
