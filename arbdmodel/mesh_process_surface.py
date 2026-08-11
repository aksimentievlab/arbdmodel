import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import os
from .grid import writeDx
from .engine import HydroProRunner
from .logger import logger

"""
Specialized surface mesh processor that handles triangle surface meshes 
""" 
    # Conversion factors
class SurfaceMeshProcessor:
    """Process surface meshes to calculate inertia, hydrodynamics and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=19.3, simconf=None, 
                 unit_scale=MICRON_TO_ANGSTROM, work_dir=None):
        
        """
        Initialize processor with surface mesh file
        Args:
            mesh_file: Path to .msh file with surface triangles
            density: Material density in g/cm^3
            simconf: SimConf object containing configuration parameters
            unit_scale: Conversion factor from input units to angstroms
            work_dir: Directory to store output files (default: current directory)
        """
        self.mesh_file = Path(mesh_file)
        self.density = density*0.6022
        self.unit_scale = unit_scale
        self.name = str(self.mesh_file.stem)
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        os.makedirs(self.work_dir, exist_ok=True)

        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()
        self.simconf=simconf
        # Extract parameters from simconf
        self.temperature = simconf.temperature
        self.viscosity = simconf.viscosity
        self.solvent_density = simconf.solvent_density
        self.binary_path = simconf.get_binary('hydropro')
        self.attached_patricles=[]
        # Initialize gmsh and read mesh
        gmsh.initialize()
        try:
            gmsh.open(str(self.mesh_file))
            logger.info(f"Successfully opened mesh file: {self.mesh_file}")

            self.nodes, self.elements = self._load_surface_mesh()
            logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.elements)} elements from surface mesh")
            
            # Calculate basic properties before alignment
            self.volume = self._calculate_volume()
            self.mass = self.volume * self.density
            self.density_correction = 1.0
                
            logger.info(f"Calculated volume: {self.volume:.2f} Å³")
            logger.info(f"Calculated mass: {self.mass:.2f} amu")
            
            # Align mesh to center of mass and principal axes
            self._align_mesh()
            
            # Calculate final properties after alignment
            self.inertia_tensor = self._calculate_inertia_tensor()

            self.principal_moments = np.diag(self.inertia_tensor)
            logger.info(f"Principal moments of inertia: {self.principal_moments}")
            
        except Exception as e:
            logger.error(f"Error processing mesh: {e}")
        finally:
            gmsh.finalize()

    def get_attached_particles(self):
        from . import ParticleType, PointParticle
        rbp=ParticleType(name=f"{self.name}_attached", mass=1,diffusivity=1)
        logger.info(f"attaching particles type {self.name}_attached using mesh nodes")

        for i, node in enumerate(self.nodes):
            x, y, z = node
            rbpi=arbdmodel.PointParticle(rbp,name=f"{self.name}_{i}",position=[x,y,z])
            self.attached_patricles.append(rbpi)

    def _load_surface_mesh(self):
        """Load triangular surface mesh directly"""
        # Get all 2D entities (surfaces)
        surface_entities = gmsh.model.getEntities(2)
        if not surface_entities:
            logger.error("No surface entities found in mesh")
            
        # Collect all triangular elements
        all_nodes = set()
        all_elements = []
        
        for dim, tag in surface_entities:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
            
            # Find triangular elements (type 2 in gmsh)
            tri_idx = None
            for i, type_num in enumerate(element_types):
                if type_num == 2:  # Triangle elements
                    tri_idx = i
                    break
                    
            if tri_idx is not None:
                element_node_tags = np.array(node_tags[tri_idx]).reshape(-1, 3) - 1  # Convert to 0-based
                all_elements.extend(element_node_tags)
                all_nodes.update(element_node_tags.flatten())
        
        if not all_elements:
            logger.error("No triangular elements found in mesh")
            
        # Get coordinates of all nodes
        node_coords = {}
        for node_idx in all_nodes:
            coords = gmsh.model.mesh.getNode(int(node_idx + 1))[0]  # +1 because gmsh uses 1-based indexing
            node_coords[node_idx] = coords
            
        # Create node array in sorted order
        sorted_node_indices = sorted(all_nodes)
        nodes = np.array([node_coords[idx] for idx in sorted_node_indices]) * self.unit_scale  # Apply unit conversion
        
        # Create mapping from old to new indices
        node_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_node_indices)}
        
        # Remap element indices
        remapped_elements = []
        for element in all_elements:
            remapped_elements.append([node_index_map[idx] for idx in element])
        
        return nodes, np.array(remapped_elements)

    def _calculate_volume(self):
        """Calculate volume of the closed surface mesh using the divergence theorem"""
        total_volume = 0
        # For a closed surface mesh, orientation is important for correct signing
        for element in self.elements:
            # Get vertices of triangle
            v0, v1, v2 = self.nodes[element]
            
            # Calculate signed volume of tetrahedron formed by triangle and origin
            # V = (1/6) * ((v1-v0) × (v2-v0)) · v0
            v01 = v1 - v0
            v02 = v2 - v0
            cross_product = np.cross(v01, v02)
            
            # For a properly oriented closed mesh, we use the signed volume
            # This relies on triangle winding order being consistent
            volume = np.dot(cross_product, v0) / 6.0
            
            # Add the signed volume (will be negative for triangles facing inward)
            total_volume += volume
        
        # Take absolute value of the final result to handle any mesh orientation issues
        return abs(total_volume)

    def _calculate_center_of_mass(self):
        """Calculate center of mass of the surface mesh"""
        com = np.zeros(3)
        total_weight = 0
        
        # For a surface mesh, weight each triangle by its area
        for element in self.elements:
            # Get vertices of triangle
            v0, v1, v2 = self.nodes[element]
            
            # Calculate area using cross product
            v01 = v1 - v0
            v02 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(v01, v02))
            
            # Centroid of triangle
            centroid = (v0 + v1 + v2) / 3.0
            
            # Add weighted contribution
            com += centroid * area
            total_weight += area
            
        return com / total_weight if total_weight > 0 else np.zeros(3)

    def _calculate_inertia_tensor(self):
        inertia = np.zeros((3, 3))
        
        # Compute total mass based on volume and density
        mass = self.volume * self.density * 0.6022
        
        # Get bounding box dimensions to estimate shape
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        dimensions = bounds_max - bounds_min
        
        # Sort dimensions to identify major axes
        sorted_dims = np.sort(dimensions)
        
        # For a rod-like object with aspect ratio AR
        # Assuming AR is the ratio of length to diameter
        length = np.max(dimensions)
        diameter = np.min(dimensions)
        if length > 0 and diameter > 0:
            AR = length / diameter
        else:
            AR = 1.0
            
        logger.info(f"Detected dimensions: {dimensions}")
        logger.info(f"Estimated aspect ratio: {AR:.2f}")
        
        # Calculate moments of inertia for a cylindrical rod
        # For a uniform rod of length L and radius R:
        # I_axial = (1/2)MR²
        # I_transverse = (1/12)ML² + (1/4)MR²
        radius = diameter / 2.0
        
        # Along major axis (assuming z is the major axis)
        I_axial = 0.5 * mass * radius**2
        
        # Perpendicular to major axis
        I_transverse = (1/12.0) * mass * length**2 + (1/4.0) * mass * radius**2
        
        # Create a diagonal inertia tensor based on these analytic expressions
        # Identify which axis is the major axis (longest dimension)
        major_axis_idx = np.argmax(dimensions)
        
        for i in range(3):
            if i == major_axis_idx:
                inertia[i, i] = I_axial
            else:
                inertia[i, i] = I_transverse
        
        return inertia

    def _align_mesh(self):
        """Align mesh to center of mass and principal axes with longest dimension along Z-axis"""
        # First center the mesh
        com = self._calculate_center_of_mass()
        self.nodes -= com
        
        # Get bounding box dimensions to estimate shape
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        dimensions = bounds_max - bounds_min
        logger.info(f"Pre-alignment dimensions: {dimensions}")
        
        # Calculate and diagonalize inertia tensor
        inertia = self._calculate_inertia_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # For a rod, smallest moment of inertia is along the rod axis
        # We want this to be aligned with the Z-axis (axis 2)
        sort_idx = np.argsort(eigenvalues)
        
        # Reorder to put smallest moment (rod axis) as Z
        # Z should be axis 2, so we want the smallest moment to be last
        z_vec = eigenvectors[:, sort_idx[0]]  # Vector of smallest moment (rod axis)
        x_vec = eigenvectors[:, sort_idx[1]]
        y_vec = eigenvectors[:, sort_idx[2]]
        
        # Create new eigenvector matrix with Z as the rod axis
        reordered_eigenvectors = np.column_stack([x_vec, y_vec, z_vec])
        
        # Ensure right-handed coordinate system
        if np.linalg.det(reordered_eigenvectors) < 0:
            reordered_eigenvectors[:, 0] *= -1
            
        # Rotate mesh to align with principal axes with rod along Z
        self.nodes = self.nodes @ reordered_eigenvectors
        
        # Get post-alignment dimensions
        bounds_min = np.min(self.nodes, axis=0)
        bounds_max = np.max(self.nodes, axis=0)
        dimensions = bounds_max - bounds_min
        logger.info(f"Post-alignment dimensions: {dimensions}")
        
        # Verify that Z is now the longest dimension
        if dimensions[2] < max(dimensions[0], dimensions[1]):
            logger.info("WARNING: Z-axis is not the longest dimension after alignment!")
            
        # Calculate aspect ratio (using Z as length)
        length = dimensions[2]  # Z axis should be the rod length
        width = max(dimensions[0], dimensions[1])  # Use max of X or Y for width
        self.aspect_ratio = length / width if width > 0 else 1.0
        logger.info(f"Calculated aspect ratio: {self.aspect_ratio:.6f}")
        
        # Store transformation
        self.rotation_matrix = reordered_eigenvectors
        self.translation = com
        logger.info(f"Aligned mesh to principal axes with rod along Z-axis, COM: {com}")

    def calculate_damping(self, work_dir=None):
        """Calculate hydrodynamic properties using HydroPro"""
        # Create work directory if it doesn't exist
        if work_dir is None:
            work_dir = self.work_dir / self.name
        else:
            work_dir = Path(work_dir)
            
        try:
            os.listdir(work_dir)
        except:
            os.mkdir(work_dir)

        base_path = work_dir / "hydrocal"
        
        # Save the mesh in PDB format for HydroPro
        pdb_path = str(base_path) + ".pdb"
        self.save_as_pdb(pdb_path)
        
        # Run HydroPro to get hydrodynamic properties
        self.runner = HydroProRunner(
            self.mass,
            simconf=self.simconf,
            structure_name="hydrocal",
            inertia=list(self.principal_moments),
        )
        
        results = self.runner.run_calculation(work_dir=work_dir)
        
        self.transdamp = results['translation_damping']
        self.rotdamp = results['rotation_damping']
        
        logger.info(f"Translational damping: {self.transdamp}")
        logger.info(f"Rotational damping: {self.rotdamp}")
        
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
        
    def write_no_enter_potential_dx(self, output_file=None, **kwargs):
        if output_file is None:
            outfile=self.name+".dx"
        """Generate and write potential field to DX file"""
        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)

    def save_aligned_mesh(self, output_file):
        """Save the aligned mesh to a new .msh file"""
        gmsh.initialize()
        gmsh.model.add("aligned_mesh")
        
        # For surface mesh
        entity_dim = 2
        element_type = 2  # Triangle in gmsh
            
        # Add a discrete entity
        entity_tag = gmsh.model.addDiscreteEntity(entity_dim)
        
        # Add nodes
        node_tags = []
        node_coords = []
        for i, node in enumerate(self.nodes):
            tag = i + 1
            node_tags.append(tag)
            node_coords.extend(node / self.unit_scale)  # Convert back to mesh units
            
        gmsh.model.mesh.addNodes(entity_dim, entity_tag, node_tags, node_coords)
        
        # Add elements
        element_tags = []
        element_nodes = []
        for i, element in enumerate(self.elements):
            tag = i + 1
            element_tags.append(tag)
            element_nodes.extend([int(x + 1) for x in element])  # Convert to 1-based indexing
            
        gmsh.model.mesh.addElements(entity_dim, entity_tag, [element_type], [element_tags], [element_nodes])
        
        # Write mesh
        gmsh.write(str(output_file))
        gmsh.finalize()
            
    def save_as_pdb(self, output_file):

        """Save the aligned mesh as a PDB file (coordinates in Å)"""
        # PDB format specifications
        HEADER = "HEADER    ALIGNED SURFACE MESH                   "
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
                res_num = atom_num % 10000  # PDB format limits res numbers to 9999
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
        
        logger.info(f"Saved aligned mesh as PDB: {output_file}")
    
    def save_aligned_mesh_both_formats(self, base_filename):
        """Save the aligned mesh in both MSH and PDB formats"""
        # Save as MSH (in microns)
        msh_filename = f"{base_filename}.msh"
        self.save_aligned_mesh(msh_filename)
        logger.info(f"Saved aligned mesh as MSH: {msh_filename}")
        
        # Save as PDB (in Ångströms)
        pdb_filename = f"{base_filename}.pdb"
        self.save_as_pdb(pdb_filename)
        logger.info(f"Saved aligned mesh as PDB: {pdb_filename}")


def process_surface_mesh(mesh_file, density=19.3, temperature=303, viscosity=0.01,
                        solvent_density=1.0, output_dx="surface_potential.dx", 
                        output_mesh="aligned_surface.msh", binary_path=None, **kwargs):
    """
    Process surface mesh file and calculate all properties
    
    Args:
        mesh_file: Path to .msh file containing surface triangles
        density: Material density in g/cm³
        temperature: Temperature in Kelvin
        viscosity: Solvent viscosity in poise
        solvent_density: Solvent density in g/cm³
        output_dx: Optional path to output potential DX file
        output_mesh: Optional path to save aligned mesh
        binary_path: Path to HydroPro executable
        expected_mass: Optional expected mass in amu to calibrate calculations
        expected_aspect_ratio: Optional expected aspect ratio to calibrate inertia
        **kwargs: Additional arguments for potential generation
    """
    processor = SurfaceMeshProcessor(
        mesh_file,
        density=density,
        temperature=temperature,
        viscosity=viscosity,
        solvent_density=solvent_density,
        binary_path=binary_path,
    )
    
    # Calculate hydrodynamic properties
    processor.calculate_damping()
    
    logger.info(f"Mass: {processor.mass:.3f} amu")
    logger.info(f"Volume: {processor.volume:.3f} Å³")
    logger.info("\nPrincipal moments of inertia:")
    logger.info(processor.principal_moments)
    logger.info("\nTranslational damping coefficients [1/ns]:")
    logger.info(processor.transdamp)
    logger.info("\nRotational damping coefficients [1/ns]:")
    logger.info(processor.rotdamp)
    
    if output_dx:
        processor.write_no_enter_potential_dx()
        
    if output_mesh:
        processor.save_aligned_mesh(output_mesh)
        
    return processor

if __name__ == "__main__":
    process_surface_mesh("surface_mesh.msh")
