import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import subprocess
import os
import platform
from .grid import writeDx
from .runner import HydroProRunner
from . import logger

"""
Improved mesh processor with corrected inertia calculations based purely on mesh geometry
"""

class MeshProcessor:
    """Process gmsh files to calculate inertia, hydrodynamics and generate potential fields"""
    
    # Conversion factors
    MICRON_TO_ANGSTROM = 10000
    
    def __init__(self, mesh_file, density=19.3, temperature=295, viscosity=0.01, 
                solvent_density=1.0, unit_scale=MICRON_TO_ANGSTROM, work_dir = None,
                binary_path=None, expected_mass=None):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            density: Material density in g/cm^3
            temperature: Temperature in Kelvin
            viscosity: Solvent viscosity in poise
            solvent_density: Solvent density in g/cm3
            unit_scale: Conversion factor from input units to angstroms,default microns
            binary_path: Path to HydroPro binary
            expected_mass: Optional expected mass in amu to calibrate calculations
            expected_aspect_ratio: Optional expected aspect ratio to calibrate inertia
        """
        self.mesh_file = Path(mesh_file)
        self.density = density
        self.unit_scale = unit_scale
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.binary_path = binary_path
        self.name = str(self.mesh_file.stem)
        self.expected_mass = expected_mass
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        os.makedirs(self.work_dir, exist_ok=True)

        
        # Initialize gmsh and read mesh
        gmsh.initialize()
        try:
            gmsh.open(str(self.mesh_file))
            logger.info(f"Successfully opened mesh file: {self.mesh_file}")
            
            # First, load the full volumetric mesh for physical calculations
            self.volume_nodes = self._get_nodes()
            self.volume_elements = self._get_elements()
            logger.info(f"Loaded {len(self.volume_nodes)} nodes and {len(self.volume_elements)} elements from volume mesh")
            
            # Then extract surface mesh for visualization and HydroPro
            self.nodes, self.elements = self._extract_surface_mesh()
            logger.info(f"Extracted {len(self.nodes)} nodes and {len(self.elements)} elements from surface mesh")
            
            # Calculate volume using volumetric mesh
            self.volume = self._calculate_volume()
            
            # Calculate mass with density correction if needed
            if self.expected_mass:
                # Use expected mass if provided
                self.mass = self.expected_mass
                # Update density to match expected mass
                self.density_correction = self.expected_mass / (self.volume * 0.6022)
                logger.info(f"Using expected mass: {self.mass:.2f} amu (density correction: {self.density_correction:.6f})")
            else:
                # Convert density from g/cm^3 to amu/Å^3
                self.mass = self.volume * self.density * 0.6022
                self.density_correction = 1.0
                
            logger.info(f"Calculated volume: {self.volume:.2f} Å³")
            logger.info(f"Calculated mass: {self.mass:.2f} amu")
            
            # Align both meshes to center of mass and principal axes
            # Important: Center of mass calculation needs to be fixed to use the volumetric mesh
            self._align_mesh()
            
            # Calculate inertia tensor using volumetric mesh with corrected method
            self.inertia_tensor = self._calculate_correct_inertia_tensor()

            self.principal_moments = np.diag(self.inertia_tensor)
            logger.info(f"Principal moments of inertia: {self.principal_moments}")
            
        except Exception as e:
            logger.info(f"Error processing mesh: {e}")
            raise
        finally:
            gmsh.finalize()

    def _get_nodes(self):
        """Get all mesh nodes with unit conversion"""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        return coords * self.unit_scale  # Convert to angstroms
        
    def _get_elements(self):
        """Get tetrahedral elements from the volume mesh"""
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        if not element_types:  # No volume elements found
            logger.error("No tetrahedral elements found in mesh")
            
        # Find tetrahedral elements (type 4 in gmsh)
        tet_idx = None
        for i, type_num in enumerate(element_types):
            if type_num == 4:  # Tetrahedral elements
                tet_idx = i
                break
                
        if tet_idx is None:
            logger.error("No tetrahedral elements found in mesh")
            
        # Convert to 0-based indexing and reshape to Nx4 array
        elements = np.array(node_tags[tet_idx]).reshape(-1, 4) - 1
        return elements
    
    def _extract_surface_mesh(self):
        """Extract surface mesh from a 3D volumetric mesh"""
        # Get 3D elements
        dim3_entities = gmsh.model.getEntities(3)
        if not dim3_entities:
            logger.error("No 3D volume elements found in mesh")
            
        logger.info("Found 3D volumetric mesh, extracting surface...")
        
        # Create topology for boundary extraction
        gmsh.model.mesh.createTopology()
        
        # Get surface elements
        surface_dimtags = gmsh.model.getBoundary([(3, tag) for dim, tag in dim3_entities], 
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
            coords = gmsh.model.mesh.getNode(int(old_idx + 1))[0]  # +1 because gmsh uses 1-based indexing
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

    def _calculate_correct_inertia_tensor(self):
        """Calculate inertia tensor correctly using tetrahedral elements from the volume mesh"""
        # Initialize inertia tensor
        inertia = np.zeros((3, 3))
        
        # Set density (including any correction factors)
        if hasattr(self, 'density_correction'):
            effective_density = self.density * self.density_correction
        else:
            effective_density = self.density
            
        # Conversion factor from g/cm^3 to amu/Å^3
        density_amu = effective_density * 0.6022
        
        # For each tetrahedral element
        for element in self.volume_elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.volume_nodes[element]
            
            # Calculate volume of tetrahedron
            # V = (1/6) * |((v1-v0) × (v2-v0)) · (v3-v0)|
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            volume = abs(np.dot(np.cross(v01, v02), v03)) / 6.0
            
            # Mass of tetrahedron
            tetra_mass = volume * density_amu
            
            # Centroid of tetrahedron
            centroid = (v0 + v1 + v2 + v3) / 4.0
            
            # Tetrahedron inertia tensor calculation
            # For a tetrahedron, we need to integrate over the volume
            # See: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
            
            # Set up vertices relative to centroid
            vertices_centered = np.vstack([
                v0 - centroid,
                v1 - centroid,
                v2 - centroid,
                v3 - centroid
            ])
            
            # Compute the inertia tensor for tetrahedron about its centroid
            # We'll use the formula for a general tetrahedron
            local_inertia = np.zeros((3, 3))
            
            # Set up a matrix with columns for each coordinate
            x = vertices_centered[:, 0]
            y = vertices_centered[:, 1]
            z = vertices_centered[:, 2]
            
            # Calculate second moments
            xx = np.sum(x * x) / 10.0
            yy = np.sum(y * y) / 10.0
            zz = np.sum(z * z) / 10.0
            
            xy = np.sum(x * y) / 10.0
            xz = np.sum(x * z) / 10.0
            yz = np.sum(y * z) / 10.0
            
            # Fill in the local inertia tensor
            local_inertia[0, 0] = yy + zz
            local_inertia[1, 1] = xx + zz
            local_inertia[2, 2] = xx + yy
            
            local_inertia[0, 1] = local_inertia[1, 0] = -xy
            local_inertia[0, 2] = local_inertia[2, 0] = -xz
            local_inertia[1, 2] = local_inertia[2, 1] = -yz
            
            # Scale by mass
            local_inertia *= tetra_mass
            
            # Apply parallel axis theorem to get the inertia tensor about the origin
            # I = I_cm + m * (d^2 * I_identity - d ⊗ d)
            # where d is the distance vector from the origin to the centroid
            
            # Distance squared from origin to centroid
            d_squared = np.sum(centroid * centroid)
            
            # Dyadic product of centroid vector with itself
            d_dyadic = np.outer(centroid, centroid)
            
            # Parallel axis theorem
            parallel_axis_correction = tetra_mass * (d_squared * np.eye(3) - d_dyadic)
            
            # Add to total inertia
            inertia += local_inertia + parallel_axis_correction
            
        # Scale inertia tensor to match expected mass if provided
        if self.expected_mass:
            mass_scale = self.expected_mass / (self.volume * density_amu)
            inertia *= mass_scale
            
        # Verify that the inertia tensor is positive-definite
        eigenvalues = np.linalg.eigvalsh(inertia)
        if np.any(eigenvalues <= 0):
            logger.info("WARNING: Inertia tensor is not positive-definite! Eigenvalues:", eigenvalues)
            # Fix by making all eigenvalues positive
            inertia += np.eye(3) * abs(min(0, np.min(eigenvalues))) * 1.01
            
        return inertia

    def _calculate_volume(self):
        """Calculate volume using the tetrahedral elements from volume mesh"""
        # For tetrahedral mesh, calculate volume by summing volumes of all tetrahedra
        total_volume = 0
        
        for element in self.volume_elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.volume_nodes[element]
            
            # Calculate volume of tetrahedron
            # V = (1/6) * |((v1-v0) × (v2-v0)) · (v3-v0)|
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            volume = abs(np.dot(np.cross(v01, v02), v03)) / 6.0
            
            total_volume += volume
            
        return total_volume

    def _calculate_center_of_mass_volume(self):
        """Calculate center of mass using tetrahedral elements from volume mesh"""
        com = np.zeros(3)
        total_mass = 0
        
        for element in self.volume_elements:
            # Get vertices of tetrahedron
            v0, v1, v2, v3 = self.volume_nodes[element]
            
            # Calculate volume
            v01 = v1 - v0
            v02 = v2 - v0
            v03 = v3 - v0
            volume = abs(np.dot(np.cross(v01, v02), v03)) / 6.0
            
            # Mass of tetrahedron
            tetra_mass = volume * self.density * 0.6022  # Convert to amu
            
            # Centroid of tetrahedron
            centroid = (v0 + v1 + v2 + v3) / 4.0
            
            # Add weighted contribution
            com += centroid * tetra_mass
            total_mass += tetra_mass
        
        return com / total_mass if total_mass > 0 else np.zeros(3)

    def _align_mesh(self):
        """Align mesh to center of mass and principal axes with rod along Z-axis"""
        # First center the mesh using volume-based COM calculation
        com = self._calculate_center_of_mass_volume()
        self.volume_nodes -= com
        self.nodes -= com
        
        # Calculate inertia tensor for initial alignment
        inertia = self._calculate_correct_inertia_tensor()
        
        # Diagonalize to find principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Sort by eigenvalues
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        
        # For a rod-like object:
        # The smallest moment of inertia is along the rod axis
        # We want this to be aligned with the Z-axis (index 2)
        
        # Reordering eigenvectors: 
        # - smallest moment (rod axis) should be last (Z-axis)
        z_vec = eigenvectors[:, sort_idx[0]]  # Vector with smallest moment -> Z axis
        x_vec = eigenvectors[:, sort_idx[1]]
        y_vec = eigenvectors[:, sort_idx[2]]  # Vector with largest moment -> Y axis
        
        # Create new rotation matrix with correctly ordered eigenvectors
        rotation_matrix = np.column_stack([x_vec, y_vec, z_vec])
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1  # Flip X-axis
            
        # Apply rotation to both meshes
        self.volume_nodes = self.volume_nodes @ rotation_matrix
        self.nodes = self.nodes @ rotation_matrix
        
        # Store transformation
        self.rotation_matrix = rotation_matrix
        self.translation = com
        
        # Verify dimensions after alignment
        bounds_min = np.min(self.volume_nodes, axis=0)
        bounds_max = np.max(self.volume_nodes, axis=0)
        dimensions = bounds_max - bounds_min
        
        logger.info(f"Aligned mesh to principal axes, COM: {com}")
        logger.info(f"Dimensions after alignment (X,Y,Z): {dimensions}")
        
        # Calculate aspect ratio after alignment
        length = dimensions[2]  # Z dimension (rod axis)
        width = max(dimensions[0], dimensions[1])  # Larger of X,Y (transverse)
        aspect_ratio = length / width if width > 0 else 1.0
        logger.info(f"Calculated aspect ratio after alignment: {aspect_ratio:.2f}")

    def calculate_damping(self):
        """Calculate hydrodynamic properties using HydroPro"""
        # Create work directory if it doesn't exist
        work_dir = self.work_dir
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
            binary_path=self.binary_path,
            temperature=self.temperature,
            viscosity=self.viscosity,
            solvent_density=self.solvent_density,
            structure_name="hydrocal",cal_type="mesh"
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
        
    def write_no_enter_potential(self, output_file=None, **kwargs):
        """Generate and write potential field to DX file so particles do not fall inside rigidbody."""
        if output_file is None:
            output_file=self.work_dir/f"{self.name}-no-enter.dx"
        else:
            if not os.path.isabs(output_file):
                output_file = self.work_dir / output_file

        potential, origin, delta = self.generate_potential_grid(**kwargs)
        writeDx(output_file, potential, origin, delta)
        return output_file

    def save_aligned_mesh(self, output_file):
        if not os.path.isabs(output_file):
            output_file = self.work_dir / output_file

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
        if not os.path.isabs(output_file):
            output_file = self.work_dir / output_file
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
    
    def _save_aligned_mesh_both_formats(self, base_filename):
        """Save the aligned mesh in both MSH and PDB formats"""
        if not os.path.isabs(base_filename):
            base_filename = self.work_dir / base_filename
        # Save as MSH (in microns)
        msh_filename = f"{base_filename}.msh"
        self.save_aligned_mesh(msh_filename)
        logger.info(f"Saved aligned mesh as MSH: {msh_filename}")
        
        # Save as PDB (in Ångströms)
        pdb_filename = f"{base_filename}.pdb"
        self.save_as_pdb(pdb_filename)
        logger.info(f"Saved aligned mesh as PDB: {pdb_filename}")
    

def process_mesh_file(mesh_file, density=19.3, temperature=295, viscosity=0.01,
                     solvent_density=1.0, output_dx=None, 
                     output_mesh=None, binary_path=None,
                     expected_mass=None, **kwargs):
    """
    Process mesh file and calculate all properties
    
    Args:
        mesh_file: Path to .msh file
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
    processor = MeshProcessor(
        mesh_file,
        density=density,
        temperature=temperature,
        viscosity=viscosity,
        solvent_density=solvent_density,
        binary_path=binary_path,
        expected_mass=expected_mass,)
    
    # Calculate hydrodynamic properties
    processor.calculate_damping()
    
    logger.info("\nTranslational damping coefficients [1/ns]:")
    logger.info(processor.transdamp)
    logger.info("\nRotational damping coefficients [1/ns]:")
    logger.info(processor.rotdamp)
        
    return processor
