import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import os
import sys
from .grid import writeDx


class MeshProcessor:
    """Process 3D meshes to calculate inertia, diffusion coefficients and generate potential fields"""
    
    # Conversion factors for units
    MICRON_TO_ANGSTROM = 10000  # 1 micron = 10,000 Å
    AMU_PER_GRAM = 6.02214076e23  # Avogadro's number
    CM3_TO_ANGSTROM3 = 1e24  # 1 cm³ = 10^24 Å³
    
    def __init__(self, mesh_file, temperature=295, viscosity=0.01, density=1.0, 
                 unit_scale=MICRON_TO_ANGSTROM, clustering_method='distance', num_components=3):
        """
        Initialize processor with mesh file
        
        Args:
            mesh_file: Path to .msh file
            temperature: Temperature in Kelvin (default: 295K)
            viscosity: Solvent viscosity in poise (default: 0.01 poise, water)
            density: Material density in g/cm³ (default: 1.0, water)
            unit_scale: Conversion factor from mesh units to angstroms (default: 10,000 for microns)
            clustering_method: Method for mesh decomposition ('distance' or 'curvature')
            num_components: Number of components for mesh decomposition (default: 3)
        """
        self.mesh_file = Path(mesh_file)
        self.temperature = temperature
        self.viscosity = viscosity
        self.unit_scale = unit_scale
        self.clustering_method = clustering_method
        self.num_components = num_components
        
        # Physical constants
        self.kB = 1.380649e-23  # Boltzmann constant in J/K
        self.kBT = self.kB * self.temperature  # Thermal energy
        
        # Initialize gmsh and read mesh
        gmsh.initialize()
        try:
            gmsh.open(str(self.mesh_file))
            print(f"Successfully opened mesh file: {self.mesh_file}")
            
            # Extract surface mesh from 3D mesh
            self.nodes, self.elements = self._extract_surface_mesh()
            print(f"Extracted {len(self.nodes)} nodes and {len(self.elements)} elements from surface mesh")
            
            # Calculate density in amu/Å³
            # Convert from g/cm³ to amu/Å³: (g/cm³) * (amu/g) / (Å³/cm³)
            density_conversion = self.AMU_PER_GRAM / self.CM3_TO_ANGSTROM3
            self.density = density * density_conversion
            print(f"Material density: {density} g/cm³ = {self.density:.6e} amu/Å³")
            
            # Calculate basic properties
            self.volume = self._calculate_volume()
            self.mass = self.volume * self.density
            print(f"Calculated volume: {self.volume:.2f} Å³")
            print(f"Calculated mass: {self.mass:.2f} amu")
            
            # Align mesh to principal axes
            self._align_mesh()
            basename = self.mesh_file.stem + "_aligned"
            self._save_aligned_mesh_both_formats(basename)
            # Calculate inertia tensor after alignment
            self.inertia_tensor = self._calculate_inertia_tensor()
            self.principal_moments = np.diag(self.inertia_tensor)
            print(f"Principal moments of inertia: {self.principal_moments}")
            
            # Get ellipsoid parameters for hydrodynamic calculations
            self.semi_axes, self.shape_type = self._get_ellipsoid_parameters()
            self.semi_axes = np.array([761/2, 223/2, 223/2])  # L/2, d/2, d/2 in nm
            print(f"Shape classification: {self.shape_type}")
            print(f"Semi-axes of equivalent ellipsoid: a={self.semi_axes[0]:.1f} Å, "
                  f"b={self.semi_axes[1]:.1f} Å, c={self.semi_axes[2]:.1f} Å")
            
            # Calculate friction coefficients using shape
            #self._calculate_diffusion()
            
        except Exception as e:
            print(f"Error processing mesh: {e}")
            raise
        finally:
            gmsh.finalize()

    def _extract_surface_mesh(self):
        """Extract surface mesh from a 3D volumetric mesh"""
        # First check if there are 3D elements
        dim3_entities = gmsh.model.getEntities(3)
        
        if dim3_entities:  # There are 3D elements, extract surface
            print("Found 3D volumetric mesh, extracting surface...")
            
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
            
        else:  # No 3D elements, assume it's already a surface mesh
            print("No 3D elements found, assuming surface mesh...")
            
            # Get all 2D elements
            dim2_entities = gmsh.model.getEntities(2)
            if not dim2_entities:
                raise ValueError("No 3D volume elements or 2D surface elements found in mesh")
                
            # Get nodes and elements from 2D entities
            all_nodes = {}
            all_elements = []
            
            for dim, tag in dim2_entities:
                element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
                
                # We only want triangular elements (type 2)
                tri_idx = None
                for i, type_num in enumerate(element_types):
                    if type_num == 2:  # Triangle elements
                        tri_idx = i
                        break
                        
                if tri_idx is not None:
                    # Get element connectivity
                    elems = np.array(node_tags[tri_idx]).reshape(-1, 3) - 1  # Convert to 0-based
                    all_elements.extend(elems)
                    
                    # Add all nodes to dict
                    for node_tag in np.unique(elems):
                        if node_tag not in all_nodes:
                            all_nodes[node_tag] = gmsh.model.mesh.getNode(node_tag + 1)[0]
            
            # Create arrays from dictionaries
            node_idx_map = {}
            nodes = []
            
            for i, (old_idx, coords) in enumerate(sorted(all_nodes.items())):
                nodes.append(coords)
                node_idx_map[old_idx] = i
                
            # Remap element indices
            remapped_elements = []
            for element in all_elements:
                remapped_elements.append([node_idx_map.get(idx, idx) for idx in element])
                
            # Convert to numpy arrays and apply unit conversion
            nodes = np.array(nodes) * self.unit_scale
            elements = np.array(remapped_elements)
        
        return nodes, elements

    def _calculate_volume(self):
        """Calculate approximate volume of the surface mesh using triangulation"""
        # For a closed surface mesh, we can approximate volume using the divergence theorem
        total_volume = 0
        
        # For each triangular face
        for element in self.elements:
            # Get vertices of triangle
            v0, v1, v2 = self.nodes[element]
            
            # Calculate signed volume of tetrahedron formed by triangle and origin
            # V = (1/6) * |((v1-v0) × (v2-v0)) · v0|
            v01 = v1 - v0
            v02 = v2 - v0
            normal = np.cross(v01, v02)
            volume = abs(np.dot(normal, v0)) / 6.0
            
            total_volume += volume
            
        return total_volume

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
        """Calculate inertia tensor about center of mass for surface mesh"""
        inertia = np.zeros((3, 3))
        
        # For a surface mesh, approximate inertia using triangles
        for element in self.elements:
            # Get vertices of triangle
            v0, v1, v2 = self.nodes[element]
            
            # Calculate area
            v01 = v1 - v0
            v02 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(v01, v02))
            
            # Vertices contribution to inertia tensor
            vertices = np.vstack([v0, v1, v2])
            
            # Approximate inertia contribution based on area
            for i in range(3):
                for j in range(3):
                    if i == j:
                        # Diagonal terms
                        term = np.sum(vertices[:, (i+1)%3]**2 + vertices[:, (i+2)%3]**2) / 3.0
                    else:
                        # Off-diagonal terms
                        term = -np.sum(vertices[:, i] * vertices[:, j]) / 3.0
                        
                    inertia[i,j] += self.density * area * term
        
        return inertia

    def _align_mesh(self):
        """Align mesh to center of mass and principal axes"""
        # First center the mesh
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
            
        # Rotate mesh to align with principal axes
        self.nodes = self.nodes @ eigenvectors
        
        # Store transformation
        self.rotation_matrix = eigenvectors
        self.translation = com
        print(f"Aligned mesh to principal axes, COM: {com}")

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


def process_mesh_file(mesh_file, density=1.0, temperature=295, viscosity=0.01,
                     clustering_method='distance', num_components=3, 
                     output_dx="mesh_potential.dx", output_mesh="aligned_mesh", **kwargs):
    """
    Process a mesh file to calculate hydrodynamic properties and potential fields
    
    Args:
        mesh_file: Path to .msh file
        density: Material density in g/cm³
        temperature: Temperature in Kelvin
        viscosity: Viscosity in poise
        clustering_method: Method for mesh decomposition ('distance' or 'curvature')
        num_components: Number of components for mesh decomposition
        output_dx: Path to output potential DX file
        output_mesh: Base name for output aligned mesh files (will save both .msh and .pdb)
        **kwargs: Additional arguments for potential generation
        
    Returns:
        MeshProcessor instance
    """
    processor = MeshProcessor(mesh_file, temperature, viscosity, density, 
                             clustering_method=clustering_method, 
                             num_components=num_components)
    
    # Generate and save potential field if requested
    if output_dx:
        processor.write_potential_dx(output_dx, **kwargs)
        print(f"Potential field written to {output_dx}")
        
        
    return processor
