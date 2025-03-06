import numpy as np
import gmsh
from scipy.spatial import KDTree
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

class BeadModelDiffusion:
    """
    Calculate diffusion coefficients for a mesh using the bead shell model approach.
    This provides more accurate hydrodynamic properties than ellipsoid approximations,
    especially for complex, non-ellipsoidal shapes.
    """
    
    # Physical constants
    KB = 1.380649e-23  # Boltzmann constant in J/K
    NA = 6.02214076e23  # Avogadro's number
    MICRON_TO_ANGSTROM = 10000  # 1 micron = 10,000 Å
    CM3_TO_ANGSTROM3 = 1e24  # 1 cm³ = 10^24 Å³
    
    def __init__(self, mesh_file=None, nodes=None, elements=None, 
                 temperature=295, viscosity=0.01,
                 unit_scale=MICRON_TO_ANGSTROM, density=1.0):
        """
        Initialize the bead model calculator with either a mesh file or node/element data.
        
        Args:
            mesh_file: Path to .msh file (optional if nodes/elements provided)
            nodes: Array of node coordinates (optional if mesh_file provided)
            elements: Array of element connectivity (optional if mesh_file provided)
            temperature: Temperature in Kelvin (default: 295K)
            viscosity: Solvent viscosity in poise (default: 0.01 poise, water)
            unit_scale: Conversion factor from mesh units to angstroms
            density: Material density in g/cm³ (default: 1.0, water)
        """
        self.temperature = temperature
        self.viscosity = viscosity * 0.1  # Convert poise to Pa·s
        self.unit_scale = unit_scale 
        
        # Convert density from g/cm³ to amu/Å³
        density_conversion = self.NA / self.CM3_TO_ANGSTROM3
        self.density = density * density_conversion
        
        # Thermal energy
        self.kBT = self.KB * self.temperature
        
        if mesh_file is not None:
            # Load mesh from file
            self.mesh_file = Path(mesh_file)
            self.nodes, self.elements = self._load_mesh(self.mesh_file)
            print(f"Loaded mesh with {len(self.nodes)} nodes and {len(self.elements)} elements")
        elif nodes is not None and elements is not None:
            # Use provided node and element data
            self.nodes = np.array(nodes)
            self.elements = np.array(elements)
            print(f"Using provided mesh with {len(self.nodes)} nodes and {len(self.elements)} elements")
        else:
            raise ValueError("Either mesh_file or both nodes and elements must be provided")
            
        # Calculate mesh properties
        self.surface_area = self._calculate_surface_area()
        self.volume = self._estimate_volume()
        self.mass = self.volume * self.density
        
        # Calculate and store center of mass
        self.com = self._calculate_center_of_mass()
        
        # Center the mesh at COM
        self.nodes_centered = self.nodes - self.com
        
        print(f"Surface area: {self.surface_area:.2f} Å²")
        print(f"Estimated volume: {self.volume:.2f} Å³")
        print(f"Mass: {self.mass:.2f} amu")
        print(f"Center of mass: {self.com}")
    
    def _load_mesh(self, mesh_file):
        """Load mesh from a .msh file using gmsh"""
        gmsh.initialize()
        try:
            gmsh.open(str(mesh_file))
            
            # Get mesh elements - focusing on triangular elements (type 2)
            element_types = gmsh.model.mesh.getElementTypes()
            triangle_type = None
            for etype in element_types:
                if gmsh.model.mesh.getElementProperties(etype)[0] == "Triangle":
                    triangle_type = etype
                    break
                    
            if triangle_type is None:
                raise ValueError("No triangular elements found in mesh")
                
            # Get triangular elements
            element_tags, element_node_tags = gmsh.model.mesh.getElementsByType(triangle_type)
            num_nodes_per_elem = 3  # Triangles have 3 nodes
            elements = np.array(element_node_tags).reshape(-1, num_nodes_per_elem) - 1  # Convert to 0-based
            
            # Get all node coordinates
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes_dict = {}
            for i, tag in enumerate(node_tags):
                nodes_dict[tag] = node_coords[3*i:3*i+3]
                
            # Create nodes array matching element connectivity
            unique_node_tags = sorted(list(set(element_node_tags)))
            nodes = np.array([nodes_dict[tag] for tag in unique_node_tags])
            
            # Remap element indices to match the new node array
            node_mapping = {tag: i for i, tag in enumerate(unique_node_tags)}
            remapped_elements = []
            for elem in elements:
                remapped_elements.append([node_mapping[tag+1] for tag in elem])
                
            # Scale coordinates to angstroms
            nodes = np.array(nodes) * self.unit_scale
            elements = np.array(remapped_elements)
            
            return nodes, elements
            
        finally:
            gmsh.finalize()
    
    def _calculate_surface_area(self):
        """Calculate the total surface area of the mesh"""
        total_area = 0
        for element in self.elements:
            v = self.nodes[element]
            # Area of triangle using cross product
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            total_area += area
        return total_area
    
    def _estimate_volume(self):
        """
        Estimate the volume of the mesh using signed volumes of tetrahedra
        formed by triangles and the origin
        """
        total_volume = 0
        for element in self.elements:
            v = self.nodes[element]
            # Calculate signed volume of tetrahedron
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            normal = np.cross(edge1, edge2)
            volume = abs(np.dot(normal, v[0])) / 6.0
            total_volume += volume
        return total_volume
    
    def _calculate_center_of_mass(self):
        """Calculate center of mass of the mesh weighted by triangle areas"""
        com = np.zeros(3)
        total_weight = 0
        
        for element in self.elements:
            v = self.nodes[element]
            # Calculate area (weight)
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            
            # Centroid of triangle
            centroid = (v[0] + v[1] + v[2]) / 3.0
            
            # Add weighted contribution
            com += centroid * area
            total_weight += area
            
        return com / total_weight if total_weight > 0 else np.zeros(3)
    
    def _generate_bead_positions(self, bead_radius, coverage_factor=1.0, method='centroids'):
        """
        Generate positions for beads covering the mesh surface
        
        Args:
            bead_radius: Radius of each bead in Angstroms
            coverage_factor: Controls density of beads (higher means more beads)
            method: Method for generating bead positions:
                   'centroids' - Place beads at triangle centroids
                   'vertices' - Place beads at mesh vertices
                   'uniform' - Generate more uniform distribution using triangle areas
                   
        Returns:
            Array of bead positions
        """
        # Estimate number of beads based on surface area
        bead_area = np.pi * bead_radius**2
        target_beads = int(coverage_factor * self.surface_area / bead_area)
        
        if method == 'vertices':
            # Use mesh vertices (might not be ideal for uneven meshes)
            if target_beads >= len(self.nodes):
                return self.nodes_centered.copy()
            else:
                # Subsample vertices if we need fewer beads
                indices = np.random.choice(len(self.nodes), size=target_beads, replace=False)
                return self.nodes_centered[indices]
                
        elif method == 'centroids':
            # Use triangle centroids
            centroids = []
            weights = []
            
            for element in self.elements:
                v = self.nodes_centered[element]
                centroid = np.mean(v, axis=0)
                centroids.append(centroid)
                
                # Weight by triangle area for more uniform coverage
                edge1 = v[1] - v[0]
                edge2 = v[2] - v[0]
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                weights.append(area)
            
            centroids = np.array(centroids)
            
            if target_beads >= len(centroids):
                return centroids
            else:
                # Subsample centroids if we need fewer beads
                weights = np.array(weights) / sum(weights)
                indices = np.random.choice(len(centroids), size=target_beads, p=weights, replace=False)
                return centroids[indices]
                
        elif method == 'uniform':
            # More uniform distribution using weighted sampling within triangles
            positions = []
            areas = []
            
            # Calculate areas for each triangle
            for element in self.elements:
                v = self.nodes_centered[element]
                edge1 = v[1] - v[0]
                edge2 = v[2] - v[0]
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                areas.append(area)
            
            # Normalize areas
            total_area = sum(areas)
            areas = np.array(areas) / total_area
            
            # Calculate how many beads to allocate to each triangle
            num_beads_per_triangle = np.round(target_beads * areas).astype(int)
            
            # Ensure we have exactly target_beads
            diff = target_beads - sum(num_beads_per_triangle)
            if diff > 0:
                # Add remaining beads to largest triangles
                indices = np.argsort(areas)[-diff:]
                for idx in indices:
                    num_beads_per_triangle[idx] += 1
            elif diff < 0:
                # Remove extra beads from smallest triangles with at least 1 bead
                indices = np.where(num_beads_per_triangle > 0)[0]
                indices = indices[np.argsort(areas[indices])][:-diff]
                for idx in indices:
                    num_beads_per_triangle[idx] -= 1
            
            # Generate points within each triangle
            for i, element in enumerate(self.elements):
                n_beads = num_beads_per_triangle[i]
                if n_beads == 0:
                    continue
                    
                v = self.nodes_centered[element]
                
                # Generate random barycentric coordinates
                for _ in range(n_beads):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    # Ensure the sum is <= 1
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                        
                    # Third coordinate
                    r3 = 1 - r1 - r2
                    
                    # Convert to position
                    pos = r1 * v[0] + r2 * v[1] + r3 * v[2]
                    positions.append(pos)
            
            return np.array(positions)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_rpy_tensor(self, r_vec, a):
        """
        Calculate the Rotne-Prager-Yamakawa tensor for hydrodynamic interactions
        
        Args:
            r_vec: Vector between two beads
            a: Bead radius
            
        Returns:
            3x3 RPY tensor
        """
        r = np.linalg.norm(r_vec)
        
        if r == 0:
            # Self-diffusion
            return np.eye(3)
            
        r_hat = r_vec / r
        r_outer = np.outer(r_hat, r_hat)
        
        if r >= 2*a:
            # Non-overlapping beads
            return (3*a)/(4*r) * (np.eye(3) + r_outer) + \
                   (a/r)**3 * (np.eye(3) - 3*r_outer)/2
        else:
            # Overlapping beads
            return (1 - 9*r/(32*a)) * np.eye(3) + \
                   (3*r)/(32*a) * r_outer
    
    def _calculate_diffusion_matrix(self, bead_positions, bead_radius):
        """
        Calculate the diffusion matrix for the bead model
        
        Args:
            bead_positions: Array of bead positions (centered around COM)
            bead_radius: Radius of each bead in Angstroms
            
        Returns:
            NxN diffusion matrix (N = 3 * number of beads)
        """
        n_beads = len(bead_positions)
        print(f"Calculating diffusion matrix for {n_beads} beads...")
        
        # Convert units
        a = bead_radius * 1e-10  # Bead radius in meters
        
        # Stokes-Einstein diffusion coefficient for a single bead
        D0 = self.kBT / (6 * np.pi * self.viscosity * a)
        
        # Initialize diffusion matrix
        D_matrix = np.zeros((3*n_beads, 3*n_beads))
        
        # Fill the matrix using the RPY tensor
        for i in range(n_beads):
            for j in range(i, n_beads):  # Exploit symmetry
                if i == j:
                    # Self-diffusion
                    D_matrix[3*i:3*i+3, 3*j:3*j+3] = D0 * np.eye(3)
                else:
                    # Hydrodynamic coupling
                    r_vec = bead_positions[i] - bead_positions[j]
                    r_vec_m = r_vec * 1e-10  # Convert to meters
                    
                    # RPY tensor
                    hij = self._calculate_rpy_tensor(r_vec_m, a)
                    
                    # Diffusion tensor block
                    block = D0 * hij
                    
                    # Fill both blocks (matrix is symmetric)
                    D_matrix[3*i:3*i+3, 3*j:3*j+3] = block
                    D_matrix[3*j:3*j+3, 3*i:3*i+3] = block
        
        return D_matrix
    
    def _extract_diffusion_tensors(self, D_matrix, bead_positions):
        """
        Extract translational and rotational diffusion tensors from the grand diffusion matrix
        
        Args:
            D_matrix: Grand diffusion matrix (3N x 3N)
            bead_positions: Array of bead positions centered at COM
            
        Returns:
            D_tt: Translational diffusion tensor (3x3)
            D_rr: Rotational diffusion tensor (3x3)
        """
        n_beads = len(bead_positions)
        
        # Create projection matrices
        S = np.zeros((3*n_beads, 3))  # Translational projection
        R = np.zeros((3*n_beads, 3))  # Rotational projection
        
        for i in range(n_beads):
            # Translation projection (uniform)
            S[3*i:3*i+3, :] = np.eye(3) / n_beads
            
            # Rotation projection (cross product operator)
            r = bead_positions[i]
            r_cross = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            
            R[3*i:3*i+3, :] = r_cross
        
        # Extract diffusion tensors
        D_tt = S.T @ D_matrix @ S  # Translation-translation
        D_rr = R.T @ D_matrix @ R  # Rotation-rotation
        D_tr = S.T @ D_matrix @ R  # Translation-rotation
        D_rt = R.T @ D_matrix @ S  # Rotation-translation
        
        return D_tt, D_rr, D_tr, D_rt
    
    def _apply_principal_axis_transform(self, tensor):
        """
        Transform a tensor to get principal values and axes
        
        Args:
            tensor: 3x3 tensor
            
        Returns:
            eigenvalues: Principal values
            eigenvectors: Principal axes
        """
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        
        # Ensure eigenvalues are in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] = -eigenvectors[:, 0]
            
        return eigenvalues, eigenvectors
    
    def calculate_diffusion(self, bead_radius=5.0, coverage_factor=1.0, method='uniform'):
        """
        Calculate diffusion coefficients using the bead model
        
        Args:
            bead_radius: Radius of each bead in Angstroms
            coverage_factor: Controls bead density (higher means more beads)
            method: Method for generating bead positions
            
        Returns:
            dict with diffusion coefficients and related data
        """
        print(f"Calculating diffusion using {method} bead placement method with radius {bead_radius}Å")
        
        # Generate bead positions
        bead_positions = self._generate_bead_positions(bead_radius, coverage_factor, method)
        print(f"Generated {len(bead_positions)} beads")
        
        # Calculate diffusion matrix
        D_matrix = self._calculate_diffusion_matrix(bead_positions, bead_radius)
        
        # Extract diffusion tensors
        D_tt, D_rr, D_tr, D_rt = self._extract_diffusion_tensors(D_matrix, bead_positions)
        
        # Get principal components
        D_trans_vals, D_trans_axes = self._apply_principal_axis_transform(D_tt)
        D_rot_vals, D_rot_axes = self._apply_principal_axis_transform(D_rr)
        
        # Convert to conventional units
        D_trans_si = D_trans_vals  # m²/s
        D_rot_si = D_rot_vals  # rad²/s
        
        # Convert to ARBD units
        D_trans_arbd = D_trans_si * 10  # Å²/ns
        D_rot_arbd = D_rot_si * 1e-9  # rad²/ns
        
        # Calculate damping coefficients
        trans_damping = 1/D_trans_arbd  # ns/Å²
        rot_damping = 1/D_rot_arbd  # ns
        
        # Calculate average values
        D_trans_avg = np.mean(D_trans_si)
        D_rot_avg = np.mean(D_rot_si)
        
        # Compare with expected values (for the nanorod example)
        expected_trans = 6.87e-12  # m²/s
        expected_rot = 6000  # rad²/s
        
        trans_ratio = D_trans_avg / expected_trans
        rot_ratio = D_rot_avg / expected_rot
        
        trans_status = "GOOD - within expected range" if 0.5 < trans_ratio < 2.0 else f"DIFFERS from expected by factor of {trans_ratio:.1f}"
        rot_status = "GOOD - within expected range" if 0.5 < rot_ratio < 2.0 else f"DIFFERS from expected by factor of {rot_ratio:.1f}"
        
        # Print results
        print("\nDiffusion coefficients using bead model:")
        print(f"Translational diffusion [m²/s]: {D_trans_si}")
        print(f"Average translational diffusion: {D_trans_avg:.2e} m²/s ({trans_status})")
        print(f"Rotational diffusion [rad²/s]: {D_rot_si}")
        print(f"Average rotational diffusion: {D_rot_avg:.2e} rad²/s ({rot_status})")
        
        print(f"\nARBD-compatible damping coefficients:")
        print(f"Translational damping [ns/Å²]: {trans_damping}")
        print(f"Rotational damping [ns]: {rot_damping}")
        
        # Store all results
        results = {
            "bead_radius": bead_radius,
            "coverage_factor": coverage_factor,
            "method": method,
            "num_beads": len(bead_positions),
            "bead_positions": bead_positions,
            "D_matrix": D_matrix,
            "D_tt": D_tt,
            "D_rr": D_rr,
            "D_tr": D_tr,
            "D_rt": D_rt,
            "D_trans_si": D_trans_si,
            "D_rot_si": D_rot_si,
            "D_trans_arbd": D_trans_arbd,
            "D_rot_arbd": D_rot_arbd,
            "trans_damping": trans_damping,
            "rot_damping": rot_damping,
            "D_trans_axes": D_trans_axes,
            "D_rot_axes": D_rot_axes
        }
        
        return results
    
    def visualize_beads(self, bead_positions, bead_radius, show_mesh=True, fig=None, ax=None):
        """
        Visualize the bead model and optionally the underlying mesh
        
        Args:
            bead_positions: Array of bead positions
            bead_radius: Radius of beads (for visualization)
            show_mesh: Whether to show the mesh
            fig, ax: Optional existing figure and axis
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot beads as spheres with reduced complexity
        for pos in bead_positions:
            # Create a small sphere
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = pos[0] + bead_radius * np.outer(np.cos(u), np.sin(v))
            y = pos[1] + bead_radius * np.outer(np.sin(u), np.sin(v))
            z = pos[2] + bead_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot the sphere
            ax.plot_surface(x, y, z, color='c', alpha=0.3)
        
        if show_mesh:
            # Plot the mesh as wireframe
            for element in self.elements:
                vertices = self.nodes_centered[element]
                vertices = np.vstack([vertices, vertices[0]])  # Close the triangle
                ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-', lw=0.5, alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Bead Model ({len(bead_positions)} beads, r={bead_radius}Å)')
        
        # Set equal aspect ratio
        max_range = np.max([
            np.ptp(self.nodes_centered[:, 0]),
            np.ptp(self.nodes_centered[:, 1]),
            np.ptp(self.nodes_centered[:, 2])
        ])
        mid_x = np.mean(self.nodes_centered[:, 0])
        mid_y = np.mean(self.nodes_centered[:, 1])
        mid_z = np.mean(self.nodes_centered[:, 2])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        return fig, ax
    
    def compare_methods(self, bead_radius=5.0, coverage_factors=[0.5, 1.0, 2.0], 
                        methods=['vertices', 'centroids', 'uniform'], visualize=True):
        """
        Compare different bead placement methods and coverage factors
        
        Args:
            bead_radius: Radius of beads in Angstroms
            coverage_factors: List of coverage factors to test
            methods: List of bead placement methods to test
            visualize: Whether to create visualization plots
            
        Returns:
            DataFrame with comparison results
        """
        import pandas as pd
        
        results = []
        
        for method in methods:
            for coverage in coverage_factors:
                print(f"\nTesting {method} method with coverage factor {coverage}")
                
                # Calculate diffusion
                diffusion = self.calculate_diffusion(bead_radius, coverage, method)
                
                # Store key results
                result = {
                    "Method": method,
                    "Coverage": coverage,
                    "Num_Beads": diffusion["num_beads"],
                    "D_trans_avg (m²/s)": np.mean(diffusion["D_trans_si"]),
                    "D_trans_x (m²/s)": diffusion["D_trans_si"][0],
                    "D_trans_y (m²/s)": diffusion["D_trans_si"][1],
                    "D_trans_z (m²/s)": diffusion["D_trans_si"][2],
                    "D_rot_avg (rad²/s)": np.mean(diffusion["D_rot_si"]),
                    "D_rot_x (rad²/s)": diffusion["D_rot_si"][0],
                    "D_rot_y (rad²/s)": diffusion["D_rot_si"][1],
                    "D_rot_z (rad²/s)": diffusion["D_rot_si"][2]
                }
                
                results.append(result)
                
                # Visualize beads if requested
                if visualize:
                    fig, ax = self.visualize_beads(diffusion["bead_positions"], bead_radius)
                    plt.tight_layout()
                    plt.savefig(f"beads_{method}_coverage{coverage}.png")
                    plt.close(fig)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate expected value ratios
        expected_trans = 6.87e-12  # m²/s
        expected_rot = 6000  # rad²/s
        
        df["Trans_Ratio"] = df["D_trans_avg (m²/s)"] / expected_trans
        df["Rot_Ratio"] = df["D_rot_avg (rad²/s)"] / expected_rot
        
        return df

    def find_optimal_parameters(self, bead_radii=[3.0, 5.0, 7.0], 
                              coverage_factors=[0.5, 1.0, 1.5, 2.0],
                              method='uniform'):
        """
        Find optimal bead parameters to match expected diffusion values
        
        Args:
            bead_radii: List of bead radii to test
            coverage_factors: List of coverage factors to test
            method: Bead placement method
            
        Returns:
            Tuple of optimal radius and coverage factor
        """
        expected_trans = 6.87e-12  # m²/s
        expected_rot = 6000  # rad²/s
        
        best_params = None
        best_error = float('inf')
        
        for radius in bead_radii:
            for coverage in coverage_factors:
                print(f"\nTesting radius {radius}Å with coverage {coverage}")
                
                # Calculate diffusion
                diffusion = self.calculate_diffusion(radius, coverage, method)
                
                # Calculate average values
                D_trans_avg = np.mean(diffusion["D_trans_si"])
                D_rot_avg = np.mean(diffusion["D_rot_si"])
                
                # Calculate error (normalized distance from expected values)
                trans_error = abs(D_trans_avg - expected_trans) / expected_trans
                rot_error = abs(D_rot_avg - expected_rot) / expected_rot
                total_error = trans_error + rot_error
                
                print(f"Error: {total_error:.4f} (trans: {trans_error:.4f}, rot: {rot_error:.4f})")
                
                if total_error < best_error:
                    best_error = total_error
                    best_params = (radius, coverage, D_trans_avg, D_rot_avg, total_error)
        
        if best_params:
            radius, coverage, D_trans, D_rot, error = best_params
            print(f"\nOptimal parameters: radius = {radius}Å, coverage = {coverage}")
            print(f"Resulting diffusion: D_trans = {D_trans:.2e} m²/s, D_rot = {D_rot:.2e} rad²/s")
            print(f"Error: {error:.4f}")
            
            return (radius, coverage)
    
    def save_results_to_file(self, results, filename="diffusion_results.txt"):
        """
        Save diffusion results to a file
        
        Args:
            results: Results dictionary from calculate_diffusion
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("Bead Model Diffusion Results\n")
            f.write("============================\n\n")
            
            f.write(f"Method: {results['method']}\n")
            f.write(f"Bead radius: {results['bead_radius']} Å\n")
            f.write(f"Coverage factor: {results['coverage_factor']}\n")
            f.write(f"Number of beads: {results['num_beads']}\n\n")
            
            f.write("Translational Diffusion Coefficients (SI units):\n")
            f.write(f"Dx = {results['D_trans_si'][0]:.6e} m²/s\n")
            f.write(f"Dy = {results['D_trans_si'][1]:.6e} m²/s\n")
            f.write(f"Dz = {results['D_trans_si'][2]:.6e} m²/s\n")
            f.write(f"Average = {np.mean(results['D_trans_si']):.6e} m²/s\n\n")
            
            f.write("Rotational Diffusion Coefficients (SI units):\n")
            f.write(f"Drx = {results['D_rot_si'][0]:.6e} rad²/s\n")
            f.write(f"Dry = {results['D_rot_si'][1]:.6e} rad²/s\n")
            f.write(f"Drz = {results['D_rot_si'][2]:.6e} rad²/s\n")
            f.write(f"Average = {np.mean(results['D_rot_si']):.6e} rad²/s\n\n")
            
            f.write("ARBD-Compatible Coefficients:\n")
            f.write("Translational Damping Coefficients:\n")
            f.write(f"Tx = {results['trans_damping'][0]:.6e} ns/Å²\n")
            f.write(f"Ty = {results['trans_damping'][1]:.6e} ns/Å²\n")
            f.write(f"Tz = {results['trans_damping'][2]:.6e} ns/Å²\n\n")
            
            f.write("Rotational Damping Coefficients:\n")
            f.write(f"Rx = {results['rot_damping'][0]:.6e} ns\n")
            f.write(f"Ry = {results['rot_damping'][1]:.6e} ns\n")
            f.write(f"Rz = {results['rot_damping'][2]:.6e} ns\n\n")
            
            f.write("Translational Diffusion Principal Axes:\n")
            for i in range(3):
                f.write(f"Axis {i+1}: {results['D_trans_axes'][:,i]}\n")
            f.write("\n")
            
            f.write("Rotational Diffusion Principal Axes:\n")
            for i in range(3):
                f.write(f"Axis {i+1}: {results['D_rot_axes'][:,i]}\n")
        
        print(f"Results saved to {filename}")
