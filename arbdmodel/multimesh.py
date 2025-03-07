import numpy as np
import gmsh
from scipy.spatial import KDTree, distance
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

"""
Potential multimesh generation suggusted by Claude.
"""

class MultiBodyHydrodynamics:
    """
    Calculates hydrodynamic properties of complex shapes using a multi-body approximation.
    The shape is decomposed into multiple simpler components, and hydrodynamic interactions
    between these components are considered.
    """
    
    def __init__(self, mesh_processor, num_components=3, temperature=295, 
                 viscosity=0.01, clustering_method='distance'):
        """
        Initialize the multi-body approximation.
        
        Args:
            mesh_processor: A MeshProcessor instance containing the mesh data
            num_components: Number of components to decompose the mesh into
            temperature: Temperature in Kelvin
            viscosity: Viscosity in poise (will be converted to Pa·s)
            clustering_method: Method to use for mesh decomposition ('distance', 'curvature', or 'manual')
        """
        self.mesh_processor = mesh_processor
        self.num_components = num_components
        self.temperature = temperature
        self.viscosity = viscosity * 0.1  # Convert poise to Pa·s
        self.clustering_method = clustering_method
        
        # Store original mesh data
        self.nodes = mesh_processor.nodes
        self.elements = mesh_processor.elements
        self.mass = mesh_processor.mass
        
        # Initialize components
        self.component_nodes = []
        self.component_masses = []
        self.component_centers = []
        self.component_semi_axes = []
        self.component_rotations = []
        
        # Results
        self.translational_damping = None
        self.rotational_damping = None
        
    def decompose_mesh(self):
        """
        Decompose the mesh into multiple components based on the specified method.
        """
        if self.clustering_method == 'distance':
            self._decompose_by_distance()
        elif self.clustering_method == 'curvature':
            self._decompose_by_curvature()
        elif self.clustering_method == 'manual':
            # This would be implemented with user-defined regions
            raise NotImplementedError("Manual decomposition not yet implemented")
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
        # Calculate properties for each component
        self._calculate_component_properties()
        
    def _decompose_by_distance(self):
        """
        Decompose the mesh into components using hierarchical clustering based on distance.
        """
        print(f"Decomposing mesh into {self.num_components} components using distance-based clustering...")
        
        # Perform hierarchical clustering on node positions
        Z = linkage(self.nodes, method='ward')  # Ward minimizes the variance
        
        # Extract cluster labels
        labels = fcluster(Z, self.num_components, criterion='maxclust') - 1  # 0-based indexing
        
        # Organize nodes by component
        self.component_nodes = [[] for _ in range(self.num_components)]
        for i, label in enumerate(labels):
            self.component_nodes[label].append(self.nodes[i])
            
        # Convert to numpy arrays
        self.component_nodes = [np.array(nodes) for nodes in self.component_nodes]
        
    def _decompose_by_curvature(self):
        """
        Decompose the mesh into components based on surface curvature.
        More suitable for complex biological shapes.
        """
        print(f"Decomposing mesh into {self.num_components} components using curvature-based clustering...")
        
        # Calculate curvature for surface elements
        curvatures = self._estimate_curvatures()
        
        # Perform hierarchical clustering based on curvature and position
        # Weight position and curvature
        features = np.column_stack([self.nodes, curvatures * 100])  # Scale curvature for balance
        
        Z = linkage(features, method='ward')
        
        # Extract cluster labels
        labels = fcluster(Z, self.num_components, criterion='maxclust') - 1
        
        # Organize nodes by component
        self.component_nodes = [[] for _ in range(self.num_components)]
        for i, label in enumerate(labels):
            self.component_nodes[label].append(self.nodes[i])
            
        # Convert to numpy arrays
        self.component_nodes = [np.array(nodes) for nodes in self.component_nodes]
        
    def _estimate_curvatures(self):
        """
        Estimate local curvature for each vertex using local neighborhoods.
        """
        # Build KD-tree for efficient neighbor finding
        tree = KDTree(self.nodes)
        
        # Number of neighbors to consider
        k = 10
        
        curvatures = np.zeros(len(self.nodes))
        
        for i, point in enumerate(self.nodes):
            # Find k nearest neighbors
            distances, indices = tree.query(point, k=k+1)  # +1 because point itself is included
            neighbors = self.nodes[indices[1:]]  # Exclude the point itself
            
            # Compute local covariance matrix
            centered = neighbors - point
            cov = centered.T @ centered
            
            # Eigenvalues of covariance matrix relate to principal curvatures
            eigenvalues = np.linalg.eigvalsh(cov)
            
            # Estimate curvature as ratio of smallest to largest eigenvalue
            if eigenvalues[-1] > 0:
                curvatures[i] = eigenvalues[0] / eigenvalues[-1]
            else:
                curvatures[i] = 0
                
        return curvatures
    
    def _calculate_component_properties(self):
        """
        Calculate mass, center, and semi-axes for each component.
        """
        total_nodes = sum(len(nodes) for nodes in self.component_nodes)
        
        for i, nodes in enumerate(self.component_nodes):
            # Estimate component mass based on number of vertices
            component_mass = self.mass * len(nodes) / total_nodes
            self.component_masses.append(component_mass)
            
            # Calculate center of component
            center = np.mean(nodes, axis=0)
            self.component_centers.append(center)
            
            # Calculate gyration tensor for the component
            centered_nodes = nodes - center
            gyration_tensor = np.zeros((3, 3))
            for node in centered_nodes:
                gyration_tensor += np.outer(node, node)
            gyration_tensor /= len(nodes)
            
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(gyration_tensor)
            
            # Semi-axes in descending order (a, b, c)
            semi_axes = np.sqrt(5.0 * eigenvalues)[::-1]
            self.component_semi_axes.append(semi_axes)
            
            # Rotation matrix from eigenvectors
            # Ensure right-handed coordinate system
            eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 0] *= -1
            self.component_rotations.append(eigenvectors)
            
            print(f"Component {i+1}: Mass={component_mass:.2f} amu, "
                  f"Semi-axes=({semi_axes[0]:.2f}, {semi_axes[1]:.2f}, {semi_axes[2]:.2f}) Å")
    
    def _calculate_single_ellipsoid_damping(self, semi_axes, mass):
        """
        Calculate translational and rotational damping coefficients for a single ellipsoid.
        
        Args:
            semi_axes: Semi-axes lengths [a, b, c] in Å
            mass: Mass in amu
            
        Returns:
            trans_damping: Translational damping coefficients [gamma_a, gamma_b, gamma_c] in 1/ns
            rot_damping: Rotational damping coefficients in 1/ns
        """
        # Convert to meters
        a_m, b_m, c_m = semi_axes * 1e-10  # Å to m
        
        # Volume of ellipsoid
        volume = 4/3 * np.pi * a_m * b_m * c_m
        
        # Check shape type (tolerance for considering axes equal)
        tol = 0.05
        
        if np.isclose(a_m, b_m, rtol=tol) and np.isclose(b_m, c_m, rtol=tol):
            # Sphere case
            radius = (a_m + b_m + c_m) / 3
            
            # Translational friction coefficient
            gamma_trans = 6 * np.pi * self.viscosity * radius
            
            # Convert to ARBD units (1/ns)
            amu_to_kg = 1.66054e-27  # kg/amu
            mass_kg = mass * amu_to_kg
            trans_damping = np.array([gamma_trans, gamma_trans, gamma_trans]) / mass_kg * 1e9
            
            # Rotational friction coefficient
            gamma_rot = 8 * np.pi * self.viscosity * radius**3
            
            # Moment of inertia for solid sphere
            inertia = 2/5 * mass_kg * radius**2
            rot_damping = np.array([gamma_rot, gamma_rot, gamma_rot]) / inertia * 1e9
            
            return trans_damping, rot_damping
            
        elif np.isclose(b_m, c_m, rtol=tol):
            # Prolate ellipsoid (rod-like)
            e = np.sqrt(1 - (b_m/a_m)**2)  # Eccentricity
            
            # Shape factor
            if e > 0.99:
                S = 2 * np.log(2*a_m/b_m) - 0.5
            else:
                S = 2 * np.log((1 + e)/(1 - e)) / e - 2*e/(1 - e**2)
                
            # Translational friction coefficients
            gamma_a = 6 * np.pi * self.viscosity * b_m / S  # Along major axis
            gamma_bc = 6 * np.pi * self.viscosity * b_m / (0.5 * S + 1)  # Perpendicular
            
            # Convert to ARBD units
            amu_to_kg = 1.66054e-27
            mass_kg = mass * amu_to_kg
            trans_damping = np.array([gamma_a, gamma_bc, gamma_bc]) / mass_kg * 1e9
            
            # Moments of inertia
            I_a = 1/5 * mass_kg * (b_m**2 + c_m**2)
            I_bc = 1/5 * mass_kg * (a_m**2 + (b_m**2 + c_m**2)/2)
            
            # Rotational damping
            gamma_rot_a = 6 * self.viscosity * volume * (1 - e**2) / (e**2) * \
                          (-2*e/(1-e**2) + np.log((1+e)/(1-e)))
            gamma_rot_bc = 6 * self.viscosity * volume * (1 + e**2) / (e**2) * \
                           (2*e/(1-e**2) - (1-e**2)/(2*e) * np.log((1+e)/(1-e)))
            
            rot_damping = np.array([gamma_rot_bc, gamma_rot_a, gamma_rot_a]) / \
                          np.array([I_bc, I_a, I_a]) * 1e9
            
            return trans_damping, rot_damping
            
        elif np.isclose(a_m, b_m, rtol=tol):
            # Oblate ellipsoid (disk-like)
            e = np.sqrt(1 - (c_m/a_m)**2)
            
            # Shape factor
            if e > 0.99:
                S = np.pi * a_m / (2 * c_m)
            else:
                S = 2 * np.arctan(e/np.sqrt(1-e**2)) / (e * np.sqrt(1-e**2))
                
            # Translational friction
            gamma_ab = 6 * np.pi * self.viscosity * a_m / (1 + 0.5*S*(1-e**2)/e)
            gamma_c = 6 * np.pi * self.viscosity * a_m / (S*(1-e**2)/e)
            
            # Convert to ARBD units
            amu_to_kg = 1.66054e-27
            mass_kg = mass * amu_to_kg
            trans_damping = np.array([gamma_ab, gamma_ab, gamma_c]) / mass_kg * 1e9
            
            # Moments of inertia
            I_c = 1/5 * mass_kg * (a_m**2 + b_m**2)
            I_ab = 1/5 * mass_kg * (c_m**2 + (a_m**2 + b_m**2)/2)
            
            # Rotational damping
            gamma_rot_c = 6 * self.viscosity * volume * (2 - e**2) / (e**2) * \
                         (e/(1-e**2) - 0.5 * np.arctan(e/np.sqrt(1-e**2)) / (e * np.sqrt(1-e**2)))
            gamma_rot_ab = 6 * self.viscosity * volume * (2 + e**2) / (e**2) * \
                          (0.5 * np.arctan(e/np.sqrt(1-e**2)) / (e * np.sqrt(1-e**2)) - e/(1-e**2))
            
            rot_damping = np.array([gamma_rot_ab, gamma_rot_ab, gamma_rot_c]) / \
                          np.array([I_ab, I_ab, I_c]) * 1e9
            
            return trans_damping, rot_damping
            
        else:
            # Triaxial ellipsoid - use approximate formulas
            R_eq = (a_m * b_m * c_m)**(1/3)
            
            # Correction factors
            alpha_a = 1 - 0.25 * (1 - (a_m/R_eq)**(-2))
            alpha_b = 1 - 0.25 * (1 - (b_m/R_eq)**(-2))
            alpha_c = 1 - 0.25 * (1 - (c_m/R_eq)**(-2))
            
            # Translational friction
            gamma_a = 6 * np.pi * self.viscosity * R_eq / alpha_a
            gamma_b = 6 * np.pi * self.viscosity * R_eq / alpha_b
            gamma_c = 6 * np.pi * self.viscosity * R_eq / alpha_c
            
            # Convert to ARBD units
            amu_to_kg = 1.66054e-27
            mass_kg = mass * amu_to_kg
            trans_damping = np.array([gamma_a, gamma_b, gamma_c]) / mass_kg * 1e9
            
            # Moments of inertia
            I_a = 1/5 * mass_kg * (b_m**2 + c_m**2)
            I_b = 1/5 * mass_kg * (a_m**2 + c_m**2)
            I_c = 1/5 * mass_kg * (a_m**2 + b_m**2)
            
            # Approximate rotational damping
            beta_a = ((b_m**2 - c_m**2)/(b_m**2 + c_m**2))**2
            beta_b = ((a_m**2 - c_m**2)/(a_m**2 + c_m**2))**2
            beta_c = ((a_m**2 - b_m**2)/(a_m**2 + b_m**2))**2
            
            gamma_rot_a = 8 * np.pi * self.viscosity * (b_m**2 + c_m**2) / 3 * (1 + beta_a)
            gamma_rot_b = 8 * np.pi * self.viscosity * (a_m**2 + c_m**2) / 3 * (1 + beta_b)
            gamma_rot_c = 8 * np.pi * self.viscosity * (a_m**2 + b_m**2) / 3 * (1 + beta_c)
            
            rot_damping = np.array([gamma_rot_a, gamma_rot_b, gamma_rot_c]) / \
                          np.array([I_a, I_b, I_c]) * 1e9
            
            return trans_damping, rot_damping
    
    def _calculate_oseen_tensor(self, r_ij):
        """
        Calculate the Oseen tensor for hydrodynamic interactions between components.
        
        Args:
            r_ij: Distance vector between components i and j
            
        Returns:
            T_ij: 3x3 Oseen tensor
        """
        r = np.linalg.norm(r_ij)
        if r < 1e-10:  # Avoid division by zero
            return np.zeros((3, 3))
            
        r_hat = r_ij / r
        
        # Oseen tensor
        T_ij = (1 / (8 * np.pi * self.viscosity * r)) * (np.eye(3) + np.outer(r_hat, r_hat))
        
        return T_ij
    
    def calculate_damping_coefficients(self):
        """
        Calculate the overall damping coefficients considering hydrodynamic interactions
        between components.
        """
        if not self.component_nodes:
            self.decompose_mesh()
            
        n_components = len(self.component_nodes)
        
        # Calculate individual component damping
        component_trans_damping = []
        component_rot_damping = []
        
        for i in range(n_components):
            trans_damp, rot_damp = self._calculate_single_ellipsoid_damping(
                self.component_semi_axes[i], self.component_masses[i])
            component_trans_damping.append(trans_damp)
            component_rot_damping.append(rot_damp)
            
        # Build resistance tensors
        # For simplicity, we'll use the Oseen tensor approximation for hydrodynamic interactions
        
        # Translation-translation coupling
        tt_resistance = np.zeros((3*n_components, 3*n_components))
        
        # Fill diagonal blocks (self-resistance)
        for i in range(n_components):
            idx_i = 3*i
            tt_resistance[idx_i:idx_i+3, idx_i:idx_i+3] = np.diag(component_trans_damping[i])
            
        # Fill off-diagonal blocks (hydrodynamic interactions)
        for i in range(n_components):
            for j in range(i+1, n_components):
                r_ij = self.component_centers[j] - self.component_centers[i]
                T_ij = self._calculate_oseen_tensor(r_ij)
                
                # Convert from mobility to resistance
                coupling = -np.diag(component_trans_damping[i]) @ T_ij @ np.diag(component_trans_damping[j])
                
                idx_i, idx_j = 3*i, 3*j
                tt_resistance[idx_i:idx_i+3, idx_j:idx_j+3] = coupling
                tt_resistance[idx_j:idx_j+3, idx_i:idx_i+3] = coupling.T
                
        # Calculate effective resistance matrices
        # For translation, we need the center of mass weighting
        total_mass = sum(self.component_masses)
        mass_weights = np.array([m/total_mass for m in self.component_masses])
        
        # Projection matrix for center of mass translation
        P_trans = np.zeros((3, 3*n_components))
        for i in range(n_components):
            P_trans[:, 3*i:3*i+3] = mass_weights[i] * np.eye(3)
            
        # Effective translational resistance
        trans_resistance_eff = P_trans @ tt_resistance @ P_trans.T
        
        # Extract the effective damping coefficients
        # These are the eigenvalues of the effective resistance matrix
        trans_damping_eigenvals = np.linalg.eigvals(trans_resistance_eff)
        
        # For rotation, use similar approach but with moments of inertia
        # This is simplified and approximate
        rot_resistance_eff = np.zeros((3, 3))
        total_inertia = np.zeros(3)
        
        for i in range(n_components):
            # Approximate inertia contribution
            inertia_i = self.component_masses[i] * np.sum(self.component_semi_axes[i]**2) / 5
            total_inertia += inertia_i
            
            # Add component contribution
            rot_resistance_eff += np.diag(component_rot_damping[i]) * inertia_i
            
        # Normalize by total inertia
        rot_resistance_eff /= total_inertia
        
        # Extract rotational damping coefficients
        rot_damping_eigenvals = np.linalg.eigvals(rot_resistance_eff)
        
        # Store results
        self.translational_damping = np.sort(trans_damping_eigenvals)
        self.rotational_damping = np.sort(rot_damping_eigenvals)
        
        print("\nEffective Translational Damping Coefficients [1/ns]:")
        print(self.translational_damping)
        print("\nEffective Rotational Damping Coefficients [1/ns]:")
        print(self.rotational_damping)
        
        return self.translational_damping, self.rotational_damping
    
    def visualize_components(self, output_file="components.msh"):
        """
        Create a GMsh file visualizing the component decomposition with different colors.
        """
        if not self.component_nodes:
            self.decompose_mesh()
            
        # Initialize GMsh
        gmsh.initialize()
        gmsh.model.add("decomposed_mesh")
        
        # Add all nodes
        all_nodes = []
        node_idx_map = {}
        
        current_idx = 1
        for comp_idx, nodes in enumerate(self.component_nodes):
            for node in nodes:
                all_nodes.append(node)
                node_idx_map[(comp_idx, tuple(node))] = current_idx
                current_idx += 1
                
        # Add nodes to GMsh
        node_tags = list(range(1, len(all_nodes) + 1))
        node_coords = []
        for node in all_nodes:
            node_coords.extend(node / self.mesh_processor.unit_scale)  # Convert back to mesh units
            
        gmsh.model.mesh.addNodes(2, 1, node_tags, node_coords)
        
        # Add elements for each component with different physical groups
        for comp_idx, nodes in enumerate(self.component_nodes):
            # Create Delaunay triangulation for the component
            if len(nodes) < 4:
                continue  # Skip if too few points
                
            # For simplicity, just create point elements
            element_tags = []
            element_nodes = []
            
            for node in nodes:
                tag = node_idx_map[(comp_idx, tuple(node))]
                element_tags.append(tag)
                element_nodes.append(tag)
                
            # Add to GMsh as points
            gmsh.model.mesh.addElements(0, comp_idx+1, [15], [element_tags], [element_nodes])
            
            # Create physical group for this component
            gmsh.model.addPhysicalGroup(0, [comp_idx+1], comp_idx+1)
            gmsh.model.setPhysicalName(0, comp_idx+1, f"Component_{comp_idx+1}")
            
        # Write the mesh file
        gmsh.write(output_file)
        gmsh.finalize()
        
        print(f"Component visualization saved to {output_file}")


def process_complex_shape(mesh_file, density=1.0, temperature=295, num_components=3,
                         clustering_method='distance', output_dx="complex_shape.dx"):
    """
    Process a complex shape using multi-body approximation.
    
    Args:
        mesh_file: Path to .msh file
        density: Material density in g/cm³
        temperature: Temperature in Kelvin
        num_components: Number of components to decompose the mesh into
        clustering_method: Method to use for mesh decomposition ('distance' or 'curvature')
        output_dx: Path to output potential file
        
    Returns:
        mesh_processor: MeshProcessor instance
        multi_body: MultiBodyHydrodynamics instance
    """
    from .mesh_process import MeshProcessor
    
    # First process the mesh using the standard method
    print("Processing mesh with standard method...")
    mesh_processor = MeshProcessor(mesh_file, density, temperature)
    
    # Then apply multi-body approximation
    print("\nApplying multi-body approximation...")
    multi_body = MultiBodyHydrodynamics(
        mesh_processor, 
        num_components=num_components,
        temperature=temperature,
        viscosity=0.01,  # poise
        clustering_method=clustering_method
    )
    
    # Decompose the mesh
    multi_body.decompose_mesh()
    
    # Calculate effective damping coefficients
    trans_damping, rot_damping = multi_body.calculate_damping_coefficients()
    
    # Visualize the components
    multi_body.visualize_components(output_file=f"{mesh_file.stem}_components.msh")
    
    # Generate potential file
    if output_dx:
        mesh_processor.write_potential_dx(output_dx)
    
    print("\nComparison of Methods:")
    print("Single Ellipsoid Approximation:")
    print(f"Translational Damping: {mesh_processor.damping_coefficient}")
    print(f"Rotational Damping: {mesh_processor.rotational_damping_coefficient}")
    print("\nMulti-body Approximation:")
    print(f"Translational Damping: {trans_damping}")
    print(f"Rotational Damping: {rot_damping}")
    
    # Update mesh_processor with multi-body results
    mesh_processor.multi_body_damping_coefficient = trans_damping
    mesh_processor.multi_body_rotational_damping_coefficient = rot_damping
    
    return mesh_processor, multi_body


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multi_body_approx.py mesh_file.msh [num_components] [clustering_method]")
        sys.exit(1)
        
    mesh_file = sys.argv[1]
    num_components = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    clustering_method = sys.argv[3] if len(sys.argv) > 3 else 'distance'
    
    process_complex_shape(mesh_file, num_components=num_components, clustering_method=clustering_method)
