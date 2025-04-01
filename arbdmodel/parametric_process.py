import numpy as np
import gmsh
from pathlib import Path
import os
from .grid import writeDx

import numpy as np
import gmsh
from pathlib import Path
import os
from scipy import integrate
from .grid import writeDx

class ParametricProcessor:
    """
    Process parametrically defined shapes to create meshes and potential grids for ARBD.
    Generates volume meshes, calculates physical properties analytically, and creates
    potential fields directly from the parametric definitions.
    """
    
    def __init__(self, name, mesh_size=0.5, mesh_unit_scale=1, work_dir=None, **kwargs):
        """
        Initialize the parametric processor.
        
        Args:
            name: Base name for the generated files
            mesh_size: Characteristic mesh element size
            unit_scale: Conversion factor from angstroms to mesh units
            work_dir: Working directory for output files
        """
        self.name = name
        self.mesh_size = mesh_size
        self.unit_scale = mesh_unit_scale
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        os.makedirs(self.work_dir, exist_ok=True)
        self.mesh_file = self.work_dir / f"{name}.msh"
        self.parametric_function = None
        self.u_range = None
        self.v_range = None
        self.entity_tag = None
        self.entity_dim = None
        
        # Properties that will be calculated analytically
        self.volume = None
        self.mass = None
        self.center_of_mass = None
        self.inertia_tensor = None
        self.principal_moments = None
        
    def calculate_properties(self, density=1.0):
        """
        Calculate physical properties analytically from the parametric definition.
        
        Args:
            density: Material density in g/cm^3
            
        Returns:
            Dictionary with calculated properties
        """
        if self.parametric_function is None:
            raise ValueError("Parametric function not defined. Generate a shape first.")
        
        # Convert density to amu/Å^3
        density_amu = density * 0.6022  # Conversion factor
        
        # Use analytical formulas if available
        if hasattr(self, '_volume_formula'):
            self.volume = self._volume_formula()
            self.mass = self.volume * density_amu
            
            if hasattr(self, '_com_formula'):
                self.center_of_mass = self._com_formula()
            else:
                self.center_of_mass = np.array([0, 0, 0])  # Default to origin for symmetric shapes
                
            if hasattr(self, '_inertia_formula'):
                self.inertia_tensor = self._inertia_formula(self.mass)
            
            # Calculate principal moments
            if self.inertia_tensor is not None:
                eigenvalues, eigenvectors = np.linalg.eigh(self.inertia_tensor)
                self.principal_moments = eigenvalues
        prop_dict={
            "volume": self.volume,
            "mass": self.mass,
            "center_of_mass": self.center_of_mass,
            "inertia_tensor": self.inertia_tensor,
            "principal_moments": self.principal_moments}
        print(prop_dict)
        return prop_dict
    
    def create_potential_grid(self, spacing=2.0, buffer=20.0, max_potential=100.0, grid_file=None):
        """
        Generate a potential grid based directly on the parametric definition.
        
        Args:
            spacing: Grid spacing in mesh units
            buffer: Additional buffer space around the shape
            max_potential: Maximum potential value
            grid_file: Output DX file path (default: {name}_potential.dx)
            
        Returns:
            Path to the generated DX file
        """
        if grid_file is None:
            grid_file = self.work_dir / f"{self.name}_potential.dx"
        
        if not hasattr(self, 'distance_function'):
            raise ValueError("No distance function defined. Cannot generate potential.")
        
        # Determine bounding box from shape parameters
        if hasattr(self, 'bounding_box'):
            min_coords, max_coords = self.bounding_box()
        else:
            # Fallback to sampling if no bounding box function is available
            # Sample the parametric function to get approximate bounding box
            u_samples = np.linspace(self.u_range[0], self.u_range[1], 50)
            v_samples = np.linspace(self.v_range[0], self.v_range[1], 50)
            
            sample_points = []
            for u in u_samples:
                for v in v_samples:
                    sample_points.append(self.parametric_function(u, v))
            
            sample_points = np.array(sample_points)
            min_coords = np.min(sample_points, axis=0)
            max_coords = np.max(sample_points, axis=0)
        
        # Add buffer
        min_coords = min_coords-buffer
        max_coords = max_coords+buffer
        
        # Calculate grid dimensions
        nx = int(np.ceil((max_coords[0] - min_coords[0]) / spacing))
        ny = int(np.ceil((max_coords[1] - min_coords[1]) / spacing))
        nz = int(np.ceil((max_coords[2] - min_coords[2]) / spacing))
        
        # Create grid coordinates
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        z = np.linspace(min_coords[2], max_coords[2], nz)
        
        # Use shape-specific potential function if available
        if hasattr(self, 'generate_potential'):
            potential = self.generate_potential(x, y, z)
        else:
            # Create a grid of 3D points
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Calculate signed distance for each point using vectorized operations if possible
            # Init with a large positive value
            distances = np.ones(X.shape) * 1000.0
            
            # Calculate distances in batches to avoid memory issues
            batch_size = 10000
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            
            for i in range(0, len(points), batch_size):
                end = min(i + batch_size, len(points))
                batch_points = points[i:end]
                
                # Calculate distances for this batch
                batch_distances = np.zeros(len(batch_points))
                for j, p in enumerate(batch_points):
                    batch_distances[j] = self.distance_function(p[0], p[1], p[2])
                
                # Store in the flattened distances array
                distances.ravel()[i:end] = batch_distances
            
            # Convert distances to potential
            # Inside gets maximum value, outside decays exponentially
            scale = spacing * 2.0  # Scale factor for decay
            potential = np.zeros_like(distances)
            potential[distances <= 0] = max_potential
            potential[distances > 0] = max_potential * np.exp(-distances[distances > 0] / scale)
        
        # Write DX file
        writeDx(str(grid_file), potential, min_coords, [spacing, spacing, spacing])
        
        return grid_file
    
    def generate_sphere(self, radius, center=(0, 0, 0), density=1.0):
        """
        Generate a spherical mesh and calculate properties analytically.
        
        Args:
            radius: Radius of the sphere
            center: Center coordinates (x, y, z)
            density: Material density in g/cm^3
            
        Returns:
            Path to the generated mesh file
        """
        # Store parameters for analytical calculations
        self.radius = radius
        self.center = center
        
        # Define analytical formulas for sphere
        self._volume_formula = lambda: (4/3) * np.pi * radius**3
        self._com_formula = lambda: np.array(center)
        self._inertia_formula = lambda mass: (2/5) * mass * radius**2 * np.eye(3)
        
        # Define parametric function for sphere
        def sphere_function(u, v):
            # u is longitude (0 to 2π)
            # v is latitude (0 to π)
            x = radius * np.sin(v) * np.cos(u) + center[0]
            y = radius * np.sin(v) * np.sin(u) + center[1]
            z = radius * np.cos(v) + center[2]
            return x, y, z
        
        # Store for later use
        self.parametric_function = sphere_function
        self.u_range = (0, 2*np.pi)
        self.v_range = (0, np.pi)
        
        # Define bounding box function
        def bounding_box():
            min_coords = np.array(center) - radius
            max_coords = np.array(center) + radius
            return min_coords, max_coords
        
        self.bounding_box = bounding_box
        
        # Define distance function for potential calculation
        def sphere_distance(x, y, z):
            # Calculate distance from point to sphere center
            dx = x - center[0]
            dy = y - center[1]
            dz = z - center[2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            # Return signed distance (negative inside, positive outside)
            return dist - radius
        
        self.distance_function = sphere_distance
        
        # Define potential generator function
        def generate_potential(x, y, z):
            # Create a grid of 3D points
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Calculate distance to sphere center for each point
            dx = X - center[0]
            dy = Y - center[1]
            dz = Z - center[2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Calculate signed distance
            signed_dist = dist - radius
            
            # Create potential (negative distances are inside)
            max_val = 100.0  # Maximum potential value
            scale = 2.0  # Scale factor for exponential decay
            
            # Potential inside is maximum, outside decays exponentially
            potential = np.zeros_like(signed_dist)
            potential[signed_dist <= 0] = max_val
            potential[signed_dist > 0] = max_val * np.exp(-signed_dist[signed_dist > 0] / scale)
            
            return potential
        
        self.generate_potential = generate_potential
        
        # Calculate physical properties
        self.calculate_properties(density)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create sphere
        sphere = gmsh.model.occ.addSphere(center[0], center[1], center[2], radius)
        gmsh.model.occ.synchronize()
        
        # Store entity for later use
        self.entity_dim = 3
        self.entity_tag = sphere
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        
        # Save mesh
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
        return self.mesh_file
        
    def generate_cylinder(self, radius, height, center=(0, 0, 0), axis=(0, 0, 1), density=1.0):
        """
        Generate a cylindrical mesh and calculate properties analytically.
        
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            center: Base center coordinates (x, y, z)
            axis: Direction axis (normalized)
            density: Material density in g/cm^3
            
        Returns:
            Path to the generated mesh file
        """
        # Normalize axis
        axis_norm = np.array(axis) / np.linalg.norm(axis)
        
        # Store parameters for analytical calculations
        self.radius = radius
        self.height = height
        self.center = center
        self.axis = axis_norm
        
        # Define analytical formulas for cylinder
        self._volume_formula = lambda: np.pi * radius**2 * height
        
        # Center of mass is at the middle of the cylinder
        self._com_formula = lambda: np.array(center) + height/2 * axis_norm
        
        # Define inertia tensor formula
        def cylinder_inertia(mass):
            # Inertia tensor depends on orientation
            # First calculate inertia in the cylinder's local coordinate system
            # For a cylinder with axis along z:
            # I_xx = I_yy = m/12 * (3r² + h²) and I_zz = m/2 * r²
            I_axial = mass/2 * radius**2
            I_radial = mass/12 * (3*radius**2 + height**2)
            
            # The diagonal elements directly
            I_local = np.diag([I_radial, I_radial, I_axial])
            
            # Now rotate to the global coordinate system
            # We need to create a rotation matrix that aligns [0,0,1] with axis_norm
            v = np.cross([0, 0, 1], axis_norm)
            s = np.linalg.norm(v)
            
            if s < 1e-10:  # Axes are nearly aligned already
                R = np.eye(3)
            else:
                c = np.dot([0, 0, 1], axis_norm)  # cosine of angle
                v_skew = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                # Rodrigues' rotation formula
                R = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1-c)/s**2
            
            # Transform inertia tensor
            I_global = R.dot(I_local).dot(R.T)
            
            return I_global
        
        self._inertia_formula = cylinder_inertia
        
        # Define parametric function for cylinder
        def cylinder_function(u, v):
            # u is height (0 to height)
            # v is angle (0 to 2π)
            
            # Create two orthogonal vectors to axis
            if abs(axis_norm[2]) < 0.9:
                v1 = np.cross(axis_norm, [0, 0, 1])
            else:
                v1 = np.cross(axis_norm, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(axis_norm, v1)
            
            # Calculate point on cylinder
            base = np.array(center)
            point = base + u * axis_norm + radius * (v1 * np.cos(v) + v2 * np.sin(v))
            
            return point[0], point[1], point[2]
        
        # Store for later use
        self.parametric_function = cylinder_function
        self.u_range = (0, height)
        self.v_range = (0, 2*np.pi)
        
        # Define bounding box function
        def bounding_box():
            # Create coordinate system with axis_norm as z-axis
            if abs(axis_norm[2]) < 0.9:
                v1 = np.cross(axis_norm, [0, 0, 1])
            else:
                v1 = np.cross(axis_norm, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(axis_norm, v1)
            
            # For a cylinder, the bounding box depends on its orientation
            # We need to calculate the 8 corner points of the cylinder's bounding box
            corners = []
            
            # Center of base and top
            base_center = np.array(center)
            top_center = base_center + height * axis_norm
            
            # Calculate the 8 corners (4 on each end of the cylinder)
            for h in [base_center, top_center]:
                for i in range(4):
                    angle = i * np.pi/2
                    corner = h + radius * (v1 * np.cos(angle) + v2 * np.sin(angle))
                    corners.append(corner)
            
            corners = np.array(corners)
            min_coords = np.min(corners, axis=0)
            max_coords = np.max(corners, axis=0)
            
            return min_coords, max_coords
        
        self.bounding_box = bounding_box
        
        # Define distance function for potential calculation
        def cylinder_distance(x, y, z):
            # Calculate coordinates in cylinder's local frame
            p = np.array([x, y, z]) - np.array(center)
            
            # Calculate height along axis (dot product)
            h = np.dot(p, axis_norm)
            
            # Calculate radial distance
            radial_vector = p - h * axis_norm
            radial_dist = np.linalg.norm(radial_vector)
            
            # Calculate signed distance
            # Inside cylinder: both h between 0 and height, and radial_dist < radius
            if 0 <= h <= height and radial_dist <= radius:
                # Inside - negative distance
                dist_to_side = radius - radial_dist
                dist_to_base = h
                dist_to_top = height - h
                return -min(dist_to_side, dist_to_base, dist_to_top)
            else:
                # Outside - positive distance
                # Distance to side cylinder
                if 0 <= h <= height:
                    return radial_dist - radius
                
                # Distance to base
                if h < 0:
                    if radial_dist <= radius:
                        return -h
                    else:
                        return np.sqrt((-h)**2 + (radial_dist - radius)**2)
                
                # Distance to top
                if h > height:
                    if radial_dist <= radius:
                        return h - height
                    else:
                        return np.sqrt((h - height)**2 + (radial_dist - radius)**2)
        
        self.distance_function = cylinder_distance
        
        # Calculate physical properties
        self.calculate_properties(density)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create cylinder aligned with provided axis
        cylinder = gmsh.model.occ.addCylinder(
            center[0], center[1], center[2], 
            axis_norm[0]*height, axis_norm[1]*height, axis_norm[2]*height, 
            radius
        )
        gmsh.model.occ.synchronize()
        
        # Store entity for later use
        self.entity_dim = 3
        self.entity_tag = cylinder
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        
        # Save mesh
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
        return self.mesh_file
        
    def generate_capsule(self, radius, length=None, aspect_ratio=None, center=(0, 0, 0), axis=(0, 0, 1), density=1.0):
        """
        Generate a spherical-capped cylinder (capsule) mesh and calculate properties analytically.
        
        Args:
            radius: Radius of the capsule
            length: Total length of the capsule (including hemisphere caps) - provide either this or aspect_ratio
            aspect_ratio: Aspect ratio (length/diameter) - provide either this or length
            center: Center coordinates (x, y, z) of the capsule
            axis: Direction axis (will be normalized)
            density: Material density in g/cm^3
            
        Returns:
            Path to the generated mesh file
        """
        # Ensure either length or aspect ratio is provided
        if length is None and aspect_ratio is None:
            raise ValueError("Either length or aspect_ratio must be provided")
        if length is not None and aspect_ratio is not None:
            raise ValueError("Provide either length or aspect_ratio, not both")
        
        # Calculate length from aspect ratio if needed
        if length is None:
            length = aspect_ratio * (2 * radius)  # Length = AR * Diameter
        
        # Calculate cylinder height (total length minus caps)
        cylinder_height = length - 2 * radius
        if cylinder_height < 0:
            # In case the length is less than a sphere diameter, just create a sphere
            return self.generate_sphere(radius, center, density)
        
        # Normalize axis
        axis_norm = np.array(axis) / np.linalg.norm(axis)
        
        # Store parameters for analytical calculations
        self.radius = radius
        self.cylinder_height = cylinder_height
        self.total_length = length
        self.center = center
        self.axis = axis_norm
        
        # Define analytical formulas for capsule
        # Volume = cylinder volume + two hemisphere volumes
        self._volume_formula = lambda: np.pi * radius**2 * cylinder_height + (4/3) * np.pi * radius**3
        
        # Center of mass is at the geometric center
        self._com_formula = lambda: np.array(center)
        
        # Define inertia tensor formula for capsule
        def capsule_inertia(mass):
            # Calculate mass of cylinder and hemispheres
            total_volume = np.pi * radius**2 * cylinder_height + (4/3) * np.pi * radius**3
            cylinder_volume = np.pi * radius**2 * cylinder_height
            hemispheres_volume = (4/3) * np.pi * radius**3
            
            cylinder_mass = mass * (cylinder_volume / total_volume)
            hemispheres_mass = mass * (hemispheres_volume / total_volume)
            
            # Inertia tensor for cylinder aligned with z-axis
            cylinder_I_axial = cylinder_mass/2 * radius**2
            cylinder_I_radial = cylinder_mass/12 * (3*radius**2 + cylinder_height**2)
            
            # Inertia tensor for hemispheres
            # Each hemisphere has I = (2/5)mr² about its center
            hemisphere_I = (2/5) * (hemispheres_mass/2) * radius**2
            
            # Use parallel axis theorem to shift hemisphere inertia to capsule center
            # The distance from capsule center to hemisphere center is cylinder_height/2
            distance = cylinder_height/2
            hemisphere_I_shifted = hemisphere_I + (hemispheres_mass/2) * distance**2
            
            # Combine inertias in local coordinate system
            I_axial = cylinder_I_axial + 2 * hemisphere_I  # No shift needed for axial component
            I_radial = cylinder_I_radial + 2 * hemisphere_I_shifted
            
            # The diagonal elements in local coordinates
            I_local = np.diag([I_radial, I_radial, I_axial])
            
            # Now rotate to the global coordinate system
            # We need to create a rotation matrix that aligns [0,0,1] with axis_norm
            v = np.cross([0, 0, 1], axis_norm)
            s = np.linalg.norm(v)
            
            if s < 1e-10:  # Axes are nearly aligned already
                R = np.eye(3)
            else:
                c = np.dot([0, 0, 1], axis_norm)  # cosine of angle
                v_skew = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                # Rodrigues' rotation formula
                R = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1-c)/s**2
            
            # Transform inertia tensor
            I_global = R.dot(I_local).dot(R.T)
            
            return I_global
        
        self._inertia_formula = capsule_inertia
        
        # Define parametric function for capsule
        def capsule_function(u, v):
            """
            Parametric function for a capsule.
            
            Parameters:
            u: First parameter, goes from 0 to 1 along capsule length
            v: Second parameter, goes from 0 to 2π around capsule
            
            Returns:
            (x, y, z) coordinates on the capsule surface
            """
            # Map u to position along capsule
            total_length = cylinder_height + 2 * radius
            
            if u < 0.25:  # Bottom hemisphere
                # Map 0-0.25 to 0-π/2 for bottom hemisphere
                phi = np.pi - u * 2 * np.pi  # π down to π/2
                r = radius * np.sin(phi)  # Radius at current latitude
                h = -cylinder_height/2 - radius * np.cos(phi)  # Height from center
            elif u > 0.75:  # Top hemisphere
                # Map 0.75-1 to π/2-π for top hemisphere
                phi = (u - 0.75) * 2 * np.pi  # π/2 up to π
                r = radius * np.sin(phi)  # Radius at current latitude
                h = cylinder_height/2 + radius * np.cos(phi)  # Height from center
            else:  # Cylinder part
                # Map 0.25-0.75 to cover the cylinder
                scaled_u = (u - 0.25) / 0.5  # Rescale to 0-1
                r = radius
                h = -cylinder_height/2 + scaled_u * cylinder_height
            
            # Calculate position on circle at height h
            theta = v  # v goes from 0 to 2π
            
            # Create a coordinate system with axis_norm as z
            if abs(axis_norm[2]) < 0.9:
                v1 = np.cross(axis_norm, [0, 0, 1])
            else:
                v1 = np.cross(axis_norm, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(axis_norm, v1)
            
            # Calculate point in local coordinates first
            local_point = np.array([
                r * np.cos(theta),
                r * np.sin(theta),
                h
            ])
            
            # Transform to global coordinates
            global_point = np.array(center) + \
                           local_point[0] * v1 + \
                           local_point[1] * v2 + \
                           local_point[2] * axis_norm
            
            return global_point[0], global_point[1], global_point[2]
        
        # Store for later use
        self.parametric_function = capsule_function
        self.u_range = (0, 1)
        self.v_range = (0, 2*np.pi)
        
        # Define bounding box function
        def bounding_box():
            # Create coordinate system with axis_norm as z-axis
            if abs(axis_norm[2]) < 0.9:
                v1 = np.cross(axis_norm, [0, 0, 1])
            else:
                v1 = np.cross(axis_norm, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(axis_norm, v1)
            
            # Calculate half-length of capsule along axis
            half_length = cylinder_height/2 + radius
            
            # Calculate the bounding box in local coordinates
            local_min = np.array([-radius, -radius, -half_length])
            local_max = np.array([radius, radius, half_length])
            
            # Transform to global coordinates
            corners = []
            for x in [local_min[0], local_max[0]]:
                for y in [local_min[1], local_max[1]]:
                    for z in [local_min[2], local_max[2]]:
                        local_point = np.array([x, y, z])
                        global_point = np.array(center) + \
                                       local_point[0] * v1 + \
                                       local_point[1] * v2 + \
                                       local_point[2] * axis_norm
                        corners.append(global_point)
            
            corners = np.array(corners)
            min_coords = np.min(corners, axis=0)
            max_coords = np.max(corners, axis=0)
            
            return min_coords, max_coords
        
        self.bounding_box = bounding_box
        
        # Add distance function for potential calculation
        def capsule_distance(x, y, z):
            """
            Calculate signed distance from point (x,y,z) to capsule surface.
            Negative inside, positive outside.
            """
            # Transform point to local coordinate system
            p = np.array([x, y, z]) - np.array(center)
            
            # Calculate height along axis (dot product)
            h = np.dot(p, axis_norm)
            
            # Calculate radial distance
            radial_vector = p - h * axis_norm
            radial_dist = np.linalg.norm(radial_vector)
            
            # Clamp h to cylinder segment
            h_clamped = np.clip(h, -cylinder_height/2, cylinder_height/2)
            
            # Find closest point on cylinder axis
            closest_on_axis = np.array(center) + h_clamped * axis_norm
            
            # Distance from point to closest point on axis
            dist_to_axis = np.linalg.norm(np.array([x, y, z]) - closest_on_axis)
            
            # Signed distance
            return dist_to_axis - radius
        
        self.distance_function = capsule_distance
        
        # Calculate physical properties
        self.calculate_properties(density)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Add hemisphere at bottom
        bottom_center = np.array(center) - (cylinder_height/2) * axis_norm
        bottom_sphere = gmsh.model.occ.addSphere(
            bottom_center[0], bottom_center[1], bottom_center[2], 
            radius
        )
        
        # Add cylinder in the middle
        cylinder = gmsh.model.occ.addCylinder(
            bottom_center[0], bottom_center[1], bottom_center[2],
            cylinder_height * axis_norm[0], cylinder_height * axis_norm[1], cylinder_height * axis_norm[2],
            radius
        )
        # Add hemisphere at top
        top_center = np.array(center) + (cylinder_height/2) * axis_norm
        top_sphere = gmsh.model.occ.addSphere(
            top_center[0], top_center[1], top_center[2],
            radius
        )
        
        # Fuse all components
        capsule = gmsh.model.occ.fuse([(3, bottom_sphere)], [(3, cylinder), (3, top_sphere)])
        gmsh.model.occ.synchronize()
        
        # Store entity for later use
        self.entity_dim = 3
        self.entity_tag = capsule[0][0][1]
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        
        # Save mesh
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
        return self.mesh_file
    
        def generate_ellipsoid(self, a, b, c, center=(0, 0, 0), axis=(0, 0, 1), density=1.0):
            """
            Generate an ellipsoid mesh and calculate properties analytically.
            
            Args:
                a, b, c: Semi-principal axes of the ellipsoid
                center: Center coordinates (x, y, z)
                axis: Direction axis for orienting the ellipsoid
                density: Material density in g/cm^3
                
            Returns:
                Path to the generated mesh file
            """
            # Normalize axis
            axis_norm = np.array(axis) / np.linalg.norm(axis)
            
            # Store parameters for analytical calculations
            self.a = a
            self.b = b
            self.c = c
            self.center = center
            self.axis = axis_norm
            
            # Define analytical formulas for ellipsoid
            self._volume_formula = lambda: (4/3) * np.pi * a * b * c
            self._com_formula = lambda: np.array(center)
            
            # Define inertia tensor formula
            def ellipsoid_inertia(mass):
                # Inertia tensor in principal axes
                I_local = np.diag([
                    mass/5 * (b**2 + c**2),
                    mass/5 * (a**2 + c**2),
                    mass/5 * (a**2 + b**2)
                ])
                
                # Now rotate to the global coordinate system
                # We need to create a rotation matrix that aligns [0,0,1] with axis_norm
                v = np.cross([0, 0, 1], axis_norm)
                s = np.linalg.norm(v)
                
                if s < 1e-10:  # Axes are nearly aligned already
                    R = np.eye(3)
                else:
                    c = np.dot([0, 0, 1], axis_norm)  # cosine of angle
                    v_skew = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    # Rodrigues' rotation formula
                    R = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1-c)/s**2
                
                # Transform inertia tensor
                I_global = R.dot(I_local).dot(R.T)
                
                return I_global
            
            self._inertia_formula = ellipsoid_inertia
            
            # Define parametric function for ellipsoid
            def ellipsoid_function(u, v):
                # u is longitude (0 to 2π)
                # v is latitude (0 to π)
                
                # Create coordinate system with axis_norm as z
                if abs(axis_norm[2]) < 0.9:
                    v1 = np.cross(axis_norm, [0, 0, 1])
                else:
                    v1 = np.cross(axis_norm, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(axis_norm, v1)
                
                # Local coordinates for ellipsoid
                local_x = a * np.sin(v) * np.cos(u)
                local_y = b * np.sin(v) * np.sin(u)
                local_z = c * np.cos(v)
                
                # Transform to global coordinates
                global_point = np.array(center) + \
                            local_x * v1 + \
                            local_y * v2 + \
                            local_z * axis_norm
                
                return global_point[0], global_point[1], global_point[2]
            
            # Store for later use
            self.parametric_function = ellipsoid_function
            self.u_range = (0, 2*np.pi)
            self.v_range = (0, np.pi)
            
            # Define bounding box function
            def bounding_box():
                # Create coordinate system with axis_norm as z-axis
                if abs(axis_norm[2]) < 0.9:
                    v1 = np.cross(axis_norm, [0, 0, 1])
                else:
                    v1 = np.cross(axis_norm, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(axis_norm, v1)
                
                # Find the 8 corners of the bounding box in local coordinates
                corners = []
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        for sz in [-1, 1]:
                            local_point = np.array([sx*a, sy*b, sz*c])
                            global_point = np.array(center) + \
                                        local_point[0] * v1 + \
                                        local_point[1] * v2 + \
                                        local_point[2] * axis_norm
                            corners.append(global_point)
                
                corners = np.array(corners)
                min_coords = np.min(corners, axis=0)
                max_coords = np.max(corners, axis=0)
                
                return min_coords, max_coords
            
            self.bounding_box = bounding_box
            
            # Define distance function for potential calculation
            def ellipsoid_distance(x, y, z):
                # Transform point to local coordinate system
                p = np.array([x, y, z]) - np.array(center)
                
                # Create coordinate system with axis_norm as z
                if abs(axis_norm[2]) < 0.9:
                    v1 = np.cross(axis_norm, [0, 0, 1])
                else:
                    v1 = np.cross(axis_norm, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(axis_norm, v1)
                
                # Transform to local ellipsoid coordinates
                local_x = np.dot(p, v1)
                local_y = np.dot(p, v2)
                local_z = np.dot(p, axis_norm)
                
                # For ellipsoid, distance calculation is more complex
                # Approximate by scaling to sphere and back
                scaled_point = np.array([local_x/a, local_y/b, local_z/c])
                scaled_dist = np.linalg.norm(scaled_point)
                
                # If point is at origin, return -min(a,b,c)
                if scaled_dist < 1e-10:
                    return -min(a, b, c)
                
                # Scale back
                if scaled_dist <= 1.0:  # Inside ellipsoid
                    # Approximation: scale the distance by the radius in the direction of the point
                    dist_factor = min(a, b, c)
                    return -dist_factor * (1.0 - scaled_dist)
                else:  # Outside ellipsoid
                    # Approximation: scale the distance by the radius in the direction of the point
                    norm_vector = scaled_point / scaled_dist
                    # Get ellipsoid radius in the direction of norm_vector
                    radius_dir = np.sqrt(1.0 / (
                        (norm_vector[0]/a)**2 + 
                        (norm_vector[1]/b)**2 + 
                        (norm_vector[2]/c)**2
                    ))
                    # Scale the distance
                    unscaled_dist = np.linalg.norm(np.array([local_x, local_y, local_z]))
                    return unscaled_dist - radius_dir
            
            self.distance_function = ellipsoid_distance
            
            # Calculate physical properties
            self.calculate_properties(density)
            
            # Generate mesh using GMSH
            gmsh.initialize()
            gmsh.model.add(self.name)
            
            # Create sphere of radius 1
            sphere = gmsh.model.occ.addSphere(0, 0, 0, 1)
            
            # Scale it to create ellipsoid
            gmsh.model.occ.dilate([(3, sphere)], 0, 0, 0, a, b, c)
            
            # If axis is not aligned with z, rotate the ellipsoid
            if not np.allclose(axis_norm, [0, 0, 1]):
                # Calculate rotation axis and angle
                rotation_axis = np.cross([0, 0, 1], axis_norm)
                if np.linalg.norm(rotation_axis) > 1e-10:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    cos_angle = np.dot([0, 0, 1], axis_norm)
                    angle = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees for GMSH
                    
                    # Rotate
                    gmsh.model.occ.rotate([(3, sphere)], 0, 0, 0, 
                                        rotation_axis[0], rotation_axis[1], rotation_axis[2], 
                                        angle)
            
            # Translate to center
            gmsh.model.occ.translate([(3, sphere)], center[0], center[1], center[2])
            
            gmsh.model.occ.synchronize()
            
            # Store entity for later use
            self.entity_dim = 3
            self.entity_tag = sphere
            
            # Mesh settings
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
            
            # Generate mesh
            gmsh.model.mesh.generate(3)
            
            # Save mesh
            gmsh.write(str(self.mesh_file))
            gmsh.finalize()
            
            return self.mesh_file

    