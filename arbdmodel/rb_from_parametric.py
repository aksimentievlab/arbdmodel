import numpy as np
import gmsh
from pathlib import Path
import os
from scipy import integrate
from .rb_from_mesh import MeshRigidBodyType
from .grid import writeDx
from .logger import logger

class ParametricProcessor:
    """
    The ParametricProcessor class provides tools to generate analytic rigid body geometries
    with their physical properties calculated analytically rather than through numerical integration.
    It can create various shapes such as spheres, capsules, and ellipsoids, compute their
    physical properties (mass, moment of inertia, etc.), and generate potential grids for use
    in ARBD simulations.
    Attributes:
        name (str): Base name for generated files
        mesh_size (float): Characteristic mesh element size
        unit_scale (float): Conversion factor from angstroms to mesh units
        density (float): Material density in g/cm³
        density_amu (float): Density in amu/Å³ for mass calculations
        work_dir (Path): Working directory for output files
        mesh_file (Path): Path to the generated mesh file
        parametric_function (callable): Function defining the parametric surface
        u_range (tuple): Parameter range for u
        v_range (tuple): Parameter range for v
        entity_tag (int): GMSH entity tag
        entity_dim (int): GMSH entity dimension
        volume (float): Calculated volume
        mass (float): Calculated mass
        center_of_mass (array): Calculated center of mass
        inertia_tensor (array): Calculated inertia tensor
        principal_moments (array): Principal moments of inertia
        
    Methods:
        calculate_properties(): Calculate physical properties analytically
        create_potential_grid(): Generate a potential grid based on shape
        _to_gmsh_units(): Convert from Angstroms to gmsh units
        generate_sphere(): Generate a spherical mesh and properties
        generate_capsule(): Generate a capsule mesh and properties
        generate_ellipsoid(): Generate an ellipsoid mesh and properties
        create_rigid_body_type(): Create a RigidBodyType from the shape
    """
    
    def __init__(self, name, mesh_size=0.5, unit_scale=1.0, density=1.0, work_dir=None, **kwargs):
        """
        Initialize the parametric processor.
        
        Args:
            name: Base name for the generated files
            mesh_size: Characteristic mesh element size
            unit_scale: Conversion factor from angstroms to mesh units (e.g., 1e-4 for Å to μm)
            density: Material density in g/cm³
            work_dir: Working directory for output files
        """
        self.name = name
        self.mesh_size = mesh_size
        self.unit_scale = unit_scale  # Factor to convert from Å to mesh units
        self.density = density  # Material density in g/cm³
        self.density_amu = self.density * 0.6022  # Density in amu/Å³ for mass calculations
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
        
    def calculate_properties(self):
        """
        Calculate physical properties analytically from the parametric definition.
            
        Returns:
            Dictionary with calculated properties
        """
        if self.parametric_function is None:
            raise ValueError("Parametric function not defined. Generate a shape first.")
        
        # Use analytical formulas if available
        if hasattr(self, '_volume_formula'):
            self.volume = self._volume_formula()
            self.mass = self.volume * self.density_amu
            
            if hasattr(self, '_com_formula'):
                self.center_of_mass = self._com_formula()
            else:
                self.center_of_mass = np.array([0, 0, 0])  # Default to origin for symmetric shapes
                
            if hasattr(self, '_inertia_formula'):
                self.inertia_tensor = self._inertia_formula(self.mass)
            
            # Calculate principal moments
            if self.inertia_tensor is not None:
                eigenvalues, eigenvectors = np.linalg.eigh(self.inertia_tensor)
                self.principal_moments = eigenvalues[::-1]
                
        prop_dict = {
            "volume": self.volume,
            "mass": self.mass,
            "center_of_mass": self.center_of_mass,
            "inertia_tensor": self.inertia_tensor,
            "principal_moments": self.principal_moments
        }
        
        logger.info(prop_dict)
        return prop_dict

    def create_potential_grid(self, spacing=2, max_potential=100.0, 
                        blur_sigma=1, grid_file=None):
        """
        Generate a potential grid based on vectorized is_inside function.
        Points inside the shape have max_potential value, outside have zero.
        
        Args:
            spacing: Grid spacing in Angstroms
            max_potential: Potential value for points inside the shape
            blur_sigma: Standard deviation for Gaussian blur to smooth the potential
            grid_file: Output DX file path (default: {name}_potential.dx)
            
        Returns:
            Path to the generated DX file
        """
        if grid_file is None:
            grid_file = self.work_dir / f"{self.name}_potential.dx"
        
        if not hasattr(self, 'is_inside'):
            raise ValueError("No is_inside function defined. Cannot generate potential.")
        
        # Determine bounding box from shape parameters
        if hasattr(self, 'bounding_box'):
            min_coords, max_coords = self.bounding_box()
        else:
            # Fallback to sampling if no bounding box function is available
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
        min_coords = min_coords*1.5
        max_coords = max_coords*1.5
        
        # Calculate grid dimensions
        nx = int(np.ceil((max_coords[0] - min_coords[0]) / spacing))
        ny = int(np.ceil((max_coords[1] - min_coords[1]) / spacing))
        nz = int(np.ceil((max_coords[2] - min_coords[2]) / spacing))
        
        # Create grid coordinates
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        z = np.linspace(min_coords[2], max_coords[2], nz)
        
        # Create a grid of 3D points
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Track progress
        total_points = nx * ny * nz
        logger.info(f"Creating potential grid with {total_points} points...")
        
        # Create a binary potential grid using vectorized is_inside
        print("Calculating inside/outside mask with vectorized operation...")
        inside_mask = self.is_inside(X,Y,Z)
        
        # Initialize potential grid (all zeros by default = outside)
        potential = np.zeros((nx, ny, nz))
        
        # Set values for inside points
        potential[self.is_inside(X,Y,Z)] = max_potential
        
        # Count inside points
        num_inside = np.sum(inside_mask)
        print(f"Found {num_inside} points inside the shape ({num_inside/total_points*100:.2f}% of total)")
        
        # Apply Gaussian blur to smooth the potential field
        if blur_sigma > 0:
            print(f"Applying Gaussian blur with sigma={blur_sigma}...")
            from scipy import ndimage
            potential = ndimage.gaussian_filter(potential, sigma=blur_sigma)
        
        # Write DX file
        print(f"Writing potential to {grid_file}...")
        writeDx(str(grid_file), potential, min_coords, [spacing, spacing, spacing])
        
        print("Potential grid created successfully.")
        return grid_file

    def _to_gmsh_units(self, value):
        """Convert from Angstroms to gmsh units using the unit scale"""
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.array(value) * self.unit_scale
        else:
            return value * self.unit_scale
    
    def generate_sphere(self, radius, center=(0, 0, 0)):
        """
        Generate a spherical mesh and calculate properties analytically.
        
        Args:
            radius: Radius of the sphere in Angstroms
            center: Center coordinates (x, y, z) in Angstroms
            
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

        def is_inside(x, y, z):
            """Return True if point (x,y,z) is inside the sphere."""
            dx = x - center[0]
            dy = y - center[1]
            dz = z - center[2]
            distance_squared = dx**2 + dy**2 + dz**2
            return distance_squared <= radius**2
    
        # Define signed distance function
        def signed_distance(x, y, z):
            """Return signed distance from point to sphere surface.
            Negative inside, positive outside."""
            dx = x - center[0]
            dy = y - center[1]
            dz = z - center[2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            return distance - radius
        
        self.is_inside = is_inside
        self.signed_distance = signed_distance
        # Calculate physical properties
        self.calculate_properties()
        
        # Convert values to gmsh units for mesh generation
        gmsh_radius = self._to_gmsh_units(radius)
        gmsh_center = self._to_gmsh_units(center)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create sphere
        sphere = gmsh.model.occ.addSphere(
            gmsh_center[0], gmsh_center[1], gmsh_center[2], 
            gmsh_radius
        )
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
        
    def generate_capsule(self, radius, length=None, aspect_ratio=None, center=(0, 0, 0), axis=(0, 0, 1)):
        """
        Generate a spherical-capped cylinder (capsule) mesh and calculate properties analytically.
        
        Args:
            radius: Radius of the capsule in Angstroms
            length: Total length of the capsule in Angstroms (including hemisphere caps) - provide either this or aspect_ratio
            aspect_ratio: Aspect ratio (length/diameter) - provide either this or length
            center: Center coordinates (x, y, z) in Angstroms of the capsule
            axis: Direction axis (will be normalized)
            
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
            return self.generate_sphere(radius, center)
        
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
        def is_inside(X, Y, Z):
            # Transform points to local coordinates
            p = np.stack([X - self.center[0], Y - self.center[1], Z - self.center[2]], axis=-1)
            
            # Calculate height along axis (dot product)
            h = np.dot(p, self.axis)
            
            # Calculate radial distance
            radial_vector = p - np.multiply(h.reshape(*h.shape, 1), self.axis)
            radial_dist = np.linalg.norm(radial_vector, axis=-1)
            
            # Conditions for inside
            in_cylinder = (np.abs(h) <= self.cylinder_height/2) & (radial_dist <= self.radius)
            in_hemisphere = (~(np.abs(h) <= self.cylinder_height/2)) & (np.linalg.norm(
                p - np.multiply(np.sign(h).reshape(*h.shape, 1) * self.cylinder_height/2, self.axis), 
                axis=-1) <= self.radius)
            
            return in_cylinder | in_hemisphere
        
                     
        # Attach functions to the instance
        self.is_inside = is_inside
        
        # Calculate physical properties
        self.calculate_properties()
        
        # Convert values to gmsh units for mesh generation
        gmsh_radius = self._to_gmsh_units(radius)
        gmsh_cylinder_height = self._to_gmsh_units(cylinder_height)
        gmsh_center = self._to_gmsh_units(center)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Add hemisphere at bottom
        bottom_center = np.array(center) - (cylinder_height/2) * axis_norm
        gmsh_bottom_center = self._to_gmsh_units(bottom_center)
        bottom_sphere = gmsh.model.occ.addSphere(
            gmsh_bottom_center[0], gmsh_bottom_center[1], gmsh_bottom_center[2], 
            gmsh_radius
        )
        
        # Add cylinder in the middle
        cylinder = gmsh.model.occ.addCylinder(
            gmsh_bottom_center[0], gmsh_bottom_center[1], gmsh_bottom_center[2],
            gmsh_cylinder_height * axis_norm[0], gmsh_cylinder_height * axis_norm[1], gmsh_cylinder_height * axis_norm[2],
            gmsh_radius
        )
        
        # Add hemisphere at top
        top_center = np.array(center) + (cylinder_height/2) * axis_norm
        gmsh_top_center = self._to_gmsh_units(top_center)
        top_sphere = gmsh.model.occ.addSphere(
            gmsh_top_center[0], gmsh_top_center[1], gmsh_top_center[2],
            gmsh_radius
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

    def generate_ellipsoid(self, a, b, c, center=(0, 0, 0), axis=(0, 0, 1)):
        """
        Generate an ellipsoid mesh and calculate properties analytically.
        
        Args:
            a, b, c: Semi-principal axes of the ellipsoid in Angstroms
            center: Center coordinates (x, y, z) in Angstroms
            axis: Direction axis for orienting the ellipsoid
            
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
        def is_inside(x, y, z):
            """Return True if point (x,y,z) is inside the ellipsoid."""
            # Transform point to local coordinates
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
            
            # Check if inside the ellipsoid using the equation x²/a² + y²/b² + z²/c² ≤ 1
            return (local_x/a)**2 + (local_y/b)**2 + (local_z/c)**2 <= 1.0
        
        # Define signed distance function for the ellipsoid
        def signed_distance(x, y, z):
            """Approximate signed distance from point to ellipsoid surface.
            Negative inside, positive outside."""
            # Transform point to local coordinates
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
            
            # For ellipsoid, scaled distance to center is (x/a)² + (y/b)² + (z/c)²
            scaled_dist_squared = (local_x/a)**2 + (local_y/b)**2 + (local_z/c)**2
            scaled_dist = np.sqrt(scaled_dist_squared)
            
            if scaled_dist < 1e-10:  # Point is at or very near center
                return -min(a, b, c)
            
            # Calculate unscaled distance to point
            distance = np.linalg.norm(p)
            
            # If inside ellipsoid
            if scaled_dist <= 1.0:
                # Approximate distance to surface by scaling
                return -distance * (1.0 - scaled_dist)
            else:
                # For outside, we can approximate the surface point in the direction of the point
                direction = np.array([local_x, local_y, local_z]) / distance
                
                # Scaled radius in that direction
                r_theta = 1.0 / np.sqrt(
                    (direction[0]/a)**2 + 
                    (direction[1]/b)**2 + 
                    (direction[2]/c)**2
                )
                
                # Distance to surface is approximately distance to center minus radius in that direction
                surface_dist = distance - r_theta
                return surface_dist
        
        # Attach functions to the instance
        self.is_inside = is_inside
        self.signed_distance = signed_distance
        # Calculate physical properties
        self.calculate_properties()
        
        # Convert values to gmsh units for mesh generation
        gmsh_a = self._to_gmsh_units(a)
        gmsh_b = self._to_gmsh_units(b)
        gmsh_c = self._to_gmsh_units(c)
        gmsh_center = self._to_gmsh_units(center)
        
        # Generate mesh using GMSH
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create sphere of radius 1
        sphere = gmsh.model.occ.addSphere(0, 0, 0, 1)
        
        # Scale it to create ellipsoid
        gmsh.model.occ.dilate([(3, sphere)], 0, 0, 0, gmsh_a, gmsh_b, gmsh_c)
        
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
        gmsh.model.occ.translate([(3, sphere)], gmsh_center[0], gmsh_center[1], gmsh_center[2])
        
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

    def create_rigid_body_type(self, simconf=None):
        """
        Create a RigidBodyType directly from the generated parametric shape.
        
        Args:
            simconf: SimConf object with parameters
            
        Returns:
            A MeshRigidBodyType object ready for simulation
        """
        
        # Ensure we have calculated properties
        if self.mass is None:
            self.calculate_properties()
        
        # Generate potential grid
        potential_dx = self.work_dir / f"{self.name}_potential.dx"
        if not potential_dx.exists():
            self.create_potential_grid(grid_file=potential_dx)
        """
        from .mesh_process_volume import MeshProcessor
        rbprocess = MeshProcessor(
                self.mesh_file,
                density=density, 
                simconf=simconf, 
                unit_scale=unit_scale,
                work_dir=self.type_dir,expected_mass=expected_mass)
        
        rbprocess.calculate_damping()
        attached_particles= rbprocess.get_attached_particles()
        # Create the rigid body type

        """
        
        rb_type = MeshRigidBodyType(
            name=self.name,
            mesh_file=self.mesh_file,
            density=self.density, 
            simconf=simconf,
            unit_scale=1/self.unit_scale, expected_mass=self.mass)

        # Override the calculated properties with our analytically-determined ones
        rb_type.mass = self.mass
        rb_type.moment_of_inertia = np.sort(self.principal_moments)
        rb_type.potential_grids = [("shape", str(potential_dx), 1.0)]
        
        return rb_type