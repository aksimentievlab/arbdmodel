import numpy as np
import gmsh
from pathlib import Path
import os
from .grid import writeDx

class ParametricProcessor:
    """
    Generate 3D meshes from parametric equations and feed them into the arbdmodel workflow.
    Generates mesh, calculates hydrodynamic properties, and creates potential grids.
    
    This class eliminates the need for external mesh generation by allowing users
    to define surfaces with mathematical functions.
    """
    
    def __init__(self, name, mesh_size=0.001, unit_scale=10000, work_dir=None,function=None,
                 u_range=None, v_range=None, entity_dim=None, param_bounds=None):
        """
        Initialize the parametric surface generator.
        
        Args:
            name: Base name for the generated files
            mesh_size: Characteristic mesh element size
            unit_scale: Conversion factor from mesh units to Angstroms
            work_dir: Working directory for output files
        """
        self.name = name
        self.mesh_size = mesh_size
        self.unit_scale = unit_scale
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        os.makedirs(self.work_dir, exist_ok=True)
        self.mesh_file = self.work_dir / f"{name}.msh"
        self.parametric_function = function
        self.u_range = u_range
        self.v_range = v_range
        self.entity_tag = None
        self.entity_dim = None
        self.param_bounds = None
        
    def generate_parametric_surface(self, function, u_range, v_range, u_samples=50, v_samples=50, volume=True):
        """
        Generate a surface from a parametric function.
        
        Args:
            function: A function that takes (u,v) and returns (x,y,z)
            u_range: Tuple (u_min, u_max) for parameter u
            v_range: Tuple (v_min, v_max) for parameter v
            u_samples: Number of samples in u direction
            v_samples: Number of samples in v direction
            volume: Whether to create a volume or just a surface
            
        Returns:
            Path to the generated mesh file
        """
        # Store the parametric function and bounds for later grid generation
        self.parametric_function = function
        self.u_range = u_range
        self.v_range = v_range
        
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create points from parametric function
        u_values = np.linspace(u_range[0], u_range[1], u_samples)
        v_values = np.linspace(v_range[0], v_range[1], v_samples)
        
        # Create the parametric surface
        points = []
        for i, u in enumerate(u_values):
            for j, v in enumerate(v_values):
                x, y, z = function(u, v)
                point_tag = gmsh.model.geo.addPoint(x, y, z, self.mesh_size)
                points.append((i, j, point_tag))
        
        # Create lines and surfaces
        surfaces = []
        for i in range(u_samples-1):
            for j in range(v_samples-1):
                p1 = next(p[2] for p in points if p[0] == i and p[1] == j)
                p2 = next(p[2] for p in points if p[0] == i+1 and p[1] == j)
                p3 = next(p[2] for p in points if p[0] == i+1 and p[1] == j+1)
                p4 = next(p[2] for p in points if p[0] == i and p[1] == j+1)
                
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p4)
                l4 = gmsh.model.geo.addLine(p4, p1)
                
                curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
                surface = gmsh.model.geo.addPlaneSurface([curve_loop])
                surfaces.append(surface)
                
        gmsh.model.geo.synchronize()
        
        if volume:
            # Create volume if requested
            try:
                # Try to create a surface loop from all surfaces
                surface_loop = gmsh.model.geo.addSurfaceLoop(surfaces)
                volume = gmsh.model.geo.addVolume([surface_loop])
                self.entity_dim = 3
                self.entity_tag = volume
            except:
                # If it fails, we'll just use the surface
                print("Warning: Could not create volume, using surface instead")
                self.entity_dim = 2
                self.entity_tag = surfaces[0]  # Just use the first surface as reference
        else:
            # Just use the surface
            self.entity_dim = 2
            self.entity_tag = surfaces[0]  # Just use the first surface as reference
            
        gmsh.model.geo.synchronize()
        
        # Get parametrization bounds
        try:
            self.param_bounds = gmsh.model.getParametrizationBounds(self.entity_dim, self.entity_tag)
            print(f"Parametrization bounds: {self.param_bounds}")
        except:
            print("Warning: Could not get parametrization bounds")
            self.param_bounds = None
        
        # Generate mesh
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        gmsh.model.mesh.generate(3 if volume else 2)
        
        # Save mesh
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
        return self.mesh_file
    
    def generate_sphere(self, radius, center=(0, 0, 0)):
        """
        Generate a spherical mesh.
        
        Args:
            radius: Radius of the sphere
            center: Center coordinates (x, y, z)
            
        Returns:
            Path to the generated mesh file
        """
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
        
        # Use gmsh built-in sphere for better quality
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
        
    def generate_cylinder(self, radius, height, center=(0, 0, 0), axis=(0, 0, 1)):
        """
        Generate a cylindrical mesh.
        
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            center: Base center coordinates (x, y, z)
            axis: Direction axis (normalized)
            
        Returns:
            Path to the generated mesh file
        """
        # Define parametric function for cylinder
        def cylinder_function(u, v):
            # u is height (0 to height)
            # v is angle (0 to 2π)
            # Normalize axis
            axis_norm = np.array(axis) / np.linalg.norm(axis)
            
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
        
        # Use gmsh built-in cylinder for better quality
        gmsh.initialize()
        gmsh.model.add(self.name)
        
        # Create cylinder aligned with provided axis
        cylinder = gmsh.model.occ.addCylinder(
            center[0], center[1], center[2], 
            axis[0]*height, axis[1]*height, axis[2]*height, 
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
    
    def create_potential_grid(self, spacing=2.0, buffer=20.0, max_potential=100.0, grid_file=None):
        """
        Generate a potential grid based on the parametric definition.
        
        Args:
            spacing: Grid spacing in mesh units
            buffer: Additional buffer space around the mesh
            max_potential: Maximum potential value
            grid_file: Output DX file path (default: {name}_potential.dx)
            
        Returns:
            Path to the generated DX file
        """
        if grid_file is None:
            grid_file = self.work_dir / f"{self.name}_potential.dx"
        
        # First, get the bounding box of the mesh
        gmsh.initialize()
        gmsh.open(str(self.mesh_file))
        
        # Get mesh bounds
        entities = gmsh.model.getEntities()
        bounding_box = gmsh.model.getBoundingBox(-1, -1)  # Get overall bounding box
        
        min_x, min_y, min_z, max_x, max_y, max_z = bounding_box
        gmsh.finalize()
        
        # Add buffer
        min_x -= buffer
        min_y -= buffer
        min_z -= buffer
        max_x += buffer
        max_y += buffer
        max_z += buffer
        
        # Calculate grid dimensions
        nx = int(np.ceil((max_x - min_x) / spacing))
        ny = int(np.ceil((max_y - min_y) / spacing))
        nz = int(np.ceil((max_z - min_z) / spacing))
        
        # Create grid
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        z = np.linspace(min_z, max_z, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize potential grid
        potential = np.zeros((nx, ny, nz))
        
        # Reopen GMSH with our model
        gmsh.initialize()
        gmsh.open(str(self.mesh_file))
        
        # For each grid point, check if it's inside the entity
        # Since checking each point is slow, we'll use vectorized distance calculation
        # if a parametric function is available
        if self.parametric_function is not None:
            # Get coordinates of all grid points
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            
            # Calculate distance to the parametric surface
            if self.entity_dim == 3:  # Volume
                # For volumes, check if points are inside
                inside = np.zeros(len(points), dtype=bool)
                
                # Process in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    coords = batch.flatten().tolist()
                    result = gmsh.model.isInside(self.entity_dim, self.entity_tag, coords)
                    inside[i:i+len(batch)] = np.array(result) > 0
                
                # Set potential based on inside/outside
                shaped_inside = inside.reshape(X.shape)
                potential[shaped_inside] = max_potential
            else:
                # For surfaces, calculate distance to surface
                # This is more complex and would be slow to do point by point
                # Here we'll use a simplified approach: sample the parametric surface
                # and calculate distances to the sample points
                
                # Sample the parametric surface
                u_samples = 100
                v_samples = 100
                u_vals = np.linspace(self.u_range[0], self.u_range[1], u_samples)
                v_vals = np.linspace(self.v_range[0], self.v_range[1], v_samples)
                
                surface_points = []
                for u in u_vals:
                    for v in v_vals:
                        surface_points.append(self.parametric_function(u, v))
                
                surface_points = np.array(surface_points)
                
                # Calculate minimum distance for each grid point
                distances = np.zeros(len(points))
                
                # Process in batches
                batch_size = 1000
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    batch_distances = np.min(np.sqrt(np.sum((batch[:, np.newaxis, :] - surface_points[np.newaxis, :, :])**2, axis=2)), axis=1)
                    distances[i:i+len(batch)] = batch_distances
                
                # Reshape distances and set potential
                shaped_distances = distances.reshape(X.shape)
                potential = max_potential * np.exp(-shaped_distances / (spacing * 2))
        else:
            # If no parametric function is available, use a simpler approach
            print("Warning: No parametric function available, using simplified potential")
            # Just create a simple distance-based potential
            center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
            radius = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2
            
            # Calculate distance from center
            distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            
            # Set potential
            potential = max_potential * np.exp(-(distances - radius) / (spacing * 2))
            potential[distances < radius] = max_potential
        
        # Write DX file
        origin = [min_x, min_y, min_z]
        delta = [spacing, spacing, spacing]
        writeDx(str(grid_file), potential, origin, delta)
        
        return grid_file
    
    def create_rigid_body_type(self, density=19.3, simconf=None, output_dx=True):
        """
        Create a RigidBodyType directly from the generated mesh.
        
        Args:
            density: Material density in g/cm^3
            simconf: SimConf object with parameters
            output_dx: Whether to generate a potential DX file
            
        Returns:
            A MeshRigidBodyType object ready for simulation
        """
        from .mesh_rigidbody import MeshRigidBodyType
        
        # Generate potential grid if requested
        if output_dx:
            potential_dx = self.create_potential_grid()
            grid_path = str(potential_dx)
        else:
            grid_path = None
        
        return MeshRigidBodyType(
            name=self.name,
            mesh_file=str(self.mesh_file),
            density=density,
            simconf=simconf,
            unit_scale=self.unit_scale,
            potential_grid=grid_path
        )
    
    def visualize(self, show=True):
        """
        Visualize the generated mesh and potential grid if available.
        
        Args:
            show: Whether to show the visualization window
            
        Returns:
            PyVista plotter object
        """
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista not found. Please install with: pip install pyvista")
            return None
        
        # Create plotter
        plotter = pv.Plotter()
        
        # Add mesh
        if os.path.exists(self.mesh_file):
            mesh = pv.read(str(self.mesh_file))
            plotter.add_mesh(mesh, color='lightblue', style='surface', opacity=0.7)
        
        # Look for potential grid
        potential_dx = self.work_dir / f"{self.name}_potential.dx"
        if os.path.exists(potential_dx):
            # Read DX grid
            grid = pv.read(str(potential_dx))
            # Add contours
            contours = grid.contour(10)
            plotter.add_mesh(contours, color='red', line_width=2, opacity=0.5)
        
        # Add axes
        plotter.add_axes()
        plotter.camera_position = 'xy'
        
        if show:
            plotter.show()
        
        return plotter
