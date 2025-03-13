import os
import numpy as np
from pathlib import Path
from scipy import signal
from .logger import logger
from . import RigidBody, RigidBodyType, SimConf
from . import ArbdModel, ArbdEngine
from .structure_from_pdb import StructureProcessor
from .coords import Generate_coordinates, Generate_spanning_vectors
from .grid import Create_null

"""Structure rigid body modeling module for ARBD.

This module provides classes for structure-based rigid body modeling in the ARBD package,
using a clean implementation that processes molecular structures into grid maps.
"""

def Find_segments_num(dimensions, threshold=300):
    """Find number of segments needed"""
    in_xyz = [float(elm) for elm in dimensions]
    segments = [np.ceil(elm / threshold) for elm in in_xyz]
    return segments[0], segments[1], segments[2]

class StructureRigidBodyType(RigidBodyType):
    """RigidBodyType subclass for structure-based rigid body objects"""
    
    def __init__(self, name, structure_path, simconf=None,
                 work_dir=None, **kwargs):
        """Initialize structure rigid body type from structure files.
        Args:
            name: Name identifier for this type
            structure_path: Path to structure file (.psf/.pdb)
            simconf: SimConf object containing configuration parameters
            work_dir: Directory to store processed files (default: current directory)
        """
        # Create work directory if specified
        if work_dir:
            work_dir = Path(work_dir)
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = Path.cwd() / name
            os.makedirs(work_dir, exist_ok=True)
        
        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()

        # Process the structure to get properties and grid maps
        processor = StructureProcessor(
            structure_path=structure_path,
            simconf=simconf,
            work_dir=work_dir)
        
        # Process the structure to get all properties and grid files
        processor.process_structure()
        
        # Initialize the parent class with collected data
        super().__init__(
            name=name, 
            mass=processor.mass,
            moment_of_inertia=processor.moment_of_inertia,
            damping_coefficient=processor.transdamp,
            rotational_damping_coefficient=processor.rotdamp,
            potential_grids=processor.potential_grids,
            charge_grids=processor.charge_grids,
            pmf_grids=[],
            **kwargs
        )
        
        # Store file paths for reference
        self.aligned_pdb = processor.aligned_pdb
        self.aligned_psf = processor.aligned_psf
        
        logger.info(f"StructureRigidBodyType '{name}' initialized successfully")


class StructureRigidBodyModel(ArbdModel):
    """Model class for structure-based rigid body simulations"""
    
    def __init__(self, diffusible_objects=None, static_objects=None, 
                 config=DefaultSimConf, use_boundary=True, 
                 boundary_params=None, **kwargs):
        """Initialize structure model.
        
        Args:
            diffusible_objects: List of StructureRigidBody instances for diffusible objects
            static_objects: List of StructureRigidBody instances for static objects
            cell_vectors: List of 3 cell basis vectors defining the simulation box
            cell_origin: Cell origin coordinates
            use_boundary: Whether to create a boundary potential
            boundary_params: Optional parameters for boundary potential (well_depth, resolution, etc.)
            **kwargs: Additional arguments passed to ArbdModel
        """
        # Calculate dimensions from cell vectors if provided
        if cell_vectors is not None:
            # Extract maximum extent along each dimension
            dimensions = [0, 0, 0]
            for i in range(3):
                for v in cell_vectors:
                    dimensions[i] += abs(v[i])
            
            # Add some buffer
            dimensions = [dim * 1.2 for dim in dimensions]
        else:
            # Default dimensions if cell vectors not provided
            dimensions = kwargs.pop('dimensions', (1000, 1000, 1000))
            
        # Initialize default cell vectors if not provided
        self.cell_vectors = cell_vectors or [
            [dimensions[0], 0, 0],
            [0, dimensions[1], 0],
            [0, 0, dimensions[2]]
        ]
        self.cell_origin = cell_origin or [0, 0, 0]
        
        super().__init__(children=[], dimensions=dimensions, **kwargs)
        
        self.diffusible_objects = []
        self.static_objects = []
        self.boundary_potential = None
        
        # Process boundary if requested
        if use_boundary:
            # Create boundary potential
            from .interactions import BoundaryPotential
            
            # Use provided boundary parameters or defaults
            bp_params = boundary_params or {}
            
            boundary = BoundaryPotential(
                cell_vectors=self.cell_vectors,
                cell_origin=self.cell_origin,
                well_depth=bp_params.get('well_depth', 1.0),
                resolution=bp_params.get('resolution', 2.0),
                blur=bp_params.get('blur', 5.0)
            )
            
            # Generate the boundary file and store it
            boundary_file = boundary.write_file(bp_params.get('output_file', 'boundary.dx'))
            self.boundary_potential = boundary
            self.add_nonbonded_interaction()

            
        # Add diffusible objects
        if diffusible_objects:
            for obj in diffusible_objects:
                self.add_diffusible_object(obj)
                
        # Add static objects
        if static_objects:
            for obj in static_objects:
                self.add_static_object(obj)
                

    
    def add_diffusible_object(self, obj, copies=1, positions=None):
        """Add a diffusible (mobile) object to the model.
        
        Args:
            obj: StructureRigidBody instance
            copies: Number of copies to add
            positions: Optional list of positions for each copy
        """
        if not isinstance(obj, StructureRigidBodyType):
            raise TypeError("Object must be a StructureRigidBody")
            
        # Add the first object
        self.diffusible_objects.append(obj)
        self.add(obj)
        
        # Add copies if requested
        if copies > 1:
            for i in range(1, copies):
                # Create position for the copy
                if positions and i < len(positions):
                    position = positions[i]
                else:
                    # Generate random position within the model bounds
                    position = np.random.uniform(-0.4, 0.4, 3) * self.dimensions
                    
                # Create a new copy
                copy_obj = RigidBody(
                    type_=obj.type_,
                    position=position,
                    orientation=obj.orientation,
                    name=f"{obj.name}_{i+1}")
                
                self.diffusible_objects.append(copy_obj)
                self.add(copy_obj)
    
    def add_static_object(self, obj):
        """Add a static (immobile) object to the model.
        Args:
            obj: StructureRigidBody instance
        """

        self.static_objects.append(obj)
        self.add(obj)

        self.add_nonbonded_interaction()
        
    def generate_initial_positions(self, num_copies_per_object, initial_region=None,
                                  random_seed=None):
        """Generate initial positions for all diffusible objects.
        
        Args:
            num_copies_per_object: Dict mapping object names to number of copies
            initial_region: Dict with vectors defining initial region 
            random_seed: Seed for random number generator
        """
        # Set up default initial region if not provided
        if initial_region is None:
            max_dim = max(self.dimensions)
            initial_region = {
                'bv1': [max_dim*0.8, 0, 0],
                'bv2': [0, max_dim*0.8, 0],
                'bv3': [0, 0, max_dim*0.8],
                'origin': [0, 0, 0]
            }
            
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Create initial positions for each object type
        positions = {}
        for obj in self.diffusible_objects:
            if obj.name not in positions and obj.name in num_copies_per_object:
                # Get object dimensions
                try:
                    dimensions = [100, 100, 100]  # Default if not available
                    
                    # Generate spanning vectors
                    bv1, bv2, bv3, n1, n2, n3 = Generate_spanning_vectors(
                        initial_region['bv1'], 
                        initial_region['bv2'], 
                        initial_region['bv3'],
                        dimensions
                    )
                    
                    # Generate coordinates
                    num_copies = num_copies_per_object[obj.name]
                    coords = Generate_coordinates(
                        bv1, bv2, bv3, n1, n2, n3, num_copies,
                        initial_region['origin'], random_seed or 0
                    )
                    
                    positions[obj.name] = coords
                except Exception as e:
                    logger.error(f"Error generating positions for {obj.name}: {e}")
                    
        return positions
        
    def prepare_for_simulation(self):
        """Prepare model for simulation."""
        # Initialize potential maps
        null_dx = "null.dx"
        if not os.path.exists(null_dx):
            Create_null(null_dx)
            
        # Apply boundary potential if present
        if self.boundary_potential:
            if len(self.boundary_potential) >= 2:
                grid_file = self.boundary_potential[1]
                if os.path.exists(grid_file):
                    # Add the boundary potential as a nonbonded interaction
                    # This makes it apply to all particles in the simulation
                    self.add_nonbonded_interaction(grid_file, None, None)
                else:
                    logger.warning(f"Boundary potential file not found: {grid_file}")
            
        # Prepare objects for simulation
        for obj in self.diffusible_objects + self.static_objects:
            # Ensure all grid files for this object exist
            try:
                for potential_grid in obj.type_.potential_grids:
                    if len(potential_grid) >= 2:
                        grid_file = potential_grid[1]
                        if not os.path.exists(grid_file):
                            logger.warning(f"Grid file not found: {grid_file}")
            except (AttributeError, IndexError) as e:
                logger.warning(f"Error checking grid files for {obj.name}: {e}")
        
        # Call parent method to complete preparation
        super().prepare_for_simulation()

class SimpleArbdEngine(ArbdEngine):
    """Enhanced ARBD engine with additional functionality for structure simulations"""
    
    def __init__(self, extra_bd_file_lines="", configuration=None, **conf_params):
        """Initialize SimpleArbdEngine.
        
        Args:
            extra_bd_file_lines: Additional lines for BD configuration file
            configuration: SimConf object
            **conf_params: Additional configuration parameters
        """
        super().__init__(extra_bd_file_lines, configuration, **conf_params)
        
    def write_simulation_files(self, model, output_name, configuration=None, **conf_params):
        """Write all simulation files.
        
        Args:
            model: ArbdModel to simulate
            output_name: Base name for output files
            configuration: SimConf object
            **conf_params: Additional configuration parameters
        """
        # Call parent method to write standard files
        super().write_simulation_files(model, output_name, configuration, **conf_params)
        
        # Additional functionality for structure rigid bodies
        if hasattr(model, 'diffusible_objects') and model.diffusible_objects:
            logger.info(f"Writing {len(model.diffusible_objects)} diffusible objects")
            
        if hasattr(model, 'static_objects') and model.static_objects:
            logger.info(f"Writing {len(model.static_objects)} static objects")
            
    def run_simulation(self, model, output_name, replicas=1, gpu=0, **kwargs):
        """Run ARBD simulation.
        
        Args:
            model: ArbdModel to simulate
            output_name: Base name for output files
            replicas: Number of replicas to run
            gpu: GPU index to use
            **kwargs: Additional arguments for simulate method
        """
        # Prepare output directory
        output_dir = kwargs.get('output_directory', 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Run simulations
        for i in range(replicas):
            replica_name = f"{output_name}_{i}" if replicas > 1 else output_name
            
            # Override GPU index if running multiple replicas
            if replicas > 1:
                kwargs['gpu'] = (gpu + i) % 8  # Assuming max 8 GPUs
                
            # Run simulation
            self.simulate(model, replica_name, **kwargs)