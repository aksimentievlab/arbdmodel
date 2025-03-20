import os
import numpy as np
from pathlib import Path
from .logger import logger
from . import RigidBody, RigidBodyType, DefaultSimConf
from . import ArbdModel, ArbdEngine
from .structure_from_pdb import StructureProcessor
from .coords import Generate_coordinates, Generate_spanning_vectors

"""Structure rigid body modeling module for ARBD.

This module provides classes for structure-based rigid body modeling in the ARBD package,
using a clean implementation that processes molecular structures into grid maps.
"""

class DiffusiveRigidBodyType(RigidBodyType):
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

class StaticObject:
    """Class representing a static (immobile) object in the simulation"""
    
    def __init__(self, structure_path=None, name=None, simconf=None, work_dir=None,
                 is_gigantic=False, threshold=300):
        """Initialize static object from a structure file.
        
        Args:
            structure_path: Path to structure file (.psf/.pdb)
            name: Name for this static object (defaults to structure filename)
            simconf: SimConf object containing configuration parameters
            work_dir: Directory to store processed files
            is_gigantic: Whether this is a gigantic object requiring segmentation
            threshold: Size threshold for segmentation (if is_gigantic=True)
        """
        self.structure_path = Path(structure_path) if structure_path else None
        self.name = name or (self.structure_path.stem if self.structure_path else "static_object")
        self.is_gigantic = is_gigantic
        self.threshold = threshold
        
        # Set up working directory
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path.cwd() / f"static_{self.name}"
        
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Default config if not provided
        if simconf is None:
            simconf = DefaultSimConf()
        
        self.simconf = simconf
        self.potential_grids = []
        self.charge_grids = []
        self.elec_grid = None
        
        # Process the structure if provided
        if structure_path:
            self.process()
    
    def _find_segments_num(self, dimensions, threshold=None):
        """Find number of segments needed for a gigantic object.
        
        Args:
            dimensions: List of dimensions [x, y, z]
            threshold: Size threshold for segmentation (defaults to self.threshold)
            
        Returns:
            Tuple of (nx, ny, nz) segment counts
        """
        if threshold is None:
            threshold = self.threshold
            
        in_xyz = [float(elm) for elm in dimensions]
        segments = [np.ceil(elm / threshold) for elm in in_xyz]
        return segments[0], segments[1], segments[2]
    
    def process(self):
        """Process the structure file to create potential and density grids"""
        if not self.structure_path:
            logger.warning("No structure path provided for static object")
            return
            
        if self.is_gigantic:
            self._process_gigantic()
        else:
            self._process_standard()
    
    def _process_standard(self):
        """Process the structure normally (without segmentation)"""
        logger.info(f"Processing static object: {self.name}")
        
        # Create processor and generate maps
        processor = StructureProcessor(
            structure_path=self.structure_path,
            simconf=self.simconf,
            work_dir=self.work_dir)
        
        processor.process_structure()
        
        # Collect grids from the processor
        grid_files = processor.get_grid_files()
        
        # Store grids
        self.potential_grids = grid_files.get('potential_grids', [])
        self.charge_grids = grid_files.get('charge_grids', [])
        if hasattr(processor, 'elec_smoothed_dx') and processor.elec_smoothed_dx:
            self.elec_grid = processor.elec_smoothed_dx
    
    def _process_gigantic(self):
        """Process a gigantic structure by segmentation"""
        from subprocess import run
        import MDAnalysis as mda
        
        logger.info(f"Processing gigantic static object: {self.name}")
        
        # Get dimensions by reading the structure file
        u = mda.Universe(str(self.structure_path))
        min_coords = u.atoms.positions.min(axis=0)
        max_coords = u.atoms.positions.max(axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate segmentation
        nx, ny, nz = self._find_segments_num(dimensions, self.threshold)
        logger.info(f"Segmenting into {nx}x{ny}x{nz} parts")
        
        # Create segment directory
        segments_dir = self.work_dir / "segments"
        os.makedirs(segments_dir, exist_ok=True)
        
        # Write VMD script for segmentation
        segment_script = self.work_dir / "segment.tcl"
        with open(segment_script, 'w') as f:
            f.write(f"""
# Segment structure
mol new {self.structure_path}
set all [atomselect top all]
set mM [measure minmax $all]
set dimAl [vecsub [lindex $mM 1] [lindex $mM 0]]
set min_x [expr [lindex [lindex $mM 0] 0] - 1]
set min_y [expr [lindex [lindex $mM 0] 1] - 1]
set min_z [expr [lindex [lindex $mM 0] 2] - 1]
set dx [expr ([lindex $dimAl 0] + 2)/{nx}]
set dy [expr ([lindex $dimAl 1] + 2)/{ny}]
set dz [expr ([lindex $dimAl 2] + 2)/{nz}]

set count 0
for {{set i 0}} {{$i < {nx}}} {{incr i}} {{
  for {{set j 0}} {{$j < {ny}}} {{incr j}} {{
    for {{set k 0}} {{$k < {nz}}} {{incr k}} {{
      set outName {segments_dir}/segment_$count
      set low_x [expr $min_x + $i * $dx]
      set low_y [expr $min_y + $j * $dy]
      set low_z [expr $min_z + $k * $dz]
      set up_x [expr $min_x + ($i+1) * $dx]
      set up_y [expr $min_y + ($j+1) * $dy]
      set up_z [expr $min_z + ($k+1) * $dz]
      set sel [atomselect top "(x > $low_x and x < $up_x) and (y > $low_y and y < $up_y) and (z > $low_z and z < $up_z)"]
      set sel_N [$sel num]
      if {{$sel_N > 0}} {{
        $sel writepqr $outName.pqr
        $sel writepsf $outName.psf
        $sel writepdb $outName.pdb
        set count [expr $count + 1]
      }}
    }}
  }}
}}
exit
""")
        
        # Run segmentation
        vmd_path = self.simconf.get_binary('vmd')
        if not vmd_path:
            raise ValueError("VMD binary not found in configuration")
            
        run([vmd_path, "-dispdev", "text", "-e", str(segment_script)], check=True)
        
        # Process each segment
        segments = list(segments_dir.glob("segment_*.pdb"))
        logger.info(f"Found {len(segments)} segments to process")
        
        # Process each segment
        for i, segment in enumerate(segments):
            segment_dir = segments_dir / segment.stem
            os.makedirs(segment_dir, exist_ok=True)
            
            logger.info(f"Processing segment {i+1}/{len(segments)}: {segment.stem}")
            
            segment_processor = StructureProcessor(
                structure_path=segment,
                simconf=self.simconf,
                work_dir=segment_dir)
            
            segment_processor.process_structure()
            
            # Collect grid maps from this segment
            grid_files = segment_processor.get_grid_files()
            
            # Store grids from this segment
            self.potential_grids.extend(grid_files.get('potential_grids', []))
            self.charge_grids.extend(grid_files.get('charge_grids', []))
            
            # Store the first electrostatic grid as the main one
            if hasattr(segment_processor, 'elec_smoothed_dx') and not self.elec_grid:
                self.elec_grid = segment_processor.elec_smoothed_dx
        
        logger.info(f"Completed processing gigantic static object: {self.name}")
    
class StructureRigidBodyModel(ArbdModel):
    """Model class for structure-based rigid body simulations"""
    
    def __init__(self, cell_vectors=None, cell_origin=None, 
                 dimensions=None, buffer_factor=1.2, configuration=None, use_boundary=True, 
                 boundary_params=None, **kwargs):
        """Initialize structure model.
        
        Args:
            diffusible_objects: List of StructureRigidBody instances for diffusible objects
            static_objects: List of StructureRigidBody instances for static objects
            cell_vectors: List of 3 cell basis vectors defining the simulation box
            cell_origin: Cell origin coordinates
            dimensions: Explicit dimensions for the simulation box (overrides cell_vectors)
            buffer_factor: Factor to scale dimensions derived from cell vectors
            configuration: Configuration object, defaults to DefaultSimConf
            use_boundary: Whether to create a boundary potential
            boundary_params: Optional parameters for boundary potential (well_depth, resolution, etc.)
            **kwargs: Additional arguments passed to ArbdModel
        """
        
        self.simconf = configuration or DefaultSimConf()
        self.diffusible_objects = []
        self.static_objects = []
        self.boundary_potential = None

        super().__init__(
            children=[], 
            cell_vectors=cell_vectors, 
            cell_origin=cell_origin,
            dimensions=dimensions,
            buffer_factor=buffer_factor,
            configuration=self.simconf,
            **kwargs
        )
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
            

    def add_diffusible_object(self, structure_path=None, rb_type=None, copies=1, positions=None, 
                         orientations=None, name=None, initial_region=None, random_seed=None,
                         work_dir=None):
        """Add a diffusible (mobile) rigid body type to the model.
        
        Args:
            structure_path: Path to structure files (.psf/.pdb) to create a StructureRigidBodyType
            rb_type: Existing RigidBodyType instance (if structure_path not provided)
            copies: Number of copies to add
            positions: Optional list of positions for each copy
            orientations: Optional list of orientations for each copy
            name: Optional base name for the rigid bodies
            initial_region: Optional dict with vectors defining initial region
            random_seed: Optional seed for random number generator
            work_dir: Optional working directory for structure processing
        
        Returns:
            List of created RigidBody instances
        """
        # Process the structure to create a RigidBodyType if structure_path provided
        if structure_path is not None:
            # Create a StructureRigidBodyType from the structure files
            if name is None:
                name = Path(structure_path).stem
                
            logger.info(f"Processing structure files for {name} from {structure_path}")
            
            # Create work directory
            if work_dir is None:
                work_dir = Path.cwd() / name
                os.makedirs(work_dir, exist_ok=True)
            
            # Create the RigidBodyType
            rb_type = DiffusiveRigidBodyType(
                name=name,
                structure_path=structure_path,
                simconf=self.simconf,
                work_dir=work_dir)
            
            logger.info(f"Created StructureRigidBodyType for {name}")
            
        elif rb_type is None:
            raise ValueError("Either structure_path or rb_type must be provided")
            
        if not isinstance(rb_type, RigidBodyType):
            raise TypeError("Object must be a RigidBodyType")
        
        # Create base name for rigid bodies of this type
        base_name = name or rb_type.name
        created_bodies = []
        
        # Generate positions if not provided
        if positions is None and copies > 0:
            # Set up default initial region if not provided
            if initial_region is None:
                max_dim = max(self.dimensions)
                initial_region = {
                    'bv1': [max_dim*0.8, 0, 0],
                    'bv2': [0, max_dim*0.8, 0],
                    'bv3': [0, 0, max_dim*0.8],
                    'origin': [0, 0, 0]
                }
                
            # Get dimensions for this rigid body type
            dimensions = [100, 100, 100]  # Default
            
            # Try to get actual dimensions from structure
            if hasattr(rb_type, 'aligned_pdb') and rb_type.aligned_pdb:
                try:
                    # Try to read dimensions from dimension.dat file in the same directory
                    dim_file = rb_type.aligned_pdb.parent / f"{rb_type.aligned_pdb.stem}.dimension.dat"
                    if dim_file.exists():
                        with open(dim_file) as f:
                            dimensions = [float(line.strip()) for line in f.readlines()]
                        logger.info(f"Using dimensions from {dim_file}: {dimensions}")
                except Exception as e:
                    logger.warning(f"Could not read dimensions from file: {e}")
            
            # Generate spanning vectors
            bv1, bv2, bv3, n1, n2, n3 = Generate_spanning_vectors(
                initial_region['bv1'], 
                initial_region['bv2'], 
                initial_region['bv3'],
                dimensions
            )
            
            # Set random seed if provided
            if random_seed is not None:
                prev_state = np.random.get_state()
                np.random.seed(random_seed)
            
            # Generate coordinates
            generated_coords = Generate_coordinates(
                bv1, bv2, bv3, n1, n2, n3, copies,
                initial_region['origin'], random_seed or 0
            )
            
            # Restore random state if we changed it
            if random_seed is not None:
                np.random.set_state(prev_state)
            
            # Store positions for later use
            self.initial_positions[rb_type.name] = generated_coords
            positions = generated_coords
            
            logger.info(f"Generated {copies} initial positions for {rb_type.name}")
        
        # Set up default orientation if none provided
        default_orientation = np.eye(3)
        
        # Add requested number of copies
        for i in range(copies):
            # Create position and orientation for this copy
            if positions is not None and i < len(positions):
                position = positions[i]
            else:
                # Generate random position within the model bounds
                position = np.random.uniform(-0.4, 0.4, 3) * self.dimensions
                
            if orientations is not None and i < len(orientations):
                orientation = orientations[i]
            else:
                orientation = default_orientation
            
            # Create a new rigid body
            rb_name = f"{base_name}_{i+1}" if copies > 1 else base_name
            rb = RigidBody(
                type_=rb_type,
                position=position,
                orientation=orientation,
                name=rb_name)
            
            # Add to model
            self.diffusible_objects.append(rb)
            self.add(rb)
            created_bodies.append(rb)
            
        return created_bodies

    def add_static_object(self,structure_path, work_dir=None,is_gigantic=False, threshold=300):
        """
        Adds a static object to the simulation.
        
        This method creates a StaticObject from the specified structure file, extracts its
        electrostatic and potential/charge grids, and adds them to the simulation as
        non-bonded interactions.
        
        Parameters
        ----------
        structure_path : str
            Path to the structure file of the static object.
        work_dir : str, optional
            Working directory for the static object.
        is_gigantic : bool, default=False
            Flag indicating whether the structure is exceptionally large.
        threshold : int, default=300
            Size threshold for processing large structures.
        
        Returns
        -------
        None
            The static object is added to the simulation environment.
        
        Note
        -----
        The method adds the created StaticObject to the `static_objects` list after
        setting up all necessary non-bonded interactions.
            """

        obj=StaticObject(structure_path=structure_path, work_dir=work_dir,
                         simconf=self.simconf,is_gigantic=is_gigantic,threshold=threshold)
        elec_grid = obj.elec_grid
        potential_grids = obj.potential_grids
        charge_grids = obj.charge_grids
        # First add electrostatic grid if available
        self.add_nonbonded_interaction(elec_grid)
        self.add_nonbonded_interaction(potential_grids)
        self.add_nonbonded_interaction(charge_grids)

        self.static_objects.append(obj)
    
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
            model: StructureRigidBodyModel to simulate
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
            model: StructureRigidBodyModel to simulate
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