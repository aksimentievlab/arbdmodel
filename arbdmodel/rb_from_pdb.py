import os,subprocess
import numpy as np
from pathlib import Path
from .logger import logger
from . import RigidBody, RigidBodyType, DefaultSimConf
from . import ArbdModel
from .pdb_processor import PdbProcessor
from .coords import Generate_coordinates, Generate_spanning_vectors

"""Structure rigid body modeling module for ARBD.

This module provides classes for structure-based rigid body modeling in the ARBD package,
using a clean implementation that processes molecular structures into grid maps.
"""
  

    
class DiffusiveRigidBodyType(RigidBodyType):
    """RigidBodyType subclass for structure-based rigid body objects"""
    
    def __init__(self, name, structure_path, simconf=None, **kwargs):

        # Use current directory if work_dir is not specified
        self.work_dir = Path.cwd()
        
        # Create the RB output directory within work_dir
        rb_dir = self.work_dir / "rbs" / name
        os.makedirs(rb_dir, exist_ok=True)
        
        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()

        # Process the structure to get properties and grid maps
        processor = PdbProcessor(
            structure_path=structure_path, simconf=simconf,
            work_dir=rb_dir)  # Pass the rigid body specific directory
        
        # Process the structure to get all properties and grid files
        processor.get_rb_type()
        
        # Initialize the parent class with collected data
        super().__init__(
            name=name, 
            mass=processor.mass,
            moment_of_inertia=processor.moment_of_inertia,
            damping_coefficient=processor.transdamp,
            rotational_damping_coefficient=processor.rotdamp,
            potential_grids=processor.get_grid_files().get('potential_grids', []),
            charge_grids=processor.get_grid_files().get('charge_grids', []),
            pmf_grids=[],
            **kwargs
        )
        
        # Store file paths for reference
        self.aligned_pdb = processor.aligned_pdb
        self.aligned_psf = processor.aligned_psf
        
        logger.info(f"StructureRigidBodyType '{name}' initialized successfully")

class StaticObject:
    """Class representing a static (immobile) object in the simulation"""
    
    def __init__(self, structure_path=None, name=None, simconf=None,
                 is_gigantic=False, threshold=300, work_dir=None):
        """Initialize static object from a structure file.
        
        Args:
            structure_path: Path to structure file (.psf/.pdb)
            name: Name for this static object (defaults to structure filename)
            simconf: SimConf object containing configuration parameters
            work_dir: Working directory (defaults to current directory)
            is_gigantic: Whether this is a gigantic object requiring segmentation
            threshold: Size threshold for segmentation (if is_gigantic=True)
        """
        self.structure_path = Path(structure_path) if structure_path else None
        self.name = name or (self.structure_path.stem if self.structure_path else "static_object")
        self.is_gigantic = is_gigantic
        self.threshold = threshold
        
        # Use provided work directory, else default to ./static/<name>
        if work_dir is None:
            self.work_dir = Path.cwd() / "static" / self.name
        else:
            self.work_dir = Path(work_dir)
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
        processor = PdbProcessor(
            structure_path=self.structure_path,
            simconf=self.simconf,
            work_dir=self.work_dir)
        
        grid_files = processor.get_grid_from_pdb(
            is_gigantic=False,
            threshold=self.threshold,
        )
        
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
        
        # Use static directory for output
        static_dir = self.work_dir
        
        # Get dimensions by reading the structure file
        u = mda.Universe(str(self.structure_path))
        min_coords = u.atoms.positions.min(axis=0)
        max_coords = u.atoms.positions.max(axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate segmentation
        nx, ny, nz = self._find_segments_num(dimensions, self.threshold)
        logger.info(f"Segmenting into {nx}x{ny}x{nz} parts")
        
        # Create segment directory
        segments_dir = static_dir / "segments"
        os.makedirs(segments_dir, exist_ok=True)
        
        # Write VMD script for segmentation
        segment_script = static_dir / "segment.tcl"
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
            
            segment_processor = PdbProcessor(
                structure_path=segment,
                simconf=self.simconf,
                work_dir=segment_dir)
            
            grid_files = segment_processor.get_grid_from_pdb(
                is_gigantic=False,
                threshold=self.threshold,
            )
            
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
                 dimensions=None, buffer_factor=1.2, configuration=None, use_boundary=False, 
                 num_heavy_cluster=3,
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
        self.initial_positions = {}  # Store initial positions for each type
        self.num_heavy_cluster = num_heavy_cluster
        
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
            logger.info("Creating boundary potential")
            # Create boundary potential
            from .interactions import BoundaryPotential
            
            # Use provided boundary parameters or defaults
            bp_params = boundary_params or {}
            
            boundary = BoundaryPotential(
                cell_vectors=cell_vectors,
                cell_origin=cell_origin,
                well_depth=bp_params.get('well_depth', 1.0),
                resolution=bp_params.get('resolution', 2.0),
                blur=bp_params.get('blur', 5.0)
            )
            
            # Generate the boundary file and store it
            boundary_file = boundary.write_file(bp_params.get('output_file', 'boundary.dx'))
            self.boundary_potential = boundary_file
            #self.add_nonbonded_interaction(self.boundary_potential)
        

    def add_diffusible_object(self, structure_path=None, rb_type=None, copies=1, positions=None, 
                         orientations=None, name=None, initial_region=None, random_seed=None):
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
        
        Returns:
            List of created RigidBody instances
        """
        # Process the structure to create a RigidBodyType if structure_path provided
        if structure_path is not None:
            # Create a StructureRigidBodyType from the structure files
            if name is None:
                name = Path(structure_path).stem
                
            logger.info(f"Processing structure files for {name} from {structure_path}")
            
            # Create the RigidBodyType using rbs/name directory
            rb_type = DiffusiveRigidBodyType(
                name=name,
                structure_path=structure_path,
                simconf=self.simconf)
            
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

    def add_static_object(self, structure_path, is_gigantic=False, threshold=300, work_dir=None):
        """
        Adds a static object to the simulation.
        
        This method creates a StaticObject from the specified structure file and stores
        its electrostatic and potential/charge grids for engine-side static-field usage.
        
        Parameters
        ----------
        structure_path : str
            Path to the structure file of the static object.
        is_gigantic : bool, default=False
            Flag indicating whether the structure is exceptionally large.
        threshold : int, default=300
            Size threshold for processing large structures.
        
        Returns
        -------
        StaticObject
            The created and added static object.
        """
        name = Path(structure_path).stem
        
        static_dir = Path(work_dir) if work_dir else self.work_dir / "static" / name

        # Create the static object with static/{name} output directory
        obj = StaticObject(
            structure_path=structure_path,
            name=name,
            simconf=self.simconf,
            is_gigantic=is_gigantic,
            threshold=threshold,
            work_dir=static_dir,
        )

        # Store the static object
        self.static_objects.append(obj)
        return obj
 