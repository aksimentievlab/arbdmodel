import os,subprocess
import numpy as np
from pathlib import Path
from .logger import logger
from . import RigidBody, RigidBodyType, DefaultSimConf
from .pdb_processor import PdbProcessor


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
        processor.process_diffusive_structure()
        
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
                 is_gigantic=False, threshold=300):
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
        
        # Use current directory as base
        self.work_dir = Path.cwd()
        
        # Create static output directory
        static_dir = self.work_dir / "static" / self.name
        os.makedirs(static_dir, exist_ok=True)
        
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
        
        # Use static directory for output
        static_dir = self.work_dir / "static" / self.name
        
        # Create processor and generate maps
        processor = PdbProcessor(
            structure_path=self.structure_path,
            simconf=self.simconf,
            work_dir=static_dir)  # Use the static object directory
        
        processor.process_structure()
        
        # Collect grid files from the processor
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
        
        # Use static directory for output
        static_dir = self.work_dir / "static" / self.name
        
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
  