import os
import numpy as np
from pathlib import Path
from .logger import logger
from . import RigidBodyType, DefaultSimConf
from .pdb_processor import PdbProcessor

"""Structure rigid body modeling module for ARBD.

This module provides classes for structure-based rigid body modeling in the ARBD package,
using a clean implementation that processes molecular structures into grid maps.
"""
  

    
class DiffusiveRigidBodyType(RigidBodyType):
    """RigidBodyType subclass for structure-based rigid body objects"""
    
    def __init__(self, name, structure_path, simconf=None, **kwargs):

        charmm_params_dir = kwargs.pop("charmm_params_dir", None)
        work_dir = kwargs.pop("work_dir", None)
        num_heavy_cluster = kwargs.pop("num_heavy_cluster", None)

        # Parent model work_dir, or cwd
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()

        # Create the RB output directory within work_dir
        rb_dir = self.work_dir / "rbs" / name
        os.makedirs(rb_dir, exist_ok=True)
        
        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()

        # Process the structure to get properties and grid maps
        proc_kw = dict(
            structure_path=structure_path,
            simconf=simconf,
            work_dir=rb_dir,
            charmm_params_dir=charmm_params_dir,
        )
        if num_heavy_cluster is not None:
            proc_kw["num_heavy_cluster"] = num_heavy_cluster
        self.processor = PdbProcessor(**proc_kw)
        
        # Process only metadata first; VDW grids are populated later by model-level build_vdw_maps().
        self.processor.get_rb_type_metadata_only()
        
        # Initialize the parent class with collected data
        super().__init__(
            name=name, 
            mass=self.processor.mass,
            moment_of_inertia=self.processor.moment_of_inertia,
            damping_coefficient=self.processor.transdamp,
            rotational_damping_coefficient=self.processor.rotdamp,
            potential_grids=self.processor.get_grid_files().get('potential_grids', []),
            charge_grids=self.processor.get_grid_files().get('charge_grids', []),
            pmf_grids=[],
            **kwargs
        )
        
        # Store file paths for reference
        self.aligned_pdb = self.processor.aligned_pdb
        self.aligned_psf = self.processor.aligned_psf
        
        logger.info(f"DiffusiveRigidBodyType '{name}' initialized with metadata only")

    def finalize_grids(self, cluster_file, gaussian_width=2.5, potResolution=1, denResolution=2):
        """Generate VDW maps from pooled cluster file, smooth potentials, and refresh grid lists."""
        from .grid import smooth_grid

        p = self.processor
        p.generate_vdw_diffusive(
            cluster_file=cluster_file, potResolution=potResolution, denResolution=denResolution
        )
        if p.elec_dx and Path(p.elec_dx).exists():
            p.elec_dx = Path(smooth_grid(in_file=p.elec_dx, gaussian_sigma=gaussian_width))
        smoothed_pots = []
        for pot in p.vdw_pot_dxs:
            if pot.exists():
                smoothed_pots.append(Path(smooth_grid(in_file=pot, gaussian_sigma=gaussian_width)))
        p.vdw_pot_dxs = smoothed_pots

        grid_files = p.get_grid_files()
        self.potential_grids = grid_files["potential_grids"]
        self.charge_grids = grid_files["charge_grids"]

class StaticObject:
    """Class representing a static (immobile) object in the simulation"""
    
    def __init__(self, structure_path=None, name=None, simconf=None,
                 is_gigantic=False, threshold=300, work_dir=None, cluster_file=None,
                 pot_resolution=1, den_resolution=2, charmm_params_dir=None):
        """Initialize static object from a structure file.
        
        Args:
            structure_path: Path to structure file (.psf/.pdb)
            name: Name for this static object (defaults to structure filename)
            simconf: SimConf object containing configuration parameters
            work_dir: Working directory (defaults to current directory)
            is_gigantic: Whether this is a gigantic object requiring segmentation
            threshold: Size threshold for segmentation (if is_gigantic=True)
            cluster_file: Pooled LJ cluster file (set by RBContactModel.add() before process(); required for VDW grids)
            pot_resolution: Grid resolution for VDW potential maps (default 1 Å; high-res: 0.5)
            den_resolution: Grid resolution for VDW density maps (default 2 Å; high-res: 1)
            charmm_params_dir: Directory containing CHARMM .prm/.str/.rtf files.  If None the
                shared cache under ~/.cache/arbdmodel is used (may trigger a download).
        """
        self.structure_path = Path(structure_path) if structure_path else None
        self.name = name or (self.structure_path.stem if self.structure_path else "static_object")
        self.is_gigantic = is_gigantic
        self.threshold = threshold
        self.cluster_file = Path(cluster_file) if cluster_file is not None else None
        self.pot_resolution = pot_resolution
        self.den_resolution = den_resolution
        self.charmm_params_dir = Path(charmm_params_dir).expanduser().resolve() if charmm_params_dir else None
        
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

        # Grids are built in process(); RBContactModel.add() injects cluster_file then calls process().
    
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

    def smooth_grids(self, gaussian_width=2.5):
        """Gaussian-smooth electrostatic and VDW potential maps (call after process())."""
        from .grid import smooth_grid

        smoothed_potential = []
        for item in self.potential_grids:
            if len(item) == 3:
                name, path, scale = item
            else:
                name, path = item[0], item[1]
                scale = 1.0
            pth = Path(path)
            if pth.exists():
                out = smooth_grid(in_file=path, gaussian_sigma=gaussian_width)
                smoothed_potential.append((name, str(out), scale))
            else:
                smoothed_potential.append((name, path, scale) if len(item) == 3 else item)
        self.potential_grids = smoothed_potential
        for item in self.potential_grids:
            if item[0] == "elec":
                self.elec_grid = Path(item[1])
                break

    def _process_standard(self):
        """Process the structure normally (without segmentation)"""
        logger.info(f"Processing static object: {self.name}")

        processor = PdbProcessor(
            structure_path=self.structure_path,
            simconf=self.simconf,
            work_dir=self.work_dir,
            charmm_params_dir=self.charmm_params_dir,
        )

        grid_files = processor.get_grid_from_pdb(
            is_gigantic=False,
            threshold=self.threshold,
            cluster_file=self.cluster_file,
            potResolution=self.pot_resolution,
            denResolution=self.den_resolution,
        )

        # Store grids
        self.potential_grids = grid_files.get('potential_grids', [])
        self.charge_grids = grid_files.get('charge_grids', [])
        eg = grid_files.get("elec_grid")
        if eg:
            self.elec_grid = Path(eg)
        elif processor.elec_dx:
            self.elec_grid = Path(processor.elec_dx)
    
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
                work_dir=segment_dir,
                charmm_params_dir=self.charmm_params_dir,
            )
            
            grid_files = segment_processor.get_grid_from_pdb(
                is_gigantic=False,
                threshold=self.threshold,
                cluster_file=self.cluster_file,
                potResolution=self.pot_resolution,
                denResolution=self.den_resolution,
            )

            # Store grids from this segment
            self.potential_grids.extend(grid_files.get('potential_grids', []))
            self.charge_grids.extend(grid_files.get('charge_grids', []))

            if not self.elec_grid:
                eg = grid_files.get("elec_grid")
                if eg:
                    self.elec_grid = Path(eg)
                elif segment_processor.elec_dx:
                    self.elec_grid = Path(segment_processor.elec_dx)
        
        logger.info(f"Completed processing gigantic static object: {self.name}")
 