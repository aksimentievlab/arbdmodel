import os
import subprocess
import numpy as np
from pathlib import Path
from .logger import logger
from .engine import HydroProRunner, APBSRunner
from .core_objects import RigidBodyType
from .grid import writeDx, loadGrid, Bound_grid
from .engine import TclScriptGenerator
#Originally SimpleARBD by Chun


class PdbProcessor:
    """
    Process molecular structure files to calculate properties and generate maps for ARBD
    Common Processor class for both diffusive and static rigidbody
    """
    
    def __init__(self, structure_path, simconf=None, work_dir=None, tcl_path=None):
        """
        Initialize self with structure file
        
        Args:
            structure_path: Path to structure file (.psf/.pdb)
            simconf: SimConf object containing configuration parameters
            num_heavy_cluster: Number of heavy atom clusters for VDW maps
            work_dir: Working directory, should be either rbs or static (as enviromental potential)
        """

        self.structure_path = Path(structure_path)
        self.base_name = self.structure_path.stem
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        # Create working directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
        
        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()
            
        # Extract parameters from simconf
        self.simconf = simconf
        self.temperature = simconf.temperature
        self.viscosity = simconf.viscosity
        self.solvent_density = simconf.solvent_density
        self.num_heavy_cluster = simconf.num_heavy_cluster

        self.vmd_path = simconf.get_binary('vmd') or 'vmd'
        self.hydro_path = simconf.get_binary('hydropro')
        self.apbs_path = simconf.get_binary('apbs') or 'apbs'
        
        # Attributes for results
        self.mass = None
        self.moment_of_inertia = None
        self.transdamp = None
        self.rotdamp = None
        self.aligned_pdb = None
        self.aligned_psf = None
        self.charge_dx = None
        self.elec_dx = None
        self.vdw_pot_dxs = []
        self.vdw_den_dxs = []
        if tcl_path is None:
            tcl_path=Path.cwd().absolute()
        self.tclgen=TclScriptGenerator(work_dir=tcl_path)
        
    def process_diffusive_structure(self):
        """Process structure to get all properties and potential maps"""
        # Step 1: Align structure
        self.align_structure()
        
        # Step 2: Calculate hydrodynamic properties
        self.calculate_hydrodynamic_properties()
        
        # Step 3: Generate charge distribution
        self.generate_charge_distribution()
        
        # Step 4: Generate electrostatic map
        self.generate_electrostatic_map()
        
        # Step 5: Generate VDW maps
        self.generate_vdw_maps()
        
        # Step 6: Apply Gaussian smoothing
        self.apply_gaussian_smoothing()
        
        # Return a dictionary of grid files for use in RigidBodyType
        return self.get_grid_files()
        
    def align_structure(self):
        """Align structure to principal axes using VMD."""
        # Write alignment TCL script
        align_tcl = self.work_dir / "align.tcl"
        if not align_tcl.exists():
            align_tcl = self.tclgen.write_align_tcl()
            logger.debug(f"Alignment script written to {align_tcl}")
        

        # Run alignment
        try:
            # Copy input files to work directory if they're not already there
            input_psf = self.structure_path.with_suffix('.psf')
            input_pdb = self.structure_path.with_suffix('.pdb')
            
            # Make sure the base name matches for both files
            if input_psf.stem != input_pdb.stem:
                logger.warning(f"PSF and PDB file names don't match: {input_psf.name} and {input_pdb.name}")
            
            # Check and create symlinks if needed
            if not (self.work_dir / input_psf.name).exists():
                if os.path.isabs(str(input_psf)):
                    os.symlink(input_psf, self.work_dir / input_psf.name)
                else:
                    relative_path = os.path.relpath(input_psf, self.work_dir)
                    os.symlink(relative_path, self.work_dir / input_psf.name)
                    
            if not (self.work_dir / input_pdb.name).exists():
                if os.path.isabs(str(input_pdb)):
                    os.symlink(input_pdb, self.work_dir / input_pdb.name)
                else:
                    relative_path = os.path.relpath(input_pdb, self.work_dir)
                    os.symlink(relative_path, self.work_dir / input_pdb.name)
            
            # Run VMD
            cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.base_name} < {align_tcl}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Verify alignment succeeded
            self.aligned_pdb = self.work_dir / f"{self.base_name}.aligned.pdb"
            self.aligned_psf = self.work_dir / f"{self.base_name}.aligned.psf"
            
            if not self.aligned_pdb.exists() or not self.aligned_psf.exists():
                raise FileNotFoundError(f"Alignment failed: {self.aligned_pdb} or {self.aligned_psf} not found")
            
            # Read mass and inertia
            mass_file = self.work_dir / f"{self.base_name}.mass.txt"
            inertia_file = self.work_dir / f"{self.base_name}.inertia.txt"
            
            if not mass_file.exists() or not inertia_file.exists():
                raise FileNotFoundError(f"Mass or inertia file not found after alignment")
                
            with open(mass_file) as f:
                self.mass = float(f.readline().strip())
                
            with open(inertia_file) as f:
                self.moment_of_inertia = [float(x) for x in f.readline().strip().split()]
                
            logger.info(f"Structure aligned: Mass = {self.mass}, Inertia = {self.moment_of_inertia}")
        except Exception as e:
            logger.error(f"Error during alignment: {e}")
            raise
        
    def calculate_hydrodynamic_properties(self):
        """Calculate hydrodynamic properties using HydroPro."""
        if not self.hydro_path:
            logger.warning("HydroPro executable not provided, using default values")
            self.transdamp = [1.0, 1.0, 1.0]
            self.rotdamp = [1.0, 1.0, 1.0]
            return
            
        # Initialize HydroPro runner
        
        hydro_runner = HydroProRunner(
                mass=self.mass,
                simconf=self.simconf,
                structure_name=self.base_name)
            
            # Write config
        hydro_runner.write_config(output_path=self.work_dir / "hydropro.dat")
        self.transdamp, self.rotdamp=hydro_runner.run_calculation(self.work_dir)
                
        logger.info(f"Hydrodynamic properties: trans_damp={self.transdamp}, rot_damp={self.rotdamp}")
            
    
    def generate_charge_distribution(self, resolution=2.0):
        """Generate charge distribution using VMD."""
        aligned_name = f"{self.base_name}.aligned"
        aligned_path = str(self.work_dir / aligned_name)
        
        # Create TCL script for VMD
        charge_tcl =  Path("charge-density.tcl")
        if not charge_tcl.exists():
            charge_tcl = self.tclgen.write_charge_density_tcl(resolution=resolution)
            logger.debug(f"Charge density script written to {charge_tcl}")
        
        # Run VMD to generate charge density
        cmd = f"{self.vmd_path} -dispdev text -args {aligned_path} < {charge_tcl}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Check if charge distribution was created successfully
        charge_dx = self.work_dir / f"{aligned_name}.chargeDensity.dx"
        charge_out = self.work_dir / f"{aligned_name}.charge.dx"
        netcharge_file = self.work_dir / f"{aligned_name}.netCharge.dat"
        
        if not charge_dx.exists():
            raise FileNotFoundError(f"Charge density file not found: {charge_dx}")
        
        # Fix charge distribution - fix scientific notation in file
        temp_file1 = self.work_dir / "fix_charge_temp0.dx"
        temp_file2 = self.work_dir / "fix_charge_temp1.dx"
        
        # Handle numbers without decimal point in scientific notation
        cmd_in = f"sed -r 's/^([0-9]+)e/\\1.0e/g; s/ ([0-9]+)e/ \\1.0e/g' {str(charge_dx)} > {temp_file1}"
        os.system(cmd_in)
        cmd_in = f"sed -r 's/^(-[0-9]+)e/\\1.0e/g; s/ (-[0-9]+)e/ \\1.0e/g' {temp_file1} > {temp_file2}"
        os.system(cmd_in)
        
        # Read net charge
        with open(netcharge_file, 'r') as fin:
            netCharge = float(fin.readline().strip())
        
        # Load grid data
        grid, origin, delta = loadGrid(str(temp_file2))
        grid = grid * resolution**3
        
        # Fix charge to match net charge
        ids = np.where(np.abs(grid[:]) > 0.01)
        numPoints = np.size(ids)
        while np.abs(np.sum(grid) - netCharge) > 0.0001 and numPoints > 0:
            grid[ids] = grid[ids] + (netCharge - np.sum(grid)) / numPoints
        
        # Write output
        writeDx(str(charge_out), grid, origin, [delta, delta, delta])
        
        # Clean up temporary files
        try:
            os.remove(temp_file1)
            os.remove(temp_file2)
        except OSError:
            pass
            
        self.charge_dx = charge_out
        self.charge_density_dx = charge_dx
        
        logger.info(f"Charge distribution generated with net charge: {np.sum(grid):.6f}")
    
    def generate_electrostatic_map(self, buffer=50):
        """Generate electrostatic potential map using APBS."""
        if not self.apbs_path:
            logger.warning("APBS executable not provided, skipping electrostatic calculations")
            return
        
        aligned_name = f"{self.base_name}.aligned"
        
        # Read system dimensions
        dimension_file = self.work_dir / f"{aligned_name}.dimension.dat"
        if not dimension_file.exists():
            raise FileNotFoundError(f"Dimension file not found: {dimension_file}")
            
        with open(dimension_file, 'r') as f:
            dimensions = [float(line.strip()) for line in f.readlines()]
            
        # Initialize APBS runner
        apbs_runner = APBSRunner(self.simconf)
        
        # Write APBS configuration using the runner
        apbs_runner.run_calculation(
            structure_name=aligned_name, 
            xyz_dims=dimensions,
            salt_conc=0.15,
            buffer=buffer
        )
        
        # Run APBS
        cmd = f"cd {self.work_dir} && {self.apbs_path} {aligned_name}.apbs"
        subprocess.run(cmd, shell=True, check=True)
        
        # Bound the potential values
        tmp_file = self.work_dir / f"{aligned_name}.elec.tmp.dx"
        out_file = self.work_dir / f"{aligned_name}.elec.dx"
        Bound_grid(
            inFile=tmp_file,
            outFile=out_file,
            lowerBound=-20,
            upperBound=20
        )
        
        self.elec_dx = out_file
        logger.info(f"Electrostatic map generated for {aligned_name}")
    
    
        
        # Run VMD to generate VDW maps
        cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {aligned_path} < {vdw_tcl}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Process all VDW maps
        self.vdw_pot_dxs = []
        self.vdw_den_dxs = []
        
        for i in range(self.num_heavy_cluster + 1):
            pot_file = self.work_dir / f"{aligned_name}.vdw{i}.pot.dx"
            den_file = self.work_dir / f"{aligned_name}.vdw{i}.den.dx"
            
            if not pot_file.exists():
                logger.warning(f"VDW potential file not found: {pot_file}")
                continue
                
            # Bound the potential values
            Bound_grid(
                inFile=pot_file,
                outFile=pot_file,
                lowerBound=-20,
                upperBound=20
            )
            
            # Store for later smoothing
            self.vdw_pot_dxs.append(pot_file)
            self.vdw_den_dxs.append(den_file)
        
        logger.info(f"VDW maps generated for {aligned_name}")

    def clustering(self,num_heavy_cluster,hyd="hyd.dat",tmp="tmp.dat"):
        import numpy as np
        from scipy.cluster import vq

        numClusters = num_heavy_cluster

        d = np.loadtxt('tmp.dat')
        d_hyd = np.loadtxt('hyd.dat')

        ## build new dataset with 'count' (d[:,2]) entries of each value
        d2 = [np.outer( np.ones((1,int(d[i,2]))) , d[i,:2] ) for i in range(d.shape[0])]
        d2_hyd = [np.outer( np.ones((1,int(d_hyd[i,2]))) , d_hyd[i,:2] ) for i in range(d_hyd.shape[0])]

        d2 = np.vstack( d2 )
        d2w = vq.whiten( d2 )              # normalize features

        d2_hyd = np.vstack( d2_hyd )
        d2w_hyd = vq.whiten( d2_hyd )              # normalize features


        ind = 0;
        scalebase = d2w[ind,:];
        while np.any(scalebase == 0):
            ind = ind + 1
            scalebase = d2w[ind,:]

            scale = d2[ind,:] / d2w[ind,:]

            ind = 0;
            scalebase = d2w_hyd[ind,:];
        while np.any(scalebase == 0):
            ind = ind + 1
            scalebase = d2w_hyd[ind,:]

            scale_hyd = d2_hyd[ind,:] / d2w_hyd[ind,:]

        ## perform cluster analysis
        np.random.seed(seed=42)
        codeBook,dist = vq.kmeans(d2w , numClusters)
        assignments, dists = vq.vq(d[:,:2], scale * codeBook)

        codeBook_hyd,dist_hyd = vq.kmeans(d2w_hyd , 1)
        assignments_hyd, dists_hyd = vq.vq(d_hyd[:,:2], scale_hyd * codeBook_hyd)

        assignments_total = np.concatenate((assignments, assignments_hyd + numClusters) , axis=None)
        codeBook_total = np.concatenate((scale*codeBook, scale_hyd*codeBook_hyd), axis=0)

        print(" ".join(["%d" % a for a in assignments_total]))
        print(" ".join(["%.3f" % c[0] for c in codeBook_total]))
        print(" ".join(["%.3f" % c[1] for c in codeBook_total]))

    def apply_gaussian_smoothing(self, gaussianWidth=2.5):
        """Apply Gaussian smoothing to all potential maps."""
        aligned_name = f"{self.base_name}.aligned"
        
        # Create a tcl script for VMD to do the smoothing
        smooth_tcl = self.work_dir / "smooth.tcl"
        if not smooth_tcl.exists():
            with open(smooth_tcl, 'w') as f:
                f.write(f'''
# Get input parameters
lassign ${{argv}} in_file out_file gaussianWidth
puts "Smoothing $in_file to $out_file with gaussian width $gaussianWidth"

# Load the volumetric data
mol new $in_file

# Apply smoothing
set molid [molinfo top]
set vol [molinfo $molid get {0}]
volutil smooth $vol -gaussiansigma $gaussianWidth

# Save the result
volutil save $vol $out_file

# Exit
quit
''')
            
        # Smooth electrostatic map
        if self.elec_dx and self.elec_dx.exists():
            smoothed_elec = self.work_dir / f"{aligned_name}.elec.smoothed.dx"
            cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.elec_dx} {smoothed_elec} {gaussianWidth} < {smooth_tcl}"
            subprocess.run(cmd, shell=True, check=True)
            self.elec_smoothed_dx = smoothed_elec
            
        # Smooth VDW potential maps
        self.vdw_smoothed_dxs = []
        for i, pot_file in enumerate(self.vdw_pot_dxs):
            if pot_file.exists():
                smoothed_pot = self.work_dir / f"{aligned_name}.vdw{i}.pot.smoothed.dx"
                cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {pot_file} {smoothed_pot} {gaussianWidth} < {smooth_tcl}"
                subprocess.run(cmd, shell=True, check=True)
                self.vdw_smoothed_dxs.append(smoothed_pot)
                
        logger.info(f"Applied Gaussian smoothing to potential maps (width={gaussianWidth})")

    def get_grid_files(self):
        """Get dictionary of grid files for use in RigidBodyType."""
        potential_grids = []
        charge_grids = []
        
        # Add electrostatic grid
        if hasattr(self, 'elec_smoothed_dx') and self.elec_smoothed_dx and self.elec_smoothed_dx.exists():
            potential_grids.append(("elec", str(self.elec_smoothed_dx), 0.59616195))
        
        # Add charge grid
        if self.charge_dx and self.charge_dx.exists():
            charge_grids.append(("elec", str(self.charge_dx)))
        
        # Add VDW grids
        for i, pot_file in enumerate(self.vdw_smoothed_dxs):
            vdw_key = f"vdw{i}"
            if pot_file.exists():
                potential_grids.append((vdw_key, str(pot_file), 0.59616195))
                
        for i, den_file in enumerate(self.vdw_den_dxs):
            vdw_key = f"vdw{i}"
            if den_file.exists():
                charge_grids.append((vdw_key, str(den_file)))
        
        return {
            "potential_grids": potential_grids,
            "charge_grids": charge_grids
        }

    def get_rb_type(self):
        """Get RigidBodyType for the processed structure."""
        """Process structure to get all properties and potential maps"""
        # Step 1: Align structure
        self.align_structure()
        
        # Step 2: Calculate hydrodynamic properties
        self.calculate_hydrodynamic_properties()
        
        # Step 3: Generate charge distribution
        self.generate_charge_distribution()
        
        # Step 4: Generate electrostatic map
        self.generate_electrostatic_map()
        
        # Step 5: Generate VDW maps
        self.generate_vdw_maps()
        
        # Step 6: Apply Gaussian smoothing
        self.apply_gaussian_smoothing()
        
        # Return a dictionary of grid files for use in RigidBodyType
        rb=RigidBodyType(
            name=self.base_name, 
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            damping_coefficient=self.transdamp,
            rotational_damping_coefficient=self.rotdamp,
            potential_grids=self.get_grid_files().get('potential_grids', []),
            charge_grids=self.get_grid_files().get('charge_grids', []),
            pmf_grids=[],)
        self.aligned_pdb = self.aligned_pdb
        self.aligned_psf = self.aligned_psf
        
        logger.info(f"Pdb Processor generated '{self.base_name}' as RigidBodyType successfully")
        return rb
    
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
    
    
    def process_standard(self):
        """Process the structure normally (without segmentation)"""
        logger.info(f"Processing static object: {self.name}")
        
        # Use static directory for output
        
        # Create processor and generate maps
        self.align_structure()
        
        
        # Step 3: Generate charge distribution
        self.generate_charge_distribution()
        
        # Step 4: Generate electrostatic map
        self.generate_electrostatic_map()
        
        # Step 5: Generate VDW maps
        self.generate_vdw_maps()
        
        # Step 6: Apply Gaussian smoothing
        self.apply_gaussian_smoothing()
        
        # Collect grid files from the processor
        grid_files = self.get_grid_files()
        
        # Store grids
        self.potential_grids = grid_files.get('potential_grids', [])
        self.charge_grids = grid_files.get('charge_grids', [])
        self.elec_grid = self.elec_smoothed_dx
    
    def process_gigantic(self):
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
            
            segment_processor.process_standard()
            
            # Collect grid maps from this segment
            grid_files = segment_processor.get_grid_files()
            
            # Store grids from this segment
            self.potential_grids.extend(grid_files.get('potential_grids', []))
            self.charge_grids.extend(grid_files.get('charge_grids', []))
            
            # Store the first electrostatic grid as the main one
            if hasattr(segment_processor, 'elec_smoothed_dx') and not self.elec_grid:
                self.elec_grid = segment_processor.elec_smoothed_dx
        
        logger.info(f"Completed processing gigantic static object: {self.name}")

    def get_grid_from_pdb(self,is_gigantic=False, threshold=300):
        """Process the structure file to create potential and density grids"""
        if not self.structure_path:
            logger.warning("No structure path provided for static object")
            return
        
        self.threshold=threshold
        if is_gigantic:
            self.process_gigantic()
        else:
            self.process_standard()
    