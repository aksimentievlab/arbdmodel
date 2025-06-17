import os
import subprocess
import numpy as np
from pathlib import Path
from .logger import logger
from .engine import HydroProRunner, APBSRunner
from .core_objects import RigidBodyType
from .grid import writeDx, loadGrid, Bound_grid,smooth_grid
from .engine import TclScriptGenerator
import shutil
#Originally SimpleARBD by Chun


class PdbProcessor:
    """
    Process molecular structure files to calculate properties and generate maps for ARBD
    Common Processor class for both diffusive and static rigidbody
    """
    
    def __init__(self, structure_path, simconf=None, work_dir=None, tcl_path=None,charmm_params_dir="charmm_params",): #remember to change to None
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
        self.tcl_path=tcl_path
        self.tclgen=TclScriptGenerator(work_dir=tcl_path,charmm_params_dir=charmm_params_dir)
        
    def align_structure(self):
        """Align structure to principal axes using VMD."""
        # Write alignment TCL script
        align_tcl = self.tcl_path / "align.tcl"
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
            self.base_name=f"{self.base_name}.aligned"  # Reasign base name to aligned pdb and psf

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
        aligned_name = f"{self.base_name}"
        aligned_path = str(self.work_dir / aligned_name)
        
        # Create TCL script for VMD
        charge_tcl = self.tcl_path / "charge-density.tcl"
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
        
        aligned_name = f"{self.base_name}"
        
        # Read system dimensions
        dimension_file = self.work_dir / f"{aligned_name}.dimension.dat"
        
        with open(dimension_file, 'r') as f:
            dimensions = [float(line.strip()) for line in f.readlines()]
            
        # Initialize APBS runner
        apbs_runner = APBSRunner(structure_name=aligned_name, simconf=self.simconf,
            xyz_dims=dimensions,
            buffer=buffer)
        
        # Write APBS configuration using the runner
        apbs_runner.run_calculation(workdir=self.work_dir)
        
        # Run APBS
        #cmd = f"cd {self.work_dir} && {self.apbs_path} {aligned_name}.apbs"
        #subprocess.run(cmd, shell=True, check=True)
        
        out_file = self.work_dir / f"{aligned_name}.elec.dx"
        Bound_grid(inFile=out_file, lowerBound=-20, upperBound=20)
        
        self.elec_dx = Path(out_file)
        logger.info(f"Electrostatic map generated for {aligned_name}")


    def generate_vdw_diffusive(self,potResolution=1, denResolution=2):
        vdw_tcl = self.tcl_path / "vdw_diffusive.tcl"
        
        if not vdw_tcl.exists():
            vdw_tcl = self.tclgen.generate_diffusive_tcl(potResolution=potResolution, denResolution=denResolution)
            logger.debug(f"Clustering script written to {vdw_tcl}")

        cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.base_name} clustered.txt < {vdw_tcl}"
        subprocess.run(cmd, shell=True, check=True)

        self.vdw_pot_dxs = []
        self.vdw_den_dxs = []
        
        for i in range(self.num_heavy_cluster + 1):
            pot_file = self.work_dir / f"{self.base_name}.vdw{i}.pot.dx"
            den_file = self.work_dir / f"{self.base_name}.vdw{i}.den.dx"
            
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
        
        logger.info(f"VDW maps generated for {self.base_name}")
    

    def clustering(self, numClusters=3,potResolution=1, denResolution=2):
        vdw_tcl = self.tcl_path / "vdw_cluster.tcl"
        if not vdw_tcl.exists():
            vdw_tcl = self.tclgen.generate_cluster_tcl(potResolution=potResolution, denResolution=denResolution)
            logger.debug(f"Clustering script written to {vdw_tcl}")

        cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {self.base_name} < {vdw_tcl}"
        subprocess.run(cmd, shell=True, check=True)

        import numpy as np
        from scipy.cluster import vq
        lj_types_file=self.work_dir/ "LJTypes.txt"
        d = np.loadtxt(self.work_dir/ 'tmp.dat')
        d_hyd = np.loadtxt(self.work_dir/'hyd.dat')

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
        scalebase = d2w_hyd[ind,:]
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

        
        with open(lj_types_file, 'r') as f:
            content  = f.read()
            lj_types = []
            current_term = ""
            in_braces = False
            for char in content:
                if char == '{':
                    in_braces = True
                    continue
                elif char == '}':
                    in_braces = False
                    if current_term:
                        lj_types.append(current_term.strip())
                        current_term = ""
                    continue
                elif char.isspace() and not in_braces:
                    if current_term:
                        lj_types.append(current_term.strip())
                        current_term = ""
                    continue
                current_term += char
            
            if current_term:  # Add any remaining term
                lj_types.append(current_term.strip())
            
            # Process each term to get individual types

        type_array = {i: "" for i in range(len(set(assignments_total)))}

        for i, t in zip(assignments_total, lj_types):
            type_array[i] = type_array[i] + f" {t}"
        self.clustered_path= self.work_dir/ "clustered.txt"
        with open(self.clustered_path, 'w') as f:
            for i, (r, e) in enumerate(zip(codeBook_total[:, 0], codeBook_total[:, 1])):
                f.write(f"{r} {e}{type_array[i]}\n")

        print(" ".join(["%d" % a for a in assignments_total]))
        print(" ".join(["%.3f" % c[0] for c in codeBook_total]))
        print(" ".join(["%.3f" % c[1] for c in codeBook_total]))

    def apply_gaussian_smoothing(self, gaussianWidth=2.5):
        """Apply Gaussian smoothing to all potential maps."""
        aligned_name = f"{self.base_name}"
        
        # Create a tcl script for VMD to do the smoothing
        smooth_tcl = self.tcl_path / "smooth.tcl"
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
# !!!! Check

# Exit
quit
''')
            
        # Smooth electrostatic map
        if self.elec_dx and Path(self.elec_dx).exists():
            smoothed_elec = f"{aligned_name}.elec.smoothed.dx"
            #cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.elec_dx} {smoothed_elec} {gaussianWidth} < {smooth_tcl}"
            #subprocess.run(cmd, shell=True, check=True)
            self.elec_smoothed_dx= smooth_grid(in_file=self.elec_dx, gaussian_sigma=gaussianWidth,  )
            self.elec_smoothed_dx = smoothed_elec
            
        # Smooth VDW potential maps
        self.vdw_smoothed_dxs = []
        for i, pot_file in enumerate(self.vdw_pot_dxs):
            if pot_file.exists():
                #smoothed_pot = self.work_dir / f"{aligned_name}.vdw{i}.pot.smoothed.dx"
                #cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {pot_file} {smoothed_pot} {gaussianWidth} < {smooth_tcl}"
                #subprocess.run(cmd, shell=True, check=True)
                self.vdw_smoothed_dxs.append(smooth_grid(in_file=pot_file, gaussian_sigma=gaussianWidth,  ))
                
        logger.info(f"Applied Gaussian smoothing to potential maps (width={gaussianWidth})")
        

    def get_grid_files(self):
        """Get dictionary of grid files for use in RigidBodyType."""
        potential_grids = []
        charge_grids = []
        
        # Add electrostatic grid
        if self.elec_smoothed_dx.exists():
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
        
        self.clustering()

        self.generate_vdw_diffusive()
        
        # Step 6: Apply Gaussian smoothing
        self.apply_gaussian_smoothing()
        
        # Return a dictionary of grid files for use in RigidBodyType
        """
        rb=RigidBodyType(
            name=self.base_name, 
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            damping_coefficient=self.transdamp,
            rotational_damping_coefficient=self.rotdamp,
            potential_grids=self.get_grid_files().get('potential_grids', []),
            charge_grids=self.get_grid_files().get('charge_grids', []),
            pmf_grids=[],)
        """
        
        self.aligned_pdb = self.aligned_pdb
        self.aligned_psf = self.aligned_psf
        
        logger.info(f"Pdb Processor generated '{self.base_name}' as RigidBodyType successfully")
        return rb

    def get_static_grids(self, is_gigantic=False, potResolution=1, denResolution=2):
        """
        Process a static structure and return its grid files.
        Similar to get_rb_type() but for static objects.
        
        Args:
            is_gigantic: Whether to process as a gigantic object
            
        Returns:
            dict: Dictionary containing:
                - potential_grids: List of (name, file, scale) tuples
                - charge_grids: List of (name, file) tuples
                - elec_grid: Path to electrostatic grid file
        """
        # Step 1: Align structure
        self.align_structure()
        
        # Step 2: Generate charge distribution
        self.generate_charge_distribution()
        
        # Step 3: Generate electrostatic map
        self.generate_electrostatic_map()
        
        # Step 4: Generate VDW maps
        
        if is_gigantic:
            self._process_gigantic_vdw(
                potResolution=potResolution,  # Double resolution for gigantic
                denResolution=denResolution
            )
        else:
            self._process_standard_vdw(
                potResolution=potResolution,
                denResolution=denResolution)
        
        # Step 5: Apply Gaussian smoothing
        self.apply_gaussian_smoothing()
        
        # Step 6: Collect and return grid files
        potential_grids = []
        charge_grids = []
        
        # Add electrostatic grid
        if hasattr(self, 'elec_smoothed_dx') and self.elec_smoothed_dx.exists():
            potential_grids.append(("elec", str(self.elec_smoothed_dx), 0.59616195))
            
        # Add charge grid
        if self.charge_dx and self.charge_dx.exists():
            charge_grids.append(("elec", str(self.charge_dx)))
            
        # Add VDW grids
        for i in range(self.num_heavy_cluster + 1):
            vdw_key = f"vdw{i}"
            pot_file = self.work_dir / f"{self.base_name}.vdw{i}.pot.dx"
            
            if pot_file.exists():
                potential_grids.append((vdw_key, str(pot_file), 0.59616195))
        
        logger.info(f"Generated static grids for {self.base_name}")
        return {
            "potential_grids": potential_grids,
            "charge_grids": charge_grids,
            "elec_grid": self.elec_smoothed_dx if hasattr(self, 'elec_smoothed_dx') else None}

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

    def process_static_vdw(self, potResolution=1, denResolution=2, is_gigantic=False):
        """Process VDW maps for static objects"""

        # Generate maps based on size
        if is_gigantic:
            self._process_gigantic_vdw(potResolution, denResolution)
        else:
            self._process_standard_vdw()

    def _process_standard_vdw(self):
        """Process standard static object VDW maps"""
        # Generate static VDW map script
        vdw_script = self.tclgen.generate_static_vdw_tcl()
        
        # Run VMD
        cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {self.base_name} clustered.txt < {vdw_script}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Bound the grid values
        for i in range(self.num_heavy_cluster + 1):
            pot_file = self.work_dir / f"{self.base_name}.vdw{i}.pot.dx"
            if os.path.isfile(pot_file):
                out_file = self.work_dir / f"{self.base_name}.vdw{i}.pot.final.dx"
                Bound_grid(pot_file, out_file, -20, 20)

    def _process_gigantic_vdw(self, potResolution, denResolution):
        """Process gigantic static object VDW maps with segmentation"""
        #!!!! Needs work. standard vdw works

        # First generate VDW maps for each segment
        for segment_idx in range(self.segment_count + 1):
            segment_name = f"{self.base_name}.stat_temp.{segment_idx}"
            
            # Generate VDW maps for this segment
            vdw_script = self.tclgen.write_vdw_map_generation_static(potResolution)
            cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {segment_name} clustered.txt < {vdw_script}"
            subprocess.run(cmd, shell=True, check=True)

        # Write map gluing script
        glue_script = self.write_map_gluing_script()
        
        # Glue maps for each VDW cluster
        for i in range(self.num_heavy_cluster + 1):
            last_map = None
            
            # Process each segment
            for j in range(self.segment_count + 1):
                current_map = f"{self.base_name}.stat_temp.{j}.vdw{i}.pot.dx"
                temp_map = f"{self.base_name}.stat_temp.{j}.vdw{i}.pot.tmp.dx"
                
                if j == 0:
                    # First segment - just copy
                    shutil.copy(current_map, temp_map)
                    last_map = temp_map
                else:
                    # Glue with previous map
                    cmd = f"VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {last_map} {current_map} {temp_map} < {glue_script}"
                    subprocess.run(cmd, shell=True, check=True)
                    last_map = temp_map
            
            # Create final map
            if os.path.isfile(last_map):
                temp_out = self.work_dir / f"{self.base_name}.vdw{i}.pot.tmp.dx"
                final_out = self.work_dir / f"{self.base_name}.vdw{i}.pot.dx"
                
                shutil.copy(last_map, temp_out)
                Bound_grid(temp_out, final_out, -20, 20)

    def write_map_gluing_script(self):
        """Write TCL script for gluing maps"""
        script_path = self.work_dir / "glue_maps.tcl"
        
        script_content = '''set prefixes $argv
set InMap1 [lindex $prefixes 0]
set InMap2 [lindex $prefixes 1]
set OutMap [lindex $prefixes 2]

voltool add -i1 $InMap1 -i2 $InMap2 -union -nointerp -o $OutMap
'''
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        return script_path