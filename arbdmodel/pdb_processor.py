import os
import subprocess
import numpy as np
from pathlib import Path
import tarfile
import tempfile
from .logger import logger
from .engine import HydroProRunner, APBSRunner
from .core_objects import RigidBodyType
from .grid import writeDx, loadGrid, Bound_grid
from .engine import TclScriptGenerator
import shutil
#Originally SimpleARBD by Chun


def _default_charmm_params_dir() -> Path:
    """Shared location for CHARMM toppar files so each PdbProcessor work_dir does not re-download."""
    env = os.environ.get("ARBD_CHARMM_PARAMS")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    return (base / "arbdmodel" / "charmm_toppar_c36_feb26").resolve()


class PdbProcessor:
    CHARMM_TOPPAR_URL = "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/toppar_c36_feb26.tgz"

    """
    Process molecular structure files to calculate properties and generate maps for ARBD
    Common Processor class for both diffusive and static rigidbody
    """
    
    def __init__(
        self,
        structure_path,
        simconf=None,
        work_dir=None,
        tcl_path=None,
        charmm_params_dir=None,
        hydrogen_cluster=False,
        **kwargs,
    ):  # remember to change to None
        """
        Initialize self with structure file
        
        Args:
            structure_path: Path to structure file (.psf/.pdb)
            simconf: SimConf object containing configuration parameters
            num_heavy_cluster: Number of heavy atom clusters for VDW maps
            work_dir: Working directory, should be either rbs or static (as enviromental potential)
            pot_resolution: Grid resolution for VDW potential maps (default 1 Å; high-res: 0.5)
            den_resolution: Grid resolution for VDW density maps (default 2 Å; high-res: 1)
            charge_resolution: Grid resolution for charge density maps (default 2 Å; high-res: 1)
            elec_resolution: Grid resolution for electrostatic potential maps (default 2 Å; high-res: 1)
            hydrogen_cluster: Whether to cluster hydrogen atoms (default False)
            charmm_params_dir: Directory with CHARMM .prm/.str/.rtf. If None, uses env
                ARBD_CHARMM_PARAMS or ~/.cache/arbdmodel/charmm_toppar_c36_feb26 (shared across runs).
        """

        if work_dir is None and "output_dir" in kwargs:
            work_dir = kwargs.pop("output_dir")
        name_override = kwargs.pop("name", None)
        num_heavy_cluster = kwargs.pop("num_heavy_cluster", None)
        
        self.structure_path = Path(structure_path)
        self.base_name = name_override or self.structure_path.stem
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        # Create working directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
        
        if simconf is None:
            logger.warning("No simulation configuration provided, using default values")
            from . import DefaultSimConf
            simconf = DefaultSimConf()
            
        # Extract parameters from simconf
        self.simconf = simconf
        self.temperature = simconf.temperature
        self.viscosity = simconf.viscosity
        self.solvent_density = simconf.solvent_density

        self.pot_resolution = simconf.pot_resolution
        self.den_resolution = simconf.den_resolution
        self.elec_resolution = simconf.elec_resolution
        self.hydrogen_cluster = hydrogen_cluster

        self.num_heavy_cluster = int(num_heavy_cluster) if num_heavy_cluster is not None else simconf.num_heavy_cluster
        if self.num_heavy_cluster is None:
            logger.warning("Number of heavy cluster not provided, using default value of 3")
            self.num_heavy_cluster = 3
        self.threshold = 300

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
        self.clustered_path = self.work_dir / "clustered.txt"
        self._lj_type_records = None
        self.tcl_path = Path(tcl_path).absolute() if tcl_path is not None else Path.cwd().absolute()
        if charmm_params_dir is not None:
            self.charmm_params_dir = Path(charmm_params_dir).expanduser().resolve()
        else:
            self.charmm_params_dir = _default_charmm_params_dir()
        self._ensure_charmm_params(self.charmm_params_dir)
        self.tclgen = TclScriptGenerator(work_dir=self.tcl_path, charmm_params_dir=self.charmm_params_dir)

    def _has_charmm_params(self, params_dir: Path) -> bool:
        """Return True when CHARMM parameter files already exist."""
        if not params_dir.exists():
            return False
        for suffix in (".prm", ".str", ".rtf"):
            if any(params_dir.rglob(f"*{suffix}")):
                return True
        return False

    def _ensure_charmm_params(self, params_dir: Path):
        """Populate params_dir with .prm/.str/.rtf from the official toppar tarball if missing."""
        if self._has_charmm_params(params_dir):
            logger.info(f"Using existing CHARMM parameters in {params_dir}")
            return

        params_dir.mkdir(parents=True, exist_ok=True)
        tar_path = params_dir / "toppar_c36_feb26.tgz"

        if not tar_path.exists():
            logger.info(f"Downloading CHARMM parameters to {tar_path}")
            subprocess.run(
                ["wget", "-O", str(tar_path), self.CHARMM_TOPPAR_URL],
                check=True,
            )
        else:
            logger.info(f"Using existing CHARMM archive at {tar_path}")

        copied = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(tmpdir_path)

            for suffix in ("*.prm", "*.str", "*.rtf"):
                for src in tmpdir_path.rglob(suffix):
                    dst = params_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        copied += 1

        if not self._has_charmm_params(params_dir):
            raise RuntimeError(f"CHARMM parameter setup failed in {params_dir}")

        logger.info(f"CHARMM parameters ready in {params_dir} (copied {copied} files)")
        
    def align_structure(self):
        """Align structure to principal axes using VMD."""
        # Write alignment TCL script
        align_tcl = self.tcl_path / "align.tcl"
        aligned_pdb = self.work_dir / f"{self.base_name}.aligned.pdb"
        aligned_psf = self.work_dir / f"{self.base_name}.aligned.psf"
        mass_file = self.work_dir / f"{self.base_name}.mass.txt"
        inertia_file = self.work_dir / f"{self.base_name}.inertia.txt"

        if not align_tcl.exists():
            align_tcl = self.tclgen.write_align_tcl()
            logger.debug(f"Alignment script written to {align_tcl}")
        
        if aligned_pdb.exists() and aligned_psf.exists() and mass_file.exists() and inertia_file.exists():
            logger.info(f"Structure already aligned: {aligned_pdb} and {aligned_psf}")
            self.aligned_pdb = self.work_dir / f"{self.base_name}.aligned.pdb"
            self.aligned_psf = self.work_dir / f"{self.base_name}.aligned.psf"
            with open(mass_file) as f:
                self.mass = float(f.readline().strip())
                
            with open(inertia_file) as f:
                self.moment_of_inertia = [float(x) for x in f.readline().strip().split()]
                
            logger.info(f"Structure aligned: Mass = {self.mass}, Inertia = {self.moment_of_inertia}")
            self.base_name=f"{self.base_name}.aligned"  # Reasign base name to aligned pdb and psf
            return
        
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
            
            # Verify alignment succeeded
            self.aligned_pdb = self.work_dir / f"{self.base_name}.aligned.pdb"
            self.aligned_psf = self.work_dir / f"{self.base_name}.aligned.psf"
            
            
            cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.base_name} < {align_tcl}"
            subprocess.run(cmd, shell=True, check=True)

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
                structure_name=self.base_name,
                inertia=self.moment_of_inertia)
            
            # Write config
        hydro_runner.write_config(output_path=self.work_dir / "hydropro.dat")
        self.transdamp, self.rotdamp=hydro_runner.run_calculation(self.work_dir)
                
        logger.info(f"Hydrodynamic properties: trans_damp={self.transdamp}, rot_damp={self.rotdamp}")
            
    
    def generate_charge_distribution(self):
        resolution=self.den_resolution

        if self.den_resolution is None:
            logger.warning("Den resolution not provided, using default value of 2")
            resolution = 2.0

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
        charge_out = self.work_dir / f"{aligned_name}.{resolution}A.charge.dx"
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
        writeDx(str(charge_out), grid, origin, delta)
        
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
        resolution=self.elec_resolution
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
            buffer=buffer,resolution=resolution)
        
        # Write APBS configuration using the runner
        apbs_runner.run_calculation(workdir=self.work_dir)
        
        # Run APBS
        #cmd = f"cd {self.work_dir} && {self.apbs_path} {aligned_name}.apbs"
        #subprocess.run(cmd, shell=True, check=True)
        out_file = self.work_dir / f"{aligned_name}.{resolution}A.elec.dx"
        Bound_grid(inFile=out_file, lowerBound=-20, upperBound=20)
        
        self.elec_dx = Path(out_file)
        logger.info(f"Electrostatic map generated for {aligned_name}")


    def _read_lj_types(self, lj_types_file):
        with open(lj_types_file, "r") as f:
            content = f.read()

        lj_types = []
        current_term = ""
        in_braces = False
        for char in content:
            if char == "{":
                in_braces = True
                continue
            if char == "}":
                in_braces = False
                if current_term:
                    lj_types.append(current_term.strip())
                    current_term = ""
                continue
            if char.isspace() and not in_braces:
                if current_term:
                    lj_types.append(current_term.strip())
                    current_term = ""
                continue
            current_term += char
        if current_term:
            lj_types.append(current_term.strip())
        return lj_types

    def collect_lj_type_records(self):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """Collect per-processor LJ statistics for model-level pooled clustering."""
        vdw_tcl = self.tcl_path / "vdw_cluster.tcl"
        if not vdw_tcl.exists():
            vdw_tcl = self.tclgen.generate_cluster_tcl(potResolution=potResolution, denResolution=denResolution)
            logger.debug(f"Clustering script written to {vdw_tcl}")

        cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {self.base_name} < {vdw_tcl}"
        logger.info(f"Running clustering command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        tmp_file = self.work_dir / "tmp.dat"
        hyd_file = self.work_dir / "hyd.dat"
        lj_types_file = self.work_dir / "LJTypes.txt"

        heavy_rows = np.loadtxt(tmp_file, ndmin=2) if tmp_file.exists() and tmp_file.stat().st_size > 0 else np.empty((0, 3))
        hyd_rows = np.loadtxt(hyd_file, ndmin=2) if hyd_file.exists() and hyd_file.stat().st_size > 0 else np.empty((0, 3))
        lj_types = self._read_lj_types(lj_types_file) if lj_types_file.exists() else []

        records = []
        all_rows = []
        for row in heavy_rows:
            all_rows.append(("heavy", row))
        for row in hyd_rows:
            all_rows.append(("hydrogen", row))

        if len(all_rows) != len(lj_types):
            raise ValueError(
                f"LJ type record count mismatch for {self.base_name}: "
                f"{len(all_rows)} rows vs {len(lj_types)} type blocks."
            )

        for idx, ((group, row), atom_types) in enumerate(zip(all_rows, lj_types)):
            records.append(
                {
                    "source": self.base_name,
                    "group": group,
                    "radius": float(row[0]),
                    "epsilon": float(row[1]),
                    "count": int(row[2]) if row.shape[0] > 2 else 1,
                    "types": atom_types.strip(),
                    "index": idx,
                }
            )
        self._lj_type_records = records
        return records

    @property
    def lj_type_records(self):
        """Cached LJ records used for model-level pooled clustering."""
        if self._lj_type_records is None:
            self._lj_type_records = self.collect_lj_type_records()
        return self._lj_type_records

    def generate_vdw_diffusive(self, cluster_file=None):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        vdw_tcl = self.tcl_path / "vdw_diffusive.tcl"
        
        if not vdw_tcl.exists():
            vdw_tcl = self.tclgen.generate_diffusive_tcl(potResolution=potResolution, denResolution=denResolution)
            logger.debug(f"Clustering script written to {vdw_tcl}")
        else:
            logger.info(f"Clustering script found at {vdw_tcl}")

        cluster_path = Path(cluster_file) if cluster_file is not None else self.clustered_path
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

        cmd = f"cd {self.work_dir} && {self.vmd_path} -dispdev text -args {self.base_name} {cluster_path} < {vdw_tcl}"
        subprocess.run(cmd, shell=True, check=True)

        self.vdw_pot_dxs = []
        self.vdw_den_dxs = []
        
        for i in range(self.num_heavy_cluster +int(self.hydrogen_cluster)):
            pot_file = self.work_dir / f"{self.base_name}.vdw{i}.{potResolution}A.pot.dx"
            den_file = self.work_dir / f"{self.base_name}.vdw{i}.{denResolution}A.den.dx"
            
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
    

    def get_grid_files(self):
        """Get dictionary of grid files for use in RigidBodyType (raw paths; smoothing is applied in rb_from_pdb)."""
        potential_grids = []
        charge_grids = []
        # APBS writes phi in kT/e; ARBD expects kcal/mol -> multiply by kT
        elec_scale = 0.001987204 * self.temperature

        if self.elec_dx and Path(self.elec_dx).exists():
            potential_grids.append(("elec", str(self.elec_dx), elec_scale))
            potential_grids.append(("neg_elec", str(self.elec_dx), -elec_scale))

        if self.charge_dx and self.charge_dx.exists():
            charge_grids.append(("elec", str(self.charge_dx)))

        for i, pot_file in enumerate(self.vdw_pot_dxs):
            vdw_key = f"vdw{i}"
            if pot_file.exists():
                # VDW pot.dx maps are already in kcal/mol
                potential_grids.append((vdw_key, str(pot_file), 1.0))

        for i, den_file in enumerate(self.vdw_den_dxs):
            vdw_key = f"vdw{i}"
            if den_file.exists():
                charge_grids.append((vdw_key, str(den_file)))

        return {
            "potential_grids": potential_grids,
            "charge_grids": charge_grids,
        }

    def preprocess_diffusive_structure(self):
        """Process structure fields that do not depend on pooled LJ clustering."""
        # Step 1: Align structure
        self.align_structure()
        # Step 2: Calculate hydrodynamic properties
        self.calculate_hydrodynamic_properties()
        # Step 3: Generate charge distribution
        self.generate_charge_distribution()
        # Step 4: Generate electrostatic map
        self.generate_electrostatic_map()

    def get_rb_type(self):
        """Get RigidBodyType for the processed structure."""
        logger.warning("Only use this if you only have 1 rb type")
        self.preprocess_diffusive_structure()
        if self.clustered_path.exists():
            self.generate_vdw_diffusive(cluster_file=self.clustered_path)
        else:
            self.vdw_pot_dxs = []
            self.vdw_den_dxs = []

        # Return a dictionary of grid files for use in RigidBodyType
        rb = RigidBodyType(
            name=self.base_name, 
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            damping_coefficient=self.transdamp,
            rotational_damping_coefficient=self.rotdamp,
            potential_grids=self.get_grid_files().get('potential_grids', []),
            charge_grids=self.get_grid_files().get('charge_grids', []),
            pmf_grids=[],
        )
        
        self.aligned_pdb = self.aligned_pdb
        self.aligned_psf = self.aligned_psf
        
        logger.info(f"Pdb Processor generated '{self.base_name}' as RigidBodyType successfully")
        return rb

    def get_rb_type_metadata_only(self):
        """Build rigid-body metadata before pooled LJ clustering."""
        self.preprocess_diffusive_structure()

        rb = RigidBodyType(
            name=self.base_name,
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            damping_coefficient=self.transdamp,
            rotational_damping_coefficient=self.rotdamp,
            potential_grids=self.get_grid_files().get('potential_grids', []),
            charge_grids=self.get_grid_files().get('charge_grids', []),
            pmf_grids=[],
        )
        logger.info(f"Pdb Processor generated metadata-only RB type for '{self.base_name}'")
        return rb

    def update_rb_type_grids(self, rb_type):
        """Refresh a RigidBodyType with current generated grid files."""
        grid_files = self.get_grid_files()
        rb_type.potential_grids = grid_files.get("potential_grids", [])
        rb_type.charge_grids = grid_files.get("charge_grids", [])

    def get_static_grids(self, is_gigantic=False, cluster_file=None):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """
        Process a static structure and return its grid files.
        Similar to get_rb_type() but for static objects.
        
        Args:
            is_gigantic: Whether to process as a gigantic object
            cluster_file: Shared clustered assignment file from pooled model-level clustering (required).

        Returns:
            dict: Dictionary containing:
                - potential_grids: List of (name, file, scale) tuples
                - charge_grids: List of (name, file) tuples
                - elec_grid: Path to electrostatic grid file (raw, unsmoothed)
        """
        if cluster_file is None:
            raise ValueError(
                "cluster_file is required for static grids — call RBContactModel.build_vdw_maps() first "
                "so a pooled cluster file exists."
            )
        self.clustered_path = Path(cluster_file)

        # Step 1: Align structure
        self.align_structure()

        # Step 2: Generate charge distribution
        self.generate_charge_distribution()

        # Step 3: Generate electrostatic map
        self.generate_electrostatic_map()

        # Step 4: Generate VDW maps
        if is_gigantic and hasattr(self, "segment_count"):
            self._process_gigantic_vdw(
                potResolution=potResolution,
                denResolution=denResolution,
                cluster_file=self.clustered_path,
            )
        else:
            if is_gigantic:
                logger.warning(
                    "Gigantic static VDW requested without segmentation metadata; "
                    "falling back to standard static VDW generation."
                )
            self._process_standard_vdw(
                potResolution=potResolution,
                denResolution=denResolution,
                cluster_file=self.clustered_path,
            )

        grid_files = self.get_grid_files()
        elec_grid = None
        if self.elec_dx and Path(self.elec_dx).exists():
            elec_grid = str(Path(self.elec_dx).resolve())

        self.aligned_pdb = self.aligned_pdb
        self.aligned_psf = self.aligned_psf

        logger.info(f"Pdb Processor generated static grids for '{self.base_name}' successfully")
        return {
            "potential_grids": grid_files.get("potential_grids", []),
            "charge_grids": grid_files.get("charge_grids", []),
            "elec_grid": elec_grid,
        }


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

    def process_static_vdw(self, is_gigantic=False):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """Process VDW maps for static objects"""

        # Generate maps based on size
        if is_gigantic:
            self._process_gigantic_vdw()
        else:
            self._process_standard_vdw()

    def _process_standard_vdw(self, cluster_file=None):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """Process standard static object VDW maps"""
        # Generate static VDW map script
        vdw_script = self.tclgen.generate_static_vdw_tcl(potResolution=potResolution, denResolution=denResolution)
        cluster_path = Path(cluster_file) if cluster_file is not None else self.clustered_path
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_path}")
        
        # Run VMD
        cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {self.base_name} {cluster_path} < {vdw_script}"
        subprocess.run(cmd, shell=True, check=True)
        
        self.vdw_pot_dxs = []
        self.vdw_den_dxs = []

        # Bound the grid values
        for i in range(self.num_heavy_cluster + 1):
            pot_file = self.work_dir / f"{self.base_name}.vdw{i}.{potResolution}A.pot.dx"
            if os.path.isfile(pot_file):
                Bound_grid(inFile=pot_file, lowerBound=-20, upperBound=20)
                self.vdw_pot_dxs.append(pot_file)
            den_file = self.work_dir / f"{self.base_name}.vdw{i}.{denResolution}A.den.dx"
            if os.path.isfile(den_file):
                self.vdw_den_dxs.append(den_file)

    def _process_gigantic_vdw(self, cluster_file=None):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """Process gigantic static object VDW maps with segmentation"""
        #!!!! Needs work. standard vdw works
        cluster_path = Path(cluster_file) if cluster_file is not None else self.clustered_path
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

        # First generate VDW maps for each segment
        for segment_idx in range(self.segment_count + 1):
            segment_name = f"{self.base_name}.stat_temp.{segment_idx}"
            
            # Generate VDW maps for this segment
            vdw_script = self.tclgen.write_vdw_map_generation_static(potResolution)
            cmd = f"cd {self.work_dir} && VMDNOCUDA=1 {self.vmd_path} -dispdev text -args {segment_name} {cluster_path} < {vdw_script}"
            subprocess.run(cmd, shell=True, check=True)

        # Write map gluing script
        glue_script = self.write_map_gluing_script()
        
        # Glue maps for each VDW cluster
        for i in range(self.num_heavy_cluster + 1):
            last_map = None
            
            # Process each segment
            for j in range(self.segment_count + 1):
                current_map = f"{self.base_name}.stat_temp.{j}.vdw{i}.{potResolution}A.pot.dx"
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
                final_out = self.work_dir / f"{self.base_name}.vdw{i}.{potResolution}A.pot.dx"
                
                shutil.copy(last_map, temp_out)
                Bound_grid(inFile=temp_out, outFile=final_out, lowerBound=-20, upperBound=20)

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

    def process_diffusive_structure(self):
        """SimpleARBD-compatible entrypoint for diffusible objects."""
        return self.get_rb_type()

    def process_structure(self, is_gigantic=False, threshold=300, cluster_file=None):
        potResolution=self.pot_resolution
        denResolution=self.den_resolution
        """SimpleARBD-compatible entrypoint for static-like processing."""
        self.threshold = threshold
        grid_files = self.get_static_grids(
            is_gigantic=is_gigantic,
            potResolution=potResolution,
            denResolution=denResolution,
            cluster_file=cluster_file,
        )
        self.potential_grids = grid_files.get("potential_grids", [])
        self.charge_grids = grid_files.get("charge_grids", [])
        self.elec_grid = grid_files.get("elec_grid")
        return grid_files

    def get_grid_from_pdb(self, is_gigantic=False, threshold=300, cluster_file=None):
        """Compatibility wrapper matching legacy SimpleARBD call sites."""
        return self.process_structure(
            is_gigantic=is_gigantic,
            potResolution=self.pot_resolution,
            denResolution=self.den_resolution,
            threshold=threshold,
            cluster_file=cluster_file,
        )
