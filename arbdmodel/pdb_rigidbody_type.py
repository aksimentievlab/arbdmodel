import os
from pathlib import Path
from .logger import logger
from . import RigidBodyType, DefaultSimConf
from .pdb_processor import PdbProcessor

    
class PdbRigidBodyType(RigidBodyType):
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
            logger.warning("No simconf provided, using default simconf")
            from . import DefaultSimConf
            simconf = DefaultSimConf()

        # Process the structure to get properties and grid maps
        proc_kw = dict(
            structure_path=structure_path,
            simconf=simconf,
            work_dir=rb_dir,
            charmm_params_dir=charmm_params_dir,
            pot_resolution=simconf.pot_resolution,
            den_resolution=simconf.den_resolution,
            elec_resolution=simconf.elec_resolution,
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

    def finalize_grids(self, cluster_file, gaussian_width=2.5):
        """Generate VDW maps from pooled cluster file, smooth potentials, and refresh grid lists."""
        from .grid import smooth_grid

        p = self.processor
        p.generate_vdw_diffusive(
            cluster_file=cluster_file)
        if p.elec_dx and Path(p.elec_dx).exists():
            p.elec_dx = Path(smooth_grid(in_file=p.elec_dx, gaussian_sigma=gaussian_width))
        smoothed_pots = []
        for pot in p.vdw_pot_dxs:
            if pot.exists():
                smoothed_pots.append(Path(smooth_grid(in_file=pot, gaussian_sigma=gaussian_width)))
        p.vdw_pot_dxs = smoothed_pots

        grid_files = p.get_grid_files()
        print(grid_files)
        self.potential_grids = grid_files["potential_grids"]
        self.charge_grids = grid_files["charge_grids"]
