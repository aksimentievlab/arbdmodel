#!/usr/bin/env python3
import re
import os
import argparse
from pathlib import Path
from . import ArbdEngine
from .model import ArbdModel
from .config import SimConf, DefaultSimConf
from .logger import logger
import numpy as np
from pathlib import Path
from .logger import logger
from .coords import Generate_coordinates, Generate_spanning_vectors
from .core_objects import RigidBodyType, RigidBody
from .pdb_processor import PdbProcessor
  
class RBContactModel(ArbdModel):
    """Model class for structure-based rigid body simulations"""
    
    def __init__(self, cell_vectors=None, cell_origin=None, 
                 dimensions=None, buffer_factor=1.2, configuration=None, use_boundary=False, 
                 num_heavy_cluster=3,charmm_params_dir=None,
                 boundary_params=None, **kwargs):
        """Initialize structure model Former SimpleARBD.
        
        Args:
            diffusible_objects: List of RBContact instances for diffusible objects
            static_objects: List of RBContact instances for static objects
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
        

    def add_diffusible_object(self, structure_path,  copies=1, positions=None, 
                         orientations=None, name=None, initial_region=None, random_seed=None):
        """Add a diffusible (mobile) rigid body type to the model.
        
        Args:
            structure_path: Path to structure files (.psf/.pdb) to create a RBContactType
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

            # Create a RBContactType from the structure files
        if name is None:
            name = Path(structure_path).stem
        logger.info(f"Processing structure files for {name} from {structure_path}")
            
            # Create the RigidBodyType using rbs/name directory
        rb_dir = self.work_dir / "rbs" / name
        os.makedirs(rb_dir, exist_ok=True)
        processor = PdbProcessor(
            structure_path=structure_path,
            work_dir=rb_dir,
            name=name,
            simconf=self.simconf,
            num_heavy_cluster=self.num_heavy_cluster,
        )
        
        rb_type = processor.process_diffusive_structure()
            
        logger.info(f"Created RBType for {name}")
            
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
        
        This method processes the specified static structure and keeps its
        electrostatic and potential/charge grids for static-field usage.
        
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
        os.makedirs(static_dir, exist_ok=True)
        
        # Create the static object with static/{name} output directory
        obj = PdbProcessor(
            structure_path=structure_path,
            work_dir=static_dir,
            name=name,
            simconf=self.simconf,
            num_heavy_cluster=self.num_heavy_cluster,
        )
        
        obj.get_grid_from_pdb(is_gigantic=is_gigantic, threshold=threshold)
        """
        # Add potential grids to model
        for grid_type, grid_file, scale in obj.potential_grids:
            self.add_nonbonded_interaction(grid_type, grid_file, scale)
            
        # Add charge grids to model
        for grid_type, grid_file in obj.charge_grids:
            self.add_nonbonded_interaction(grid_type, grid_file)
            
        # Add electrostatic grid if available
        if obj.elec_grid:
            self.add_nonbonded_interaction("elec", obj.elec_grid, 0.59616195)
        """
        # Store the static object
        self.static_objects.append(obj)
        return obj
 

class RBContactEngine(ArbdEngine):
    """Enhanced ARBD engine with additional functionality for structure simulations"""
    
    def __init__(self, extra_bd_file_lines="", configuration=None, **conf_params):
        """Initialize RBContactEngine.
        
        Args:
            extra_bd_file_lines: Additional lines for BD configuration file
            configuration: SimConf object
            **conf_params: Additional configuration parameters
        """
        super().__init__(extra_bd_file_lines, configuration, **conf_params)
        
    def write_simulation_files(self, model, output_name, configuration=None, **conf_params):
        """Write all simulation files.
        
        Args:
            model: RBContactModel to simulate
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
            model: RBContactModel to simulate
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

            


class RBContactConfig:
    """
    Parse and manage RBContact configuration file.
    
    This class provides a modern, clean interface for reading RBContact
    configuration files and setting up the simulation.
    """
    
    def __init__(self, config_path):
        """
        Initialize RBContactConfig.
        
        Args:
            config_path: Path to the RBContact configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        self.config = self._parse_config()
        self.simconf = self._create_simconf()
        from .binary_manager import initialize_binary_paths
        initialize_binary_paths()
        import ipdb
        ipdb.set_trace()
        
    def _parse_config(self):
        """
        Parse the RBContact configuration file.
        
        Returns:
            Dict containing configuration parameters
        """
        logger.info(f"Parsing config file: {self.config_path}")
        
        config = {}
        with open(self.config_path) as f:
            text = f.read()
            
        # Parse diffusible objects
        match = re.search(r'Diffusible_objects:([ \w\.]+)', text)
        if match:
            config['diffusible_objects'] = match.group(1).strip().split()
            
        # Parse static objects
        match = re.search(r'Static_objects \(Enter NA for no static object\):([ \w\.]+)', text)
        if match:
            val = match.group(1).strip()
            config['static_objects'] = [] if val == 'NA' else val.split()
            
        # Parse remaining configuration parameters
        parameter_patterns = {
            'salt_concentration': r'SaltConcentration:(\s*[0-9]*\.[0-9]*)',
            'temperature': r'Temperature \(K\):(\s*[0-9]*\.?[0-9]*)',
            'viscosity': r'Viscosity:(\s*[0-9]*\.?[0-9]*)',
            'solvent_density': r'Solvent_density:(\s*[0-9]*\.?[0-9]*)',
            'num_heavy_cluster': r'Number_of_heavy_cluster \(Integer\):(\s*[0-9]+)',
            'gaussian_width': r'GaussianWidth:(\s*[0-9]*\.?[0-9]*)',
            'skip_parametrizing_diffusible': r'Skip_parametrizing_diffusible \(Yes/No\):([ \w]+)',
            'gigantic_stat_objects': r'Gigantic_stat_objects \(Yes/No\):([ \w]+)',
            'python_path': r'Python_path:(\s*\S+)',
            'hydro_path': r'Hydro_path:(\s*\S+)',
            'apbs_path': r'Apbs_path:(\s*\S+)',
            'vmd_path': r'Vmd_path:(\s*\S+)',
            'parameters_folder': r'Parameters_folder:(\s*\S+)',
            'num_replicas': r'Num_replicas \(Integer\):(\s*[0-9]+)',
            'timestep': r'Timestep \(Float\):(\s*[0-9]*\.?[0-9]*)',
            'steps': r'Steps \(Integer\):(\s*[0-9]+)',
            'interactive': r'Interactive \(Yes/No\):([ \w]+)',
            'grid_path': r'Grid_path:(\s*\S+)',
            'well_depth': r'WellDepth \(Positive\):\s*([0-9]+[\.]*[0-9]*)',
            'well_resolution': r'WellResolution \(Positive\):\s*([0-9]+[\.]*[0-9]*)',
            'arbd_path': r'ARBD_path:(\s*\S+)',
            'simulation_path': r'Path_for_ARBD_simulations:(\s*\S+)',
        }
        
        # Extract cell vectors and origin
        vector_patterns = {
            'cell_basis_vector1': r'CellBasisVector1:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'cell_basis_vector2': r'CellBasisVector2:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'cell_basis_vector3': r'CellBasisVector3:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'cell_origin': r'CellOrigin:\s*(-*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]*)',
            'initial_coor_basis_vector1': r'InitialCoorBasisVector1:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'initial_coor_basis_vector2': r'InitialCoorBasisVector2:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'initial_coor_basis_vector3': r'InitialCoorBasisVector3:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)',
            'initial_coor_origin': r'InitialCoorOrigin:\s*(-*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]*)',
        }
        
        # Extract all parameters using regex
        for param, pattern in parameter_patterns.items():
            match = re.search(pattern, text)
            if match:
                config[param] = match.group(1).strip()
                
        # Extract and convert vector parameters
        for param, pattern in vector_patterns.items():
            match = re.search(pattern, text)
            if match:
                # Convert space-separated values to list of floats
                values = [float(x) for x in match.group(1).split()]
                config[param] = values
                
        # Extract copies per object
        match = re.search(r'Number_of_copies_per_object \(Integer\(s\)\):([ 0-9]+)', text)
        if match:
            copies = match.group(1).strip().split()
            if 'diffusible_objects' in config:
                config['copies_per_object'] = {
                    obj: int(copies[i]) for i, obj in enumerate(config['diffusible_objects'])
                    if i < len(copies)
                }
                
        # Extract extra potential tags
        match = re.search(r'Extra_potentials_tags \(Path, vdw cluster group\):([\s\S]*)\n', text)
        if match:
            tags = re.findall(r'\((\S+\.dx,\s*\w+)\)', match.group(1))
            config['extra_potentials'] = []
            for tag in tags:
                parts = tag.split(',')
                config['extra_potentials'].append({
                    'path': parts[0].strip(),
                    'vdw_type': parts[1].strip()
                })
                
        # Convert appropriate values to correct types
        type_conversions = {
            'salt_concentration': float,
            'temperature': float,
            'viscosity': float,
            'solvent_density': float,
            'num_heavy_cluster': int,
            'gaussian_width': float,
            'num_replicas': int,
            'timestep': float,
            'steps': int,
            'well_depth': float,
            'well_resolution': float,
        }
        
        for param, convert in type_conversions.items():
            if param in config:
                try:
                    config[param] = convert(config[param])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {param} to {convert.__name__}")
                    
        # Boolean conversions
        bool_conversions = {
            'skip_parametrizing_diffusible': lambda x: x.lower() == 'yes',
            'gigantic_stat_objects': lambda x: x.lower() == 'yes',
            'interactive': lambda x: x.lower() == 'yes',
        }
        
        for param, convert in bool_conversions.items():
            if param in config:
                try:
                    config[param] = convert(config[param])
                except (ValueError, TypeError, AttributeError):
                    logger.warning(f"Could not convert {param} to boolean")
                    
        return config
        
    def _create_simconf(self) -> SimConf:
        """
        Create a SimConf object from the parsed configuration.
        
        Returns:
            SimConf object
        """
        # Extract parameters for SimConf
        params = {
            'temperature': self.config.get('temperature', 300),
            'viscosity': self.config.get('viscosity', 0.01),
            'solvent_density': self.config.get('solvent_density', 1.0),
            'num_heavy_cluster': self.config.get('num_heavy_cluster', 3),
            'timestep': self.config.get('timestep', 0.0002),
            'num_steps': self.config.get('steps', 10000000),
            'output_period': 1000,  # Default
        }
        
        # Add binary paths if available
        binary_paths = {
            'hydro_path': 'hydro_path',
            'apbs_path': 'apbs_path',
            'vmd_path': 'vmd_path',
            'arbd_path': 'arbd_path'
        }
        
        for config_key, simconf_key in binary_paths.items():
            if config_key in self.config:
                params[simconf_key] = self.config[config_key]
                
        return SimConf(**params)
    
    def create_model(self) -> RBContactModel:
        """
        Create a RBContactModel from the configuration.
        
        Returns:
            RBContactModel instance
        """
        # Set up cell vectors and origin for model
        cell_vectors = None
        cell_origin = None
        
        if all(key in self.config for key in ['cell_basis_vector1', 'cell_basis_vector2', 'cell_basis_vector3']):
            cell_vectors = [
                self.config['cell_basis_vector1'],
                self.config['cell_basis_vector2'],
                self.config['cell_basis_vector3']
            ]
            
        if 'cell_origin' in self.config:
            cell_origin = self.config['cell_origin']
            
        # Create the model
        model = RBContactModel(
            cell_vectors=cell_vectors,
            cell_origin=cell_origin,
            configuration=self.simconf,
            use_boundary='extra_potentials' in self.config and len(self.config['extra_potentials']) > 0,
            boundary_params={
                'well_depth': self.config.get('well_depth', 1.0),
                'resolution': self.config.get('well_resolution', 2.0),
            }
        )
        
        return model
    
    def create_engine(self) -> RBContactEngine:
        """
        Create a RBContactEngine from the configuration.
        
        Returns:
            RBContactEngine instance
        """
        # Create the engine with appropriate configuration
        engine = RBContactEngine(
            configuration=self.simconf,
            extra_bd_file_lines=''
        )
        
        return engine
        
    def setup_diffusible_objects(self, model: RBContactModel):
        """
        Set up diffusible objects in the model.
        
        Args:
            model: RBContactModel to add diffusible objects to
        """
        if 'diffusible_objects' not in self.config:
            logger.warning("No diffusible objects specified in configuration")
            return
            
        # Create initial region from configuration
        initial_region = None
        if all(key in self.config for key in [
            'initial_coor_basis_vector1', 
            'initial_coor_basis_vector2', 
            'initial_coor_basis_vector3',
            'initial_coor_origin'
        ]):
            initial_region = {
                'bv1': self.config['initial_coor_basis_vector1'],
                'bv2': self.config['initial_coor_basis_vector2'],
                'bv3': self.config['initial_coor_basis_vector3'],
                'origin': self.config['initial_coor_origin']
            }
            
        # Add each diffusible object
        work_root = Path(self.config.get('parameters_folder', './parameters'))
            
        for obj_name in self.config['diffusible_objects']:
            # Skip parametrization if requested
            if self.config.get('skip_parametrizing_diffusible', False):
                logger.info(f"Skipping parametrization for {obj_name} (as requested in config)")
                continue
                
            # Determine number of copies
            copies = self.config.get('copies_per_object', {}).get(obj_name, 1)
            
            # Find structure files
            psf_file = Path(f"{obj_name}.psf")
            pdb_file = Path(f"{obj_name}.pdb")
            
            if not (psf_file.exists() and pdb_file.exists()):
                logger.warning(f"Structure files for {obj_name} not found: {psf_file}, {pdb_file}")
                continue
                
            logger.info(f"Adding diffusible object: {obj_name} with {copies} copies")
            
            # Create work directory
            work_dir = work_root / obj_name
            
            # Add to model
            model.add_diffusible_object(
                structure_path=psf_file,  # Use PSF as primary file
                copies=copies,
                name=obj_name,
                initial_region=initial_region,
                #random_seed=42,  # Fixed seed for reproducibility
                )
            
    def setup_static_objects(self, model: RBContactModel):
        """
        Set up static objects in the model.
        
        Args:
            model: RBContactModel to add static objects to
        """
        if 'static_objects' not in self.config or not self.config['static_objects']:
            logger.info("No static objects specified in configuration")
            return
            
        # Process each static object
        for obj_name in self.config['static_objects']:
            # Find structure files
            psf_file = Path(f"{obj_name}.psf")
            pdb_file = Path(f"{obj_name}.pdb")
            
            if not (psf_file.exists() and pdb_file.exists()):
                logger.warning(f"Structure files for static object {obj_name} not found: {psf_file}, {pdb_file}")
                continue
                
            # Determine if it's a gigantic object
            is_gigantic = self.config.get('gigantic_stat_objects', False)
            
            logger.info(f"Adding static object: {obj_name} (gigantic: {is_gigantic})")
            
            # Create work directory
            work_dir = Path(self.config.get('parameters_folder', './parameters')) / f"static_{obj_name}"
            
            # Add to model
            model.add_static_object(
                structure_path=psf_file,  # Use PSF as primary file
                work_dir=work_dir,
                is_gigantic=is_gigantic,
                threshold=300  # Default threshold
            )
            
    def run_simulation(self, model: RBContactModel, engine: RBContactEngine):
        """
        Run the simulation.
        
        Args:
            model: RBContactModel to simulate
            engine: RBContactEngine to use for simulation
        """
        # Set up output directory
        sim_path = self.config.get('simulation_path', './simulation')
        output_dir = Path(sim_path) / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of replicas
        replicas = self.config.get('num_replicas', 1)
        
        # Run simulation
        logger.info(f"Running simulation with {replicas} replicas")
        
        engine.run_simulation(
            model=model,
            output_name=Path(self.config_path).stem,
            replicas=replicas,
            output_directory=str(output_dir),
            directory=str(sim_path)
        )
   
def rbcontact():
    """
    Main function to process RBContact configuration file and run simulation.
    """
    parser = argparse.ArgumentParser(description='Process RBContact configuration file')
    parser.add_argument('config_file', help='Path to RBContact configuration file')
    parser.add_argument('--setup-only', action='store_true', help='Only set up the simulation, do not run it')
    args = parser.parse_args()
    
    # Parse configuration file
    try:
        config = RBContactConfig(args.config_file)
    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        return 1
        
    # Create model and engine
    model = config.create_model()
    engine = config.create_engine()
    
    # Set up diffusible and static objects
    config.setup_diffusible_objects(model)
    config.setup_static_objects(model)
    
    if not args.setup_only:
        # Run simulation
        config.run_simulation(model, engine)
    else:
        logger.info("Setup complete. Simulation not started (--setup-only flag used)")
    
    return 0
