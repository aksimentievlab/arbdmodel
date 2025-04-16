#!/usr/bin/env python3
import re
import sys
import argparse
from pathlib import Path

from .rb_from_pdb import StructureRigidBodyModel, SimpleArbdEngine
from .sim_config import SimConf
from .logger import logger

class SimpleArbdConfig:
    """
    Parse and manage SimpleARBD configuration file.
    
    This class provides a modern, clean interface for reading SimpleARBD
    configuration files and setting up the simulation.
    """
    
    def __init__(self, config_path):
        """
        Initialize SimpleArbdConfig.
        
        Args:
            config_path: Path to the SimpleARBD configuration file
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
        Parse the SimpleARBD configuration file.
        
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
    
    def create_model(self) -> StructureRigidBodyModel:
        """
        Create a StructureRigidBodyModel from the configuration.
        
        Returns:
            StructureRigidBodyModel instance
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
        model = StructureRigidBodyModel(
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
    
    def create_engine(self) -> SimpleArbdEngine:
        """
        Create a SimpleArbdEngine from the configuration.
        
        Returns:
            SimpleArbdEngine instance
        """
        # Create the engine with appropriate configuration
        engine = SimpleArbdEngine(
            configuration=self.simconf,
            extra_bd_file_lines=''
        )
        
        return engine
        
    def setup_diffusible_objects(self, model: StructureRigidBodyModel):
        """
        Set up diffusible objects in the model.
        
        Args:
            model: StructureRigidBodyModel to add diffusible objects to
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
                random_seed=42,  # Fixed seed for reproducibility
                )
            
    def setup_static_objects(self, model: StructureRigidBodyModel):
        """
        Set up static objects in the model.
        
        Args:
            model: StructureRigidBodyModel to add static objects to
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
            
    def run_simulation(self, model: StructureRigidBodyModel, engine: SimpleArbdEngine):
        """
        Run the simulation.
        
        Args:
            model: StructureRigidBodyModel to simulate
            engine: SimpleArbdEngine to use for simulation
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


def simplearbd():
    """
    Main function to process SimpleARBD configuration file and run simulation.
    """
    parser = argparse.ArgumentParser(description='Process SimpleARBD configuration file')
    parser.add_argument('config_file', help='Path to SimpleARBD configuration file')
    parser.add_argument('--setup-only', action='store_true', help='Only set up the simulation, do not run it')
    args = parser.parse_args()
    
    # Parse configuration file
    try:
        config = SimpleArbdConfig(args.config_file)
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
