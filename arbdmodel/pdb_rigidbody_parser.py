#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

from . import ArbdEngine
from .config import SimConf
from .logger import logger
from .pdb_rigidbody_type import PdbRigidBodyType



class PdbRBEngine(ArbdEngine):
    """Engine wrapper for contact-model simulations."""

    def __init__(self, extra_bd_file_lines="", configuration=None, **conf_params):
        super().__init__(extra_bd_file_lines, configuration, **conf_params)

    def write_simulation_files(self, model, output_name, configuration=None, **conf_params):
        super().write_simulation_files(model, output_name, configuration, **conf_params)

        if getattr(model, "diffusible_objects", None):
            logger.info(f"Writing {len(model.diffusible_objects)} diffusible objects")
        if getattr(model, "static_objects", None):
            logger.info(f"Writing {len(model.static_objects)} static objects")

    def run_simulation(self, model, output_name, replicas=1, gpu=0, **kwargs):
        output_dir = kwargs.get("output_directory", "output")
        os.makedirs(output_dir, exist_ok=True)

        for i in range(replicas):
            replica_name = f"{output_name}_{i}" if replicas > 1 else output_name
            if replicas > 1:
                kwargs["gpu"] = (gpu + i) % 8
            self.simulate(model, replica_name, **kwargs)


class PdbRBConfig:
    """Parse and manage contact-model configuration files."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._parse_config()
        self.simconf = self._create_simconf()

        from .binary_manager import initialize_binary_paths
        initialize_binary_paths()

    def _parse_config(self):
        logger.info(f"Parsing config file: {self.config_path}")
        config = {}
        with open(self.config_path) as f:
            text = f.read()

        match = re.search(r"Diffusible_objects:([ \w\.]+)", text)
        if match:
            config["diffusible_objects"] = match.group(1).strip().split()

        match = re.search(r"Static_objects \(Enter NA for no static object\):([ \w\.]+)", text)
        if match:
            value = match.group(1).strip()
            config["static_objects"] = [] if value == "NA" else value.split()

        parameter_patterns = {
            "salt_concentration": r"SaltConcentration:(\s*[0-9]*\.[0-9]*)",
            "temperature": r"Temperature \(K\):(\s*[0-9]*\.?[0-9]*)",
            "viscosity": r"Viscosity:(\s*[0-9]*\.?[0-9]*)",
            "solvent_density": r"Solvent_density:(\s*[0-9]*\.?[0-9]*)",
            "num_heavy_cluster": r"Number_of_heavy_cluster \(Integer\):(\s*[0-9]+)",
            "gaussian_width": r"GaussianWidth:(\s*[0-9]*\.?[0-9]*)",
            "skip_parametrizing_diffusible": r"Skip_parametrizing_diffusible \(Yes/No\):([ \w]+)",
            "gigantic_stat_objects": r"Gigantic_stat_objects \(Yes/No\):([ \w]+)",
            "python_path": r"Python_path:(\s*\S+)",
            "hydro_path": r"Hydro_path:(\s*\S+)",
            "apbs_path": r"Apbs_path:(\s*\S+)",
            "vmd_path": r"Vmd_path:(\s*\S+)",
            "parameters_folder": r"Parameters_folder:(\s*\S+)",
            "num_replicas": r"Num_replicas \(Integer\):(\s*[0-9]+)",
            "timestep": r"Timestep \(Float\):(\s*[0-9]*\.?[0-9]*)",
            "steps": r"Steps \(Integer\):(\s*[0-9]+)",
            "interactive": r"Interactive \(Yes/No\):([ \w]+)",
            "grid_path": r"Grid_path:(\s*\S+)",
            "well_depth": r"WellDepth \(Positive\):\s*([0-9]+[\.]*[0-9]*)",
            "well_resolution": r"WellResolution \(Positive\):\s*([0-9]+[\.]*[0-9]*)",
            "arbd_path": r"ARBD_path:(\s*\S+)",
            "simulation_path": r"Path_for_ARBD_simulations:(\s*\S+)",
            "pot_resolution": r"PotResolution:(\s*[0-9]*\.?[0-9]*)",
            "den_resolution": r"DenResolution:(\s*[0-9]*\.?[0-9]*)",
        }
        vector_patterns = {
            "cell_basis_vector1": r"CellBasisVector1:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "cell_basis_vector2": r"CellBasisVector2:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "cell_basis_vector3": r"CellBasisVector3:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "cell_origin": r"CellOrigin:\s*(-*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]*)",
            "initial_coor_basis_vector1": r"InitialCoorBasisVector1:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "initial_coor_basis_vector2": r"InitialCoorBasisVector2:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "initial_coor_basis_vector3": r"InitialCoorBasisVector3:\s*([0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]* [0-9]+[\.]*[0-9]*)",
            "initial_coor_origin": r"InitialCoorOrigin:\s*(-*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]* -*[0-9]+[\.]*[0-9]*)",
        }

        for param, pattern in parameter_patterns.items():
            match = re.search(pattern, text)
            if match:
                config[param] = match.group(1).strip()

        for param, pattern in vector_patterns.items():
            match = re.search(pattern, text)
            if match:
                config[param] = [float(x) for x in match.group(1).split()]

        match = re.search(r"Number_of_copies_per_object \(Integer\(s\)\):([ 0-9]+)", text)
        if match and "diffusible_objects" in config:
            copies = match.group(1).strip().split()
            config["copies_per_object"] = {
                obj: int(copies[i]) for i, obj in enumerate(config["diffusible_objects"]) if i < len(copies)
            }

        match = re.search(r"Extra_potentials_tags \(Path, vdw cluster group\):([\s\S]*)\n", text)
        if match:
            tags = re.findall(r"\((\S+\.dx,\s*\w+)\)", match.group(1))
            config["extra_potentials"] = []
            for tag in tags:
                parts = tag.split(",")
                config["extra_potentials"].append(
                    {"path": parts[0].strip(), "vdw_type": parts[1].strip()}
                )

        type_conversions = {
            "salt_concentration": float,
            "temperature": float,
            "viscosity": float,
            "solvent_density": float,
            "num_heavy_cluster": int,
            "gaussian_width": float,
            "num_replicas": int,
            "timestep": float,
            "steps": int,
            "well_depth": float,
            "well_resolution": float,
            "pot_resolution": float,
            "den_resolution": float,
        }
        for param, convert in type_conversions.items():
            if param in config:
                try:
                    config[param] = convert(config[param])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {param} to {convert.__name__}")

        bool_conversions = {
            "skip_parametrizing_diffusible": lambda x: x.lower() == "yes",
            "gigantic_stat_objects": lambda x: x.lower() == "yes",
            "interactive": lambda x: x.lower() == "yes",
        }
        for param, convert in bool_conversions.items():
            if param in config:
                try:
                    config[param] = convert(config[param])
                except (ValueError, TypeError, AttributeError):
                    logger.warning(f"Could not convert {param} to boolean")

        return config

    def _create_simconf(self):
        params = {
            "temperature": self.config.get("temperature", 300),
            "viscosity": self.config.get("viscosity", 0.01),
            "solvent_density": self.config.get("solvent_density", 1.0),
            "num_heavy_cluster": self.config.get("num_heavy_cluster", 3),
            "timestep": self.config.get("timestep", 0.0002),
            "num_steps": self.config.get("steps", 10000000),
            "output_period": 1000,
        }

        binary_paths = {
            "hydro_path": "hydro_path",
            "apbs_path": "apbs_path",
            "vmd_path": "vmd_path",
            "arbd_path": "arbd_path",
        }
        for config_key, simconf_key in binary_paths.items():
            if config_key in self.config:
                params[simconf_key] = self.config[config_key]

        return SimConf(**params)

    def create_model(self):
        cell_vectors = None
        cell_origin = None

        if all(
            key in self.config for key in ("cell_basis_vector1", "cell_basis_vector2", "cell_basis_vector3")
        ):
            cell_vectors = [
                self.config["cell_basis_vector1"],
                self.config["cell_basis_vector2"],
                self.config["cell_basis_vector3"],
            ]
        if "cell_origin" in self.config:
            cell_origin = self.config["cell_origin"]

        return PdbRBModel(
            cell_vectors=cell_vectors,
            cell_origin=cell_origin,
            configuration=self.simconf,
            use_boundary="extra_potentials" in self.config and len(self.config["extra_potentials"]) > 0,
            boundary_params={
                "well_depth": self.config.get("well_depth", 1.0),
                "resolution": self.config.get("well_resolution", 2.0),
            },
            gaussian_width=self.config.get("gaussian_width", 2.5),
            num_heavy_cluster=self.config.get("num_heavy_cluster", self.simconf.num_heavy_cluster or 3),
            pot_resolution=self.config.get("pot_resolution", 1),
            den_resolution=self.config.get("den_resolution", 2),
        )

    def create_engine(self):
        return ContactModelEngine(configuration=self.simconf, extra_bd_file_lines="")

    def setup_diffusible_objects(self, model):
        if "diffusible_objects" not in self.config:
            logger.warning("No diffusible objects specified in configuration")
            return

        initial_region = None
        if all(
            key in self.config
            for key in (
                "initial_coor_basis_vector1",
                "initial_coor_basis_vector2",
                "initial_coor_basis_vector3",
                "initial_coor_origin",
            )
        ):
            initial_region = {
                "bv1": self.config["initial_coor_basis_vector1"],
                "bv2": self.config["initial_coor_basis_vector2"],
                "bv3": self.config["initial_coor_basis_vector3"],
                "origin": self.config["initial_coor_origin"],
            }

        for obj_name in self.config["diffusible_objects"]:
            if self.config.get("skip_parametrizing_diffusible", False):
                logger.info(f"Skipping parametrization for {obj_name} (as requested in config)")
                continue

            copies = self.config.get("copies_per_object", {}).get(obj_name, 1)
            psf_file = Path(f"{obj_name}.psf")
            pdb_file = Path(f"{obj_name}.pdb")
            if not (psf_file.exists() and pdb_file.exists()):
                logger.warning(f"Structure files for {obj_name} not found: {psf_file}, {pdb_file}")
                continue

            model.add_diffusible_object(
                structure_path=psf_file,
                copies=copies,
                name=obj_name,
                initial_region=initial_region,
            )

    def setup_static_objects(self, model):
        if "static_objects" not in self.config or not self.config["static_objects"]:
            logger.info("No static objects specified in configuration")
            return

        for obj_name in self.config["static_objects"]:
            psf_file = Path(f"{obj_name}.psf")
            pdb_file = Path(f"{obj_name}.pdb")
            if not (psf_file.exists() and pdb_file.exists()):
                logger.warning(f"Structure files for static object {obj_name} not found: {psf_file}, {pdb_file}")
                continue

            is_gigantic = self.config.get("gigantic_stat_objects", False)
            work_dir = Path(self.config.get("parameters_folder", "./parameters")) / f"static_{obj_name}"
            model.add_static_object(
                structure_path=psf_file,
                work_dir=work_dir,
                is_gigantic=is_gigantic,
                threshold=300,
            )

    def run_simulation(self, model, engine):
        sim_path = self.config.get("simulation_path", "./simulation")
        output_dir = Path(sim_path) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        replicas = self.config.get("num_replicas", 1)

        engine.run_simulation(
            model=model,
            output_name=Path(self.config_path).stem,
            replicas=replicas,
            output_directory=str(output_dir),
            directory=str(sim_path),
        )


def pdb_rb_parsing():
    parser = argparse.ArgumentParser(description="Process contact-model configuration file")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the simulation")
    args = parser.parse_args()

    try:
        cfg = PdbRBConfig(args.config_file)
    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        return 1

    model = cfg.create_model()
    engine = cfg.create_engine()
    cfg.setup_diffusible_objects(model)
    model.build_vdw_maps()
    cfg.setup_static_objects(model)
    model.generate_all_structures()

    if not args.setup_only:
        cfg.run_simulation(model, engine)
    else:
        logger.info("Setup complete. Simulation not started (--setup-only flag used)")

    return 0
