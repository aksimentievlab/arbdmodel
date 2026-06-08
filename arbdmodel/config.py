# -*- coding: utf-8 -*-
"""
Simulation configuration module. Provides SimConf and DefaultSimConf 
classes for configuring simulation parameters.
"""

import os
import numpy as np
from copy import copy, deepcopy
from .logger import logger, devlogger
from .binary_manager import BinaryManager

def _get_properties_and_dict_keys(obj):
    import inspect
    cls = obj.__class__

    def filter_props(name_type):
        nt = name_type
        return not nt[0].startswith('_') and isinstance(nt[1], property)
    properties = [name for name, type_ in filter(filter_props, inspect.getmembers(obj.__class__))]
    return properties + list(obj.__dict__.keys())

class SimConf:
    """
    Class describing properties for a simulation (ARBD or NAMD).
    This class stores various simulation parameters, manages binary paths, and provides
    methods to combine simulation configurations.

    Parameters:
        num_steps (int): Total number of simulation steps.
        output_period (int): Frequency of output generation.
        integrator (str): Type of integrator used (e.g., 'MD', 'BD').
        timestep (float): Simulation time step.
        thermostat (str): Type of thermostat used.
        barostat (str): Type of barostat used.
        temperature (float): Simulation temperature (must be positive).
        pressure (float): Simulation pressure.
        cutoff (float): Cutoff distance for interactions.
        pairlist_distance (float): Distance for generating pair lists.
        decomp_period (int): Period for decomposition.
        gpu (int): GPU device index to be used.
        seed (int): Random seed for reproducibility.
        restart_file (str): Path to restart file.

        # ARBD-specific parameters
        rigid_body_integrator (str): Type of rigid body integrator. Lagevin only.
        rigid_body_grid_grid_period (int): Period for rigid body grid updates.

        # SimpleARBD parameters
        viscosity (float): Solvent viscosity. 
        solvent_density (float): Solvent density. in g/cm^3

    """

    def __init__(self, num_steps=None, output_period=None,
                 integrator=None, timestep=None, thermostat=None, barostat=None,
                 temperature=None, pressure=None,
                 cutoff=None, pairlist_distance=None, decomp_period=None, gpu=None,
                 seed=None, restart_file=None,
                 ## ARBD-specific
                 rigid_body_integrator=None,
                 rigid_body_grid_grid_period=None,
                 ## SimpleARBD parameters
                 viscosity=None, solvent_density=None, salt=None,num_heavy_cluster=None,
                 den_resolution=None,
                 pot_resolution=None,
                 elec_resolution=None,
                 hydrogen_cluster=False,
                 ## Binary paths
                 arbd_path=None, namd_path=None, vmd_path=None, 
                 hydropro_path=None, apbs_path=None, **kwargs):


        self.num_steps = num_steps
        self.output_period = output_period

        self.integrator = integrator
        self.timestep = timestep
        self.thermostat = thermostat
        self.barostat = barostat
        self.num_heavy_cluster = num_heavy_cluster
        self.temperature = temperature
        self.pressure = pressure
        self.cutoff = cutoff
        self.pairlist_distance = pairlist_distance
        self.decomp_period = decomp_period
        self.seed = seed
        self.restart_file = restart_file
        self.gpu = gpu

        self.rigid_body_integrator = rigid_body_integrator
        self.rigid_body_grid_grid_period = rigid_body_grid_grid_period

        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.salt=salt

        self.den_resolution = den_resolution
        self.pot_resolution = pot_resolution
        self.elec_resolution = elec_resolution
        self.hydrogen_cluster = hydrogen_cluster

        # Set binary paths
        if arbd_path: BinaryManager.set_binary_path('arbd', arbd_path)
        if namd_path: BinaryManager.set_binary_path('namd', namd_path)
        if vmd_path: BinaryManager.set_binary_path('vmd', vmd_path)
        if hydropro_path: BinaryManager.set_binary_path('hydropro', hydropro_path)
        if apbs_path: BinaryManager.set_binary_path('apbs', apbs_path)

    def get_binary(self, name=None):
        """
        Get the path to a specific binary with improved error handling.
        
        Args:
            name: The name of the binary (e.g., 'arbd', 'hydropro')
                
        Returns:
            Path to the binary if found, None otherwise
            
        This method does not raise exceptions when binaries are not found, allowing
        for graceful handling of missing optional dependencies.
        """
        if name is None:
            binary_path = BinaryManager.export_to_dict()
            print(binary_path)
        else:
            binary_path = BinaryManager.get_binary_path(name)
            
            # Check if we found a binary and convert to string if needed
            if binary_path is not None:
                return str(binary_path)

            # Determine if this is an essential binary
            if BinaryManager.is_binary_essential(name):
                logger.warning(f"Essential binary '{name}' not found. Core functionality may be limited.")
        
    
    def set_binary(self, name, path):
        """
        Set the path to a specific binary.

        This method updates the path for a binary in the BinaryManager. It associates the given binary name
        with the specified file path.

        Parameters
        ----------
        name : str
            The name of the binary to set the path for.
        path : str
            The file system path to the binary.

        Returns
        -------
        str
            The path that was set.

        Examples
        --------
        >>> config.set_binary('ffmpeg', '/usr/local/bin/ffmpeg')
        '/usr/local/bin/ffmpeg'
        """
        """Set the path to a specific binary."""
        BinaryManager.set_binary_path(name, path)
        return path

    @property
    def temperature(self):
        return self.__temperature
    
    @temperature.setter
    def temperature(self, value):
        if value is not None and value <= 0:
            raise ValueError("Temperature must be positive")
        self.__temperature = value

    def combine(self, other, policy='override', warn=False):
        """
        Combines two SimConf objects into a new one based on a specified policy.

        This method creates a new SimConf object whose properties are initialized from 'self',
        but are potentially overridden with properties from 'other' according to the specified policy.

        Parameters
        ----------
        other : SimConf
            The SimConf object whose properties may override the properties of 'self'.
        policy : str, optional
            The policy to use when combining properties. Options are:
            - 'override': Always use the value from 'other' if it's not None (default).
            - 'best': Use the more appropriate value based on property-specific rules:
                - For 'timestep', 'output_period', 'decomp_period': uses the minimum value.
                - For 'num_steps', 'cutoff', 'pairlist_distance': uses the maximum value.
                - For 'integrator': prefers 'MD' over 'BD'.
                - For other attributes: uses the value from 'other' with a warning.
        warn : bool, optional
            If True, log warnings when attribute values differ and policy='best' (default: False).

        Returns
        -------
        SimConf
            A new SimConf object with combined properties.

        Raises
        ------
        ValueError
            If an unrecognized policy is specified.
        """
        new_conf = copy(self)
        for attr in _get_properties_and_dict_keys(other):
            oldval = None
            val = other.__getattribute__(attr)
            if val is not None:
                try:
                    oldval = self.__getattribute__(attr)
                except:
                    pass
                if oldval != val and (oldval is not None) and \
                   (val is not None) and policy != 'override':
                    if policy == 'best':
                        if attr in ('timestep', 'output_period', 'decomp_period'):
                            if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using {min([oldval,val])}')
                            new_conf.__setattr__(attr, min([oldval,val]))
                        elif attr in ('num_steps', 'cutoff', 'pairlist_distance'):
                            if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using {max([oldval,val])}')
                            new_conf.__setattr__(attr, max([oldval,val]))
                        elif attr == 'integrator':
                            if 'MD' in (oldval,val) and 'BD' in (oldval,val):
                                if warn: logger.warning(f'Combining attribute {attr}: {oldval} != {val}, using "MD"')
                                new_conf.__setattr__(attr,'MD')
                            else:
                                logger.warning(f'Unsure how to combine {oldval} and {val} for {attr} under policy {policy}; using {val}')
                                new_conf.__setattr__(attr, val)
                        else:
                            logger.warning(f'Unsure how to combine {oldval} and {val} for {attr} under policy {policy}; using {val}')
                            new_conf.__setattr__(attr, val)                            
                    else:
                        raise ValueError(f'Unrecognized policy "{policy}" for combining SimConfs')
                else:
                    new_conf.__setattr__(attr, val)
        return new_conf

    def items(self):
        for attr in _get_properties_and_dict_keys(self):
            val = self.__getattribute__(attr)
            yield attr, val

class DefaultSimConf(SimConf):
    """
    This class extends SimConf to provide default binary paths and configuration parameters
    for molecular dynamics simulations using ARBD.

    Parameters
    ----------
    num_steps : float, optional
        Number of simulation steps to run, default 1e5.
    output_period : float, optional
        Period for output generation, default 1e3.
    integrator : str, optional
        Integration method, default 'MD'.
    timestep : float, optional
        Simulation timestep in ns, default 20e-6.
    thermostat : str, optional
        Thermostat type, default 'Langevin'.
    barostat : str or None, optional
        Barostat type if any, default None.
    temperature : float, optional
        Simulation temperature in K, default 295.
    pressure : float, optional
        Pressure value for barostat if used, default 1.
    cutoff : float, optional
        Interaction cutoff distance, default 50.
    pairlist_distance : float or None, optional
        Pairlist buffer distance, default None.
    decomp_period : int, optional
        Period for domain decomposition updates, default 40.
    seed : int or None, optional
        Random seed for simulation, default None (random).
    restart_file : str or None, optional
        Path to restart file if continuing a simulation, default None.
    gpu : int, optional
        GPU device index to use, default 0.
    viscosity : float, optional
        Solvent viscosity, default 0.01.
    solvent_density : float, optional
        Density of the solvent, default 1.0.
    **kwargs
        Additional keyword arguments to pass to SimConf.

    Properties
    ----------
    temperature : float
        Simulation temperature with validation (must be positive).

    Notes
    -----
    The class automatically searches for binary dependencies like 'arbd', 'hydropro', 'apbs',
    'vmd', and 'namd', and adds their paths to the configuration if found.
    Essential binaries (currently only 'arbd') will generate warnings if not found.
    """
    def __init__(self, num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=20e-6, thermostat='Langevin', 
                 temperature=295, pressure=1,
                 cutoff=50,  decomp_period=40,restart_file=None, gpu=0,
                 viscosity=0.01, solvent_density=1.0, salt=0.15,num_heavy_cluster=3,
                 den_resolution=2,
                 pot_resolution=2,
                 elec_resolution=2,
                 hydrogen_cluster=False,
                 **kwargs):
        
        # Set default paths only for essential binaries or those that exist
        default_paths = {}
        essential_binaries = ["arbd"]  # These are required for basic functionality
        optional_binaries = ["hydropro", "apbs", "vmd", "namd"]  # These are optional
        
        # First add essential binaries
        for binary_name in essential_binaries:
            resource_path = BinaryManager.get_binary_path(binary_name)
            if resource_path:
                default_paths[f"{binary_name}_path"] = resource_path
            else:
                logger.warning(f"Essential binary '{binary_name}' not found. Some functionality may be limited.")
        
        # Then add optional binaries only if they exist
        for binary_name in optional_binaries:
            resource_path = BinaryManager.get_binary_path(binary_name)
            if resource_path:
                default_paths[f"{binary_name}_path"] = resource_path
        
        # Initialize with binary paths and other parameters
        SimConf.__init__(self, 
                         num_steps=num_steps, 
                         output_period=output_period,
                         integrator=integrator, 
                         timestep=timestep, 
                         thermostat=thermostat, 
                         temperature=temperature, 
                         pressure=pressure,
                         cutoff=cutoff, 
                         decomp_period=decomp_period,
                         restart_file=restart_file, 
                         gpu=gpu,
                         viscosity=viscosity,
                         solvent_density=solvent_density,
                         num_heavy_cluster=num_heavy_cluster,
                         salt=salt,
                         den_resolution=den_resolution,
                         pot_resolution=pot_resolution,
                         elec_resolution=elec_resolution,
                         hydrogen_cluster=hydrogen_cluster,
                         **{**default_paths, **kwargs})  # User-provided values override defaults

    @property
    def temperature(self):
        return self.__temperature
    
    @temperature.setter
    def temperature(self, value):
        if (value <= 0):
            raise ValueError("Temperature must be positive")
        self.__temperature = value