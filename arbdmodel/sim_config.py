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
    """ Class describing properties for a (ARBD or NAMD) simulation """

    def __init__(self, num_steps=None, output_period=None,
                 integrator=None, timestep=None, thermostat=None, barostat=None,
                 temperature=None, pressure=None,
                 cutoff=None, pairlist_distance=None, decomp_period=None, gpu=None,
                 seed=None, restart_file=None,
                 ## ARBD-specific
                 rigid_body_integrator=None,
                 rigid_body_grid_grid_period=None,
                 ## SimpleARBD parameters
                 viscosity=None, solvent_density=None, num_heavy_cluster=None,
                 ## Binary paths
                 arbd_path=None, namd_path=None, vmd_path=None, 
                 hydropro_path=None, apbs_path=None, gmsh_path=None, **kwargs):


        self.num_steps = num_steps
        self.output_period = output_period

        self.integrator = integrator
        self.timestep = timestep
        self.thermostat = thermostat
        self.barostat = barostat

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
        self.num_heavy_cluster = num_heavy_cluster
        
        # Set binary paths
        if arbd_path: BinaryManager.set_binary_path('arbd', arbd_path)
        if namd_path: BinaryManager.set_binary_path('namd', namd_path)
        if vmd_path: BinaryManager.set_binary_path('vmd', vmd_path)
        if hydropro_path: BinaryManager.set_binary_path('hydropro', hydropro_path)
        if apbs_path: BinaryManager.set_binary_path('apbs', apbs_path)
        if gmsh_path: BinaryManager.set_binary_path('gmsh', gmsh_path)

    def get_binary(self, name):
        """
        Get the path to a specific binary with improved error handling.
        
        Args:
            name: The name of the binary (e.g., 'arbd', 'hydropro')
                
        Returns:
            Path to the binary if found, None otherwise
            
        This method does not raise exceptions when binaries are not found, allowing
        for graceful handling of missing optional dependencies.
        """
        binary_path = BinaryManager.get_binary_path(name)
        
        # Check if we found a binary and convert to string if needed
        if binary_path is not None:
            return str(binary_path)
        
        # Determine if this is an essential binary
        if BinaryManager.is_binary_essential(name):
            logger.warning(f"Essential binary '{name}' not found. Core functionality may be limited.")
        
        return None
    
    def set_binary(self, name, path):
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
        Creates a new SimConf object whose properties are
        initialized to be from "self", but are overridden with
        properties in "other", provided they are not None
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
    """ Generic class describing properties for a simulation with default binary paths """

    def __init__(self, num_steps=1e5, output_period=1e3,
                 integrator='MD', timestep=20e-6, thermostat='Langevin', barostat=None,
                 temperature=295, pressure=1,
                 cutoff=50, pairlist_distance=None, decomp_period=40,
                 seed=None, restart_file=None, gpu=0,
                 viscosity=0.01, solvent_density=1.0, num_heavy_cluster=3,
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
                         barostat=barostat,
                         temperature=temperature, 
                         pressure=pressure,
                         cutoff=cutoff, 
                         pairlist_distance=pairlist_distance, 
                         decomp_period=decomp_period,
                         seed=seed, 
                         restart_file=restart_file, 
                         gpu=gpu,
                         viscosity=viscosity,
                         solvent_density=solvent_density,
                         num_heavy_cluster=num_heavy_cluster,
                         **{**default_paths, **kwargs})  # User-provided values override defaults
        
        # Store these for direct access
        self.num_steps = num_steps
        self.output_period = output_period
        self.__temperature = temperature
        self.pressure = pressure
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.num_heavy_cluster = num_heavy_cluster

    @property
    def temperature(self):
        return self.__temperature
    
    @temperature.setter
    def temperature(self, value):
        if (value <= 0):
            raise ValueError("Temperature must be positive")
        self.__temperature = value