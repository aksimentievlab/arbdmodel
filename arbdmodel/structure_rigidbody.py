import os
import os
import sys
import numpy as np
from pathlib import Path

from . import logger
from . import RigidBody, RigidBodyType
from .engine import ArbdModel,SimConf
from .structure_from_pdb import StructureProcessor

"""Structure rigid body modeling module for ARBD.

This module provides classes for structure-based rigid body modeling in the ARBD package,
using a clean implementation that processes molecular structures into grid maps.
"""

class StructureRigidBodyType(RigidBodyType):  #diffusible objects

    """RigidBodyType subclass for structure-based rigid body objects"""
    
    def __init__(self, name, structure_path, 
                 temperature=300, viscosity=0.01, solvent_density=1.0,
                 vmd_path=None, hydro_path=None, apbs_path=None,
                 parameters_folder="./parameters", num_heavy_cluster=3, 
                 work_dir=None, **kwargs):
        """Initialize structure rigid body type from structure files.
        
        Args:
            name: Name identifier for this type
            structure_path: Path to structure file (.psf/.pdb)
            temperature: Temperature in K (default: 300)
            viscosity: Solvent viscosity in poise (default: 0.01)
            solvent_density: Solvent density in g/cm3 (default: 1.0)
            vmd_path: Path to VMD executable (default: 'vmd')
            hydro_path: Path to HydroPro executable
            apbs_path: Path to APBS executable (default: 'apbs')
            parameters_folder: Path to parameters folder
            num_heavy_cluster: Number of heavy atom clusters for VDW maps
            work_dir: Directory to store processed files (default: current directory)
        """
        # Create work directory if specified
        if work_dir:
            work_dir = Path(work_dir)
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = Path.cwd() / name
            os.makedirs(work_dir, exist_ok=True)
            
        # Process the structure to get properties and grid maps
        processor = StructureProcessor(
            structure_path=structure_path,
            temperature=temperature,
            viscosity=viscosity,
            solvent_density=solvent_density,
            num_heavy_cluster=num_heavy_cluster,
            vmd_path=vmd_path,
            hydro_path=hydro_path,
            apbs_path=apbs_path,
            parameters_folder=parameters_folder,
            work_dir=work_dir)
        
        # Process the structure to get all properties and grid files
        processor.process_structure()
        
        # Initialize the parent class with collected data
        super().__init__(
            name=name, 
            mass=processor.mass,
            moment_of_inertia=processor.moment_of_inertia,
            damping_coefficient=processor.transdamp,
            rotational_damping_coefficient=processor.rotdamp,
            potential_grids=processor.potential_grids,
            charge_grids=processor.charge_grids,
            pmf_grids=[],
            **kwargs
        )
        
        # Store file paths for reference
        self.aligned_pdb = processor.aligned_pdb
        self.aligned_psf = processor.aligned_psf
        
        logger.info(f"StructureRigidBodyType '{name}' initialized successfully")


class StructureRigidBody(RigidBody):
    """RigidBody subclass representing a structure-based rigid body"""
    
    def __init__(self, type_, position, orientation, name="molecule", **kwargs):
        """Initialize structure-based rigid body.
        
        Args:
            type_: StructureRigidBodyType defining properties
            position: Initial position array
            orientation: Initial orientation matrix
            name: Name identifier 
        """
        if not isinstance(type_, StructureRigidBodyType):
            raise TypeError("type_ must be a StructureRigidBodyType")
            
        super().__init__(
            type_=type_,
            position=position,
            orientation=orientation,
            name=name,
            **kwargs
        )


class StructureRigidBodyModel(ArbdModel):
    """Model class for structure-based rigid body simulations"""
    
    def __init__(self,temperature,viscosity,solvent_density,
                 diffusible_objects=None, static_objects=None, 
                 dimensions=(1000, 1000, 1000), **kwargs):
        """Initialize structure model.
        
        Args:
            diffusible_objects: List of StructureRigidBody instances for diffusible objects
            static_objects: List of StructureRigidBody instances for static objects
            dimensions: System dimensions tuple
            **kwargs: Additional arguments passed to ArbdModel
        """
        super().__init__(children=[], dimensions=dimensions, **kwargs)
        
        self.diffusible_objects = []
        self.static_objects = []
        
        # Add diffusible objects
        if diffusible_objects:
            for obj in diffusible_objects:
                self.diffusible_objects.append(obj)
                self.add(obj)
                
        # Add static objects
        if static_objects:
            for obj in static_objects:
                processor = StructureProcessor(structure_path=structure_path,
                temperature=temperature,
                viscosity=viscosity,
                solvent_density=solvent_density,
                num_heavy_cluster=num_heavy_cluster,
                vmd_path=vmd_path,
                apbs_path=apbs_path,
                parameters_folder=parameters_folder,
                work_dir=work_dir)
                
                self.static_objects.append(obj)
                self.add(obj)

