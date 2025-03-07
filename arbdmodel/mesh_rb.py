# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from . import logger

from . import RigidBody, RigidBodyType
from .engine import ArbdModel
from .interactions import AbstractPotential
from mesh_process_volume  import MeshProcessor
"""Rigid body shape modeling module for arbdmodel package.

This module provides classes for simple rigid body shape mesh objects into ARBD model.

Input: .msh file
"""


class MeshRBType(RigidBodyType):
    """RigidBodyType subclass for shape rigid body objects"""
    
    def __init__(self, name, msh_file, density=19.3, temperature=303, viscosity=0.01, 
                solvent_density=1.0, unit_scale=1e4,use_surface=False,
                **kwargs):
        """Initialize shape type.
        
        Args:
            name: Name identifier for this type
            msh_file: Path to structure file (.psf/.pdb)
            mass: Mass in AMU
            moment_of_inertia: 3-element list of principal moments of inertia
            diffusivity: Optional diffusion coefficient
            damping_coefficient: Optional damping coefficient
            temperature: Temperature in K (default: 300)
            viscosity: Solvent viscosity in poise (default: 0.01)
            solvent_density: Solvent density in g/cm3 (default: 1.0)
            vmd_path: Path to VMD executable

        """
        self.msh_file = Path(msh_file)
        if use_surface:
            logger.warning(f'Thie moment of inertia using surface mesh is still inaccurate. consider use volume mesh')
            from mesh_process_surface import SurfaceMeshProcessor
            rbprocess=SurfaceMeshProcessor(self.msh_file, density=density, temperature=temperature,
                            viscosity=viscosity, 
                            solvent_density=solvent_density, unit_scale=unit_scale,
                            binary_path=None)
        else:
            rbprocess=MeshProcessor(self.msh_file, density=density, temperature=temperature,
                            viscosity=viscosity, 
                            solvent_density=solvent_density, unit_scale=unit_scale,
                            binary_path=None)
        
        rbprocess.calculate_damping()
        potential_grids=[]
        potential_grids.append(rbprocess.write_no_enter_potential())

        super().__init__(
            name=name,
            mass=rbprocess.mass,
            moment_of_inertia=rbprocess.principal_moments,
            damping_coefficient=rbprocess.transdamp,
            rotational_damping_coefficient=rbprocess.rotational_damping_coefficient,
            potential_grids = potential_grids,
            **kwargs
        )

 
class MeshRBObject(RigidBody):
    """RigidBody subclass representing a shape object"""
    
    def __init__(self, type_, position, orientation, name="protein", **kwargs):
        """Initialize shape object.
        
        Args:
            type_: ShapeType defining properties
            position: Initial position array
            orientation: Initial orientation matrix
            name: Name identifier 
        """
        if not isinstance(type_, MeshRBType):
            raise TypeError("type_ must be a ShapeRBType")
            
        super().__init__(
            type_=type_,
            position=position,
            orientation=orientation,
            name=name,
            **kwargs
        )
        

class MeshRbModel(ArbdModel):
    """Model class for shape-based rigid body simulations"""
    
    def __init__(self, objects, dimensions=(1000, 1000, 1000), static_objects=None, **kwargs):
        """Initialize shape model.
        
        Args:
            objects: List of ShapeRBObject instances
            dimensions: System dimensions tuple
            static_objects: Optional list of static ShapeRBObject instances 
            **kwargs: Additional arguments passed to ArbdModel
        """
        super().__init__(children=[], dimensions=dimensions, **kwargs)
        
        self.diffusible_objects = []
        
        # Add diffusible objects
        for obj in objects:
            if not isinstance(obj, MeshRBObject):
                raise TypeError("Objects must be ShapeRBObject instances")
            self.diffusible_objects.append(obj)
            self.add(obj)
            
