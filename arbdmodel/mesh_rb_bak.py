# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from . import logger

from .arbd_objects import RigidBody, RigidBodyType
from . import ArbdModel
from .interactions import AbstractPotential

"""Rigid body shape modeling module for arbdmodel package.

This module provides classes for simple metal rigid body shape mesh objects into ARBD model.

Input: .msh file
"""


class MeshRBType(RigidBodyType):
    """RigidBodyType subclass for shape rigid body objects"""
    
    def __init__(self, name, msh_file, mass=None, moment_of_inertia=None,
                 diffusivity=None, damping_coefficient=None, vmd_path=None, **kwargs):
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
        super().__init__(
            name=name,
            mass=mass,
            moment_of_inertia=moment_of_inertia,
            diffusivity=diffusivity, 
            damping_coefficient=damping_coefficient,
            **kwargs
        )
        self.msh_file = Path(msh_file)
        
        # Store paths to required executables
        self.vmd_path = vmd_path or 'vmd'
        
        
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
        
        # Add electrostatics potential to potential grids
        if hasattr(type_, 'charge_dx'):
            self.add_potential_grid('elec', type_.charge_dx, scale=0.59616195)
            self.add_charge_grid('elec', type_.charge_density_dx)

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
            
