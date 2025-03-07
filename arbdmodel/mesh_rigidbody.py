# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from . import logger

from . import RigidBody, RigidBodyType


"""Rigid body shape modeling module for arbdmodel package.

This module provides classes for simple rigid body shape mesh objects into ARBD model.

Input: .msh file
"""


class MeshRigidBodyType(RigidBodyType):
    """RigidBodyType subclass for shape rigid body objects"""
    
    def __init__(self, name, mesh_file, density=19.3, temperature=300, viscosity=0.01, 
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
        """
        self.type_dir = Path.cwd() / name
        try:
            os.makedirs(self.type_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {self.type_dir}: {e}")
            self.type_dir = Path.cwd()

        self.mesh_file = Path(mesh_file)
        if use_surface:
            logger.warning(f'The moment of inertia using surface mesh is still inaccurate. consider use volume mesh')
            from .mesh_process_surface import SurfaceMeshProcessor
            rbprocess = SurfaceMeshProcessor( self.mesh_file, density=density, 
                temperature=temperature, viscosity=viscosity, 
                solvent_density=solvent_density, 
                unit_scale=unit_scale,binary_path=None,
                work_dir=self.type_dir)
        else:
            from .mesh_process_volume  import MeshProcessor
            rbprocess = MeshProcessor( self.mesh_file,density=density, 
                temperature=temperature, viscosity=viscosity, 
                solvent_density=solvent_density, unit_scale=unit_scale,
                binary_path=None, work_dir=self.type_dir)
        rbprocess.calculate_damping()
        potential_dx = str(self.type_dir / f"{name}_potential.dx")
        potential_grid = rbprocess.write_no_enter_potential(output_file=potential_dx)
        potential_grids = [(potential_grid, 1.0)]
        attached_particles=[] #rbprocess.nodes 

        super().__init__(
            name=name,
            mass=rbprocess.mass,
            moment_of_inertia=rbprocess.principal_moments,
            damping_coefficient=rbprocess.transdamp,
            rotational_damping_coefficient=rbprocess.rotdamp,
            potential_grids = potential_grids, 
            attached_particles=attached_particles,
            **kwargs)
