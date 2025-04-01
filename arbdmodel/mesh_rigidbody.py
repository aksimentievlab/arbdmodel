# -*- coding: utf-8 -*-

import os
from pathlib import Path
from .logger import logger
from .core_objects import RigidBodyType


"""Rigid body shape modeling module for arbdmodel package.
This module provides classes for simple rigid body shape mesh objects into ARBD model.
Input: .msh file, or other gmsh supported file
"""

class MeshRigidBodyType(RigidBodyType):
    """RigidBodyType subclass for shape rigid body objects"""
    
    def __init__(self, name, mesh_file, density=19.3, simconf=None, unit_scale=1e4, 
                 use_surface=False,potential_grids=tuple(), expected_mass=None,**kwargs):
        """Initialize shape type.
        
        Args:
            name: Name identifier for this type
            mesh_file: Path to structure file (.psf/.pdb)
            density: Material density in g/cm^3
            simconf: SimConf object containing configuration parameters
            unit_scale: Conversion factor from input units to angstroms
            use_surface: Whether to use surface mesh instead of volume mesh
        """
        self.type_dir = Path.cwd() / name
        try:
            os.makedirs(self.type_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {self.type_dir}: {e}")
            self.type_dir = Path.cwd()

        self.mesh_file = Path(mesh_file)

        if simconf is None:
            from . import DefaultSimConf
            simconf = DefaultSimConf()

        if use_surface:
            logger.warning(f'The moment of inertia using surface mesh is still inaccurate. consider use volume mesh')
            from .mesh_process_surface import SurfaceMeshProcessor
            rbprocess = SurfaceMeshProcessor(
                self.mesh_file, 
                density=density, 
                simconf=simconf,
                unit_scale=unit_scale,
                work_dir=self.type_dir, expected_mass=expected_mass)
        else:
            from .mesh_process_volume import MeshProcessor
            rbprocess = MeshProcessor(
                self.mesh_file,
                density=density, 
                simconf=simconf, 
                unit_scale=unit_scale,
                work_dir=self.type_dir,expected_mass=expected_mass)
        print(rbprocess.volume)
        rbprocess.calculate_damping()
        #potential_dx = str(self.type_dir / f"{name}_potential.dx")
        #potential_grid = rbprocess.write_no_enter_potential(output_file=potential_dx)
        #potential_grids = [(potential_grid, 1.0)]
        attached_particles= rbprocess.get_attached_particles()

        super().__init__(
            name=name,
            mass=rbprocess.mass,
            moment_of_inertia=rbprocess.principal_moments,
            damping_coefficient=rbprocess.transdamp,
            rotational_damping_coefficient=rbprocess.rotdamp,
            #potential_grids=potential_grids, 
            attached_particles=attached_particles,
            **kwargs)
