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
from .runner import HydroProRunner, APBSRunner

from .shape_rb_cal import Get_damping_coefficients, Fix_charge, Bound_grid

"""Rigid body shape modeling module for arbdmodel package.

This module provides classes for rigid body shape modeling in the arbdmodel package,
adapting the original SimpleARBD functionality to use RigidBody and RigidBodyType.
SimpleARBD uses a different clustering scheme as Chris original script
that helps resolved the sickness issue between proteins
maybe we should have an option for users, depending if they want proteins to stick or not.
Generating BD boundary using input formats from MD
"""

class ShapeRBType(RigidBodyType):
    """RigidBodyType subclass for shape rigid body objects"""
    
    def __init__(self, name, structure_path, mass=None, moment_of_inertia=None,
                 diffusivity=None, damping_coefficient=None, 
                 temperature=300, viscosity=0.01, solvent_density=1.0,
                 vmd_path=None, hydro_path=None, apbs_path=None,
                 **kwargs):
        """Initialize shape type.
        
        Args:
            name: Name identifier for this type
            structure_path: Path to structure file (.psf/.pdb)
            mass: Mass in AMU
            moment_of_inertia: 3-element list of principal moments of inertia
            diffusivity: Optional diffusion coefficient
            damping_coefficient: Optional damping coefficient
            temperature: Temperature in K (default: 300)
            viscosity: Solvent viscosity in poise (default: 0.01)
            solvent_density: Solvent density in g/cm3 (default: 1.0)
            vmd_path: Path to VMD executable
            hydro_path: Path to HydroPro executable
            apbs_path: Path to APBS executable
        """
        super().__init__(
            name=name,
            mass=mass,
            moment_of_inertia=moment_of_inertia,
            diffusivity=diffusivity, 
            damping_coefficient=damping_coefficient,
            **kwargs
        )
        self.structure_path = Path(structure_path)
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        
        # Store paths to required executables
        self.vmd_path = vmd_path or 'vmd'
        self.hydro_path = hydro_path or 'hydropro'
        self.apbs_path = apbs_path or 'apbs'
        
        # Initialize runners
        self.hydro_runner = HydroProRunner(
            self.hydro_path,
            temperature=temperature,
            viscosity=viscosity,
            solvent_density=solvent_density
        )
        self.apbs_runner = APBSRunner(self.apbs_path)
        
    def prepare_structure(self, align=True, skip_parametrizing=False):
        """Prepare structure files by aligning and parametrizing.
        
        Args:
            align: Whether to align the structure 
            skip_parametrizing: Skip parametrization steps
        """
        if skip_parametrizing:
            logger.info(f"Skipping parametrization for {self.name}")
            return
            
        base_name = self.structure_path.stem
        work_dir = Path.cwd()
        
        if align:
            # Write alignment TCL script
            align_tcl = work_dir / "align.tcl"
            text = '''lassign $argv prefix
            set seltext all

            proc rotationIsRightHanded {R {tol 0.01}} {
                set x [coordtrans $R {1 0 0}]
                set y [coordtrans $R {0 1 0}]
                set z [coordtrans $R {0 0 1}]

                set l [veclength [vecsub $z [veccross $x $y]]]
                return [expr {$l < $tol}]
            }

            set ID [mol new $prefix.psf]
            mol addfile $prefix.pdb waitfor all
            set all [atomselect $ID all]
            set sel [atomselect $ID $seltext]

            ## Center system on $sel
            $all moveby [vecinvert [measure center $sel weight mass]]

            set continue 1
            while { $continue } {
                ## Get current moment of inertia to determine rotation to align
                lassign [measure inertia $sel moments] com principleAxes

                ## Convert 3x3 rotation to 4x4 vmd transformation
                set R [trans_from_rotate $principleAxes]
                ## Fix left-handed principle axes sometimes returned by 'measure inertia'
                if { ! [rotationIsRightHanded $R] } {
                    puts "This was true"
                    # puts "rotation $R is not right handed! Fixing!"
                    set R [transmult {{1 0 0 0} {0 1 0 0} {0 0 -1 0} {0 0 0 1}} $R]
                }

                puts "My rotation is here: $R"
                puts "My second line is here: [lassign [measure inertia $sel moments] com principleAxes]"

                ## Apply rotation and check that it worked
                $all move $R

                ## Get current moment of inertia to determine rotation to align

                lassign [measure inertia $sel moments] com principleAxes moments
                puts $principleAxes
                set goodcount 0
                foreach x0 {{1 0 0} {0 1 0} {0 0 1}} {
                    set x [coordtrans [trans_from_rotate $principleAxes] $x0]
                    if {[veclength [vecsub $x $x0]] < 0.01} {
                        incr goodcount
                    }
                }
                if { $goodcount == 3 } {
                    set continue 0
                }
            }

            ## Write transformation matrix to return to original conformation
            set ch [open $prefix.rotate-back.txt w]
            foreach line [trans_to_rotate [transtranspose $R]] {
                puts $ch $line
            }
            close $ch

            ## Write out moments of inertia
            set ms ""
            foreach m $moments { lappend ms [veclength $m] }
            set ch [open $prefix.inertia.txt w]
            puts $ch $ms
            close $ch

            ## Write out mass
            set ch [open $prefix.mass.txt w]
            puts $ch [measure sumweights $sel weight mass]
            close $ch

            ## Write out psf, pdb of transformed selection
            $sel writepdb $prefix.aligned.pdb
            $sel writepsf $prefix.aligned.psf'''
            
            with open(align_tcl, 'w') as fout:
                fout.write(text)

            # Run alignment
            cmd = f"{self.vmd_path} -dispdev text -args {base_name} < {align_tcl}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Aligned files will be: {base_name}.aligned.{pdb,psf}
            self.aligned_pdb = work_dir / f"{base_name}.aligned.pdb"
            self.aligned_psf = work_dir / f"{base_name}.aligned.psf"
            
            if not (self.aligned_pdb.exists() and self.aligned_psf.exists()):
                raise RuntimeError(f"Alignment failed for {base_name}")
                
        # Get mass from aligned structure
        mass_file = work_dir / f"{base_name}.mass.txt"
        if not mass_file.exists():
            raise FileNotFoundError(f"Mass file not found: {mass_file}")
            
        with open(mass_file) as f:
            self.mass = float(f.readline().strip())
            
        # Run HydroPro to get hydrodynamic properties
        results = self.hydro_runner.run_calculation(
            base_name,
            self.mass,
            work_dir=str(work_dir)
        )
        
        self.damping_coefficient = results['translation_damping']
        self.rotational_damping_coefficient = results['rotation_damping']
        
        # Generate charge distribution
        self._generate_charge_distribution(work_dir)
        
    def _generate_charge_distribution(self, work_dir: Path, resolution: float = 2.0):
        """Generate charge distribution using VMD.
        
        Args:
            work_dir: Working directory
            resolution: Grid resolution in Angstroms
        """
        base_name = self.structure_path.stem
        charge_tcl = work_dir / "charge.tcl"

        text = '''lassign $argv prefix
        set resolution ''' + str(resolution) + '''
        set ID [mol new $prefix.psf]
        mol addfile $prefix.pdb
        set all [atomselect $ID all]
        set netCharge [measure sumweights $all weight charge]

        ## Write out charge density
        volmap density $all -o $prefix.chargeDensity.dx -res $resolution -weight charge
        ## Write out pqr for subsequent EM calculation
        $all writepqr $prefix.pqr

        set ch [open $prefix.netCharge.dat w]
        puts $ch $netCharge
        close $ch

        set ch [open $prefix.dimension.dat w]
        set minmax [measure minmax $all]
        set x_dim [expr [lindex [lindex $minmax 1] 0] - [lindex [lindex $minmax 0] 0]]
        set y_dim [expr [lindex [lindex $minmax 1] 1] - [lindex [lindex $minmax 0] 1]]
        set z_dim [expr [lindex [lindex $minmax 1] 2] - [lindex [lindex $minmax 0] 2]]
        puts $ch $x_dim
        puts $ch $y_dim
        puts $ch $z_dim
        close $ch'''
        with open(charge_tcl, 'w') as f:
            f.write(text)

        # Run VMD to generate charge density
        cmd = f"{self.vmd_path} -dispdev text -args {base_name}.aligned < {charge_tcl}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Fix charge distribution
        charge_dx = work_dir / f"{base_name}.aligned.chargeDensity.dx"
        charge_out = work_dir / f"{base_name}.aligned.charge.dx"
        netcharge = work_dir / f"{base_name}.aligned.netCharge.dat"
        
        if not charge_dx.exists():
            raise FileNotFoundError(f"Charge density file not found: {charge_dx}")
            
        Fix_charge(str(charge_dx), str(charge_out), str(netcharge))
        
        self.charge_dx = charge_out
        self.charge_density_dx = charge_dx

class ShapeRBObject(RigidBody):
    """RigidBody subclass representing a shape object"""
    
    def __init__(self, type_, position, orientation, name="protein", **kwargs):
        """Initialize shape object.
        
        Args:
            type_: ShapeType defining properties
            position: Initial position array
            orientation: Initial orientation matrix
            name: Name identifier 
        """
        if not isinstance(type_, ShapeRBType):
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

class ShapeRbModel(ArbdModel):
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
        self.static_objects = []
        
        # Add diffusible objects
        for obj in objects:
            if not isinstance(obj, ShapeRBObject):
                raise TypeError("Objects must be ShapeRBObject instances")
            self.diffusible_objects.append(obj)
            self.add(obj)
            
        # Add static objects if provided
        if static_objects:
            for obj in static_objects:
                if not isinstance(obj, ShapeRBObject):
                    raise TypeError("Static objects must be ShapeRBObject instances")
                self.static_objects.append(obj)  
                self.add(obj)


class ShapeRbModel2(ArbdModel):
    """Model class for shape-based rigid body simulations"""
    
    def __init__(self, config=None, dimensions=(1000, 1000, 1000), **kwargs):
        """Initialize shape model.
        
        Args:
            config: Configuration dictionary 
            dimensions: System dimensions tuple
            **kwargs: Additional arguments passed to ArbdModel
        """
        super().__init__(children=[], dimensions=dimensions, **kwargs)
        
        self.config = config
        self.diffusible_objects = []
        self.static_objects = []
        
        # Parse config and create objects
        self._setup_from_config()
        
    def _setup_from_config(self):
        """Setup model based on configuration."""
        # Create diffusible objects
        for obj_path in self.config['diffusible_objects']:
            obj_type = ShapeRBType(
                name=Path(obj_path).stem,
                structure_path=obj_path
            )
            obj_type.prepare_structure(
                skip_parametrizing=self.config['Skip_parametrizing_diffusible'] == "Yes"
            )
            
            # Create object instance
            obj = ShapeRBObject(
                type_=obj_type,
                position=np.zeros(3),  # Initial position set later
                orientation=np.eye(3)   # Initial orientation set later
            )
            self.diffusible_objects.append(obj)
            self.add(obj)
            
        # Create static objects if any
        if len(self.config['static_objects']) > 0:
            for obj_path in self.config['static_objects']:
                obj_type = ShapeRBType(
                    name=Path(obj_path).stem,
                    structure_path=obj_path
                )
                obj_type.prepare_structure()
                
                obj = ShapeRBObject(
                    type_=obj_type,
                    position=np.zeros(3),
                    orientation=np.eye(3)
                )
                self.static_objects.append(obj)
                self.add(obj)
                
    def prepare_for_simulation(self):
        """Prepare model for simulation by setting up necessary files and parameters."""
        # Apply hydrodynamic properties
        self._setup_hydrodynamics()
        
        # Get charge distributions
        self._setup_charge_distributions()
        
        # Setup electrostatic maps
        self._setup_electrostatics()
        
        # Setup VDW interactions
        self._setup_vdw()
        
        # Apply Gaussian smoothing
        self._apply_smoothing()
        
        # Create simulation boundary
        self._setup_boundary()
        
        # Generate initial coordinates
        self._generate_initial_coords()
        
    def _setup_hydrodynamics(self):
        """Setup hydrodynamic properties for diffusible objects."""
        # TODO: Implement hydrodynamics setup using HydroPro
        pass
        
    def _setup_charge_distributions(self):
        """Calculate charge distributions."""
        # TODO: Implement charge distribution calculation
        pass
        
    def _setup_electrostatics(self):
        """Setup electrostatic potential maps."""
        # TODO: Implement APBS-based electrostatics calculation
        pass
        
    def _setup_vdw(self):
        """Setup van der Waals interaction maps."""
        # TODO: Implement VDW potential calculation
        pass
        
    def _apply_smoothing(self):
        """Apply Gaussian smoothing to potential maps."""
        # TODO: Implement Gaussian smoothing
        pass
        
    def _setup_boundary(self):
        """Setup simulation boundary conditions."""
        # TODO: Implement boundary setup
        pass
        
    def _generate_initial_coords(self):
        """Generate initial coordinates for all objects."""
        # TODO: Implement initial coordinate generation
        pass


def read_shape_config(config_path):
    """Read shape configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config = {}
    with open(config_path) as f:
        text = f.read()
        
    # Required parameters
    import re
    patterns = {
        'diffusible_objects': r'Diffusible_objects:([ \w]+)',
        'static_objects': r'Static_objects \(Enter NA for no static object\):([ \w]+)',
        'temperature': r'Temperature \(K\):(\s*[0-9]*.*[0-9]*)',
        'viscosity': r'Viscosity:(\s*[0-9]*.*[0-9]*)', 
        'solvent_density': r'Solvent_density:(\s*[0-9]*.*[0-9]*)',
        'Skip_parametrizing_diffusible': r'Skip_parametrizing_diffusible \(Yes/No\):([ \w]+)',
        'vmd_path': r'Vmd_path:(\s*\S+)',
        'hydro_path': r'Hydro_path:(\s*\S+)',
        'apbs_path': r'Apbs_path:(\s*\S+)'
    }
    
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Required parameter {key} not found in config")
        value = m.group(1).strip()
        
        if key == 'static_objects' and value == 'NA':
            config[key] = []
        elif key == 'diffusible_objects':
            config[key] = value.split()
        elif key == 'static_objects':
            config[key] = value.split()
        else:
            config[key] = value
            
    return config