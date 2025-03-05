"""Integration with external tools for shape rigid body calculations.
Original script by Chun Kit Chan, 2024

This module provides interfaces to external tools used in shape rigid body modeling:
- HydroPro for hydrodynamic calculations
- APBS for electrostatics calculations
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from . import logger, get_resource_path

class HydroProRunner:
    """Interface to HydroPro for hydrodynamic calculations"""
    
    def __init__(self, mass,binary_path=None, temperature=295, viscosity=0.01, solvent_density=1.0,structure_name="hydrocal"):
        """Initialize HydroPro interface.
        
        Args:
            binary_path: Path to HydroPro executable. If None, uses bundled binary
            temperature: Temperature in Kelvin (default: 295K)
            viscosity: Solvent viscosity in poise (default: 0.01)
            solvent_density: Solvent density in g/cm3 (default: 1.0)
            mass: mass in amu
        """
        if binary_path is None:
            # Determine correct binary based on platform
            if platform.system() == 'Windows':
                binary_name = 'hydropro10-msd.exe'
            else:  # Unix-like systems
                binary_name = 'hydropro10-lnx.exe'
            
            self.binary = get_resource_path('hydropro10') / binary_name
        else:
            self.binary = Path(binary_path)
            
        if not self.binary.exists():
            raise FileNotFoundError(f"HydroPro binary not found at {self.binary}")
            
        # Make binary executable if needed (Unix only)
        if platform.system() != 'Windows' and not os.access(self.binary, os.X_OK):
            os.chmod(self.binary, 0o755)
            
        self.temperature = temperature
        self.viscosity = viscosity
        self.solvent_density = solvent_density
        self.structure_name=structure_name
        self.mass=mass
        
    def write_config(self,output_path="hydropro.dat"):
        mass=self.mass
        structure_name=self.structure_name
        """Write HydroPro configuration file.
        
        Args:
            structure_name: Base name of structure files
            mass: Mass in AMU
            output_path: Path to write config file
        """
        temperature_c = self.temperature - 273.15  # Convert K to C
        
        config = f"""hydro
        {structure_name}.hydro
        hydro.pdb
        1
        2.9,
        6,
        1.2,
        3.0,
        {temperature_c},
        {self.viscosity},
        {mass},
        1.0,
        {self.solvent_density}
        -1,
        -1,
        0,
        1
        *"""
        
        with open(output_path, 'w') as f:
            f.write(config)

    def parse_output(self, output_file):
        mass=self.mass
        """Parse HydroPro output file to get damping coefficients.
        
        Args:
            output_file: Path to HydroPro output file
            mass: Mass in AMU used to normalize coefficients
            
        Returns:
            tuple of (translation_damping, rotation_damping)
        """
        with open(output_file) as f:
            lines = f.readlines()
        mass=self.mass
            
        # Skip header
        line_num = 48
        
        # Read translational coefficients
        Dx = float(lines[line_num].strip().split()[0])
        Dy = float(lines[line_num+1].strip().split()[1])
        Dz = float(lines[line_num+2].strip().split()[2])
        
        # Skip two lines
        line_num += 4
        
        # Read rotational coefficients
        Rx = float(lines[line_num].strip().split()[3])
        Ry = float(lines[line_num+1].strip().split()[4])
        Rz = float(lines[line_num+2].strip().split()[5])
        
        # Convert units
        # Translation: "(295 k K) / (( cm^2/s) *  amu)" "1/ns"
        trans_damp = [24.527692/(x*mass) for x in [Dx, Dy, Dz]]
        
        # Rotation: "(295 k K) / ((1 /s) *  amu AA^2)" "1/ns"
        rot_damp = [2.4527692e+17 / (x*mass) for x in [Rx, Ry, Rz]]
        
        return trans_damp, rot_damp
                
    def run_calculation(self,work_dir="."):
        """Run HydroPro calculation.
        
        Args:
            structure_name: Base name of structure files
            mass: Mass in AMU
            work_dir: Working directory for calculation
        
        Returns:
            Dictionary containing:
            - translation_damping: [Dx, Dy, Dz]
            - rotation_damping: [Rx, Ry, Rz]
        """
        structure_name, mass=self.structure_name, self.mass
        original_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Write config
            self.write_config()
            
            # Link structure file
            pdb_path = Path(f"{structure_name}.pdb")
            if not pdb_path.exists():
                raise FileNotFoundError(f"Structure file not found: {pdb_path}")
            os.symlink(pdb_path, "hydro.pdb")
            
            # Run HydroPro
            result = subprocess.run([str(self.binary)], 
                                 capture_output=True, 
                                 text=True,
                                 check=True)
            
            # Parse results
            trans_damp, rot_damp = self._parse_output(f"{structure_name}.hydro-res.txt", mass)
            
            return {
                "translation_damping": trans_damp,
                "rotation_damping": rot_damp
            }
            
        finally:
            if os.path.exists("hydro.pdb"):
                os.unlink("hydro.pdb")
            os.chdir(original_dir)

class APBSRunner:
    """Interface to APBS for electrostatics calculations"""
    
    def __init__(self, binary_path, psize_script=None):
        """Initialize APBS interface.
        
        Args:
            binary_path: Path to APBS executable
            psize_script: Optional path to psize.py script
        """
        self.binary = Path(binary_path)
        if not self.binary.exists():
            raise FileNotFoundError(f"APBS binary not found at {binary_path}")
            
        self.psize = Path(psize_script) if psize_script else None
        
    def write_config(self, structure_name, xyz_dims, salt_conc=0.15, 
                    temperature=300, buffer=50, large_system='Off'):
        """Write APBS configuration file.
        
        Args:
            structure_name: Base name of structure files
            xyz_dims: [x, y, z] dimensions
            salt_conc: Salt concentration in M
            temperature: Temperature in K
            buffer: Grid buffer size in Å
            large_system: 'On' or 'Off' for large system mode
        """
        # Calculate grid dimensions
        xyz_cg = [str(int(dim + buffer)) for dim in xyz_dims]
        
        if large_system == 'Off':
            xyz_dime = xyz_cg
            center = 'mol 1'
        else:
            # For large systems, reduce grid density
            dividend = 2
            xyz_dime = [str(int((dim + buffer) / dividend)) for dim in xyz_dims]
            center = 'mol 1'
            
        config = f"""read
        mol pqr {structure_name}.pqr
        end
        elec
        mg-auto
        dime {' '.join(xyz_dime)}
        cglen {' '.join(xyz_cg)}
        cgcent {center}
        fglen {' '.join(xyz_cg)}
        fgcent {center}
        mol 1
        npbe
        bcfl sdh
        srfm smol
        chgm spl2
        ion 1 {salt_conc} 2.0
        ion -1 {salt_conc} 2.0
        pdie 12.0
        sdie 78.54
        sdens 10.0
        srad 1.4
        swin 0.3
        temp {temperature}
        gamma 0.105
        calcenergy no
        calcforce no
        write pot dx {structure_name}.elec.tmp
        end
        quit"""

        with open(f"{structure_name}.apbs", 'w') as f:
            f.write(config)
            
    def run_calculation(self, structure_name, xyz_dims, work_dir=".", 
                       salt_conc=0.15, temperature=300):
        """Run APBS calculation.
        
        Args:
            structure_name: Base name of structure files
            xyz_dims: [x, y, z] dimensions
            work_dir: Working directory
            salt_conc: Salt concentration in M
            temperature: Temperature in K
            
        Returns:
            Path to output potential file
        """
        original_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Write config
            self.write_config(structure_name, xyz_dims, salt_conc, temperature)
            
            # Run APBS
            result = subprocess.run([str(self.binary), f"{structure_name}.apbs"],
                                 capture_output=True,
                                 text=True, 
                                 check=True)
            
            output_file = Path(f"{structure_name}.elec.tmp")
            if not output_file.exists():
                raise RuntimeError("APBS failed to generate output file")
                
            return output_file
            
        finally:
            os.chdir(original_dir)