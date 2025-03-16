import parmed
from arbdmodel import ParmedArbd
import MDAnalysis as mda

#Example 1: Simple system with default parameters

# Create from PSF/PDB files
model = ParmedArbd(
    psf='molecule.psf', 
    pdb='molecule.pdb',
    parameter_files=['parm10.prm', 'frcmod.DNA.OL15.prm'],
    cutoff=12
)

# Run a simulation
model.simulate(
    'simulation_run',
    num_steps=10000,
    output_period=100
)


#Example 2: Dual topology model for base pair mutation
# Load structures
p1 = parmed.load_file('unmutated.psf')
p2 = parmed.load_file('mutated.psf')
c1 = parmed.load_file('unmutated.pdb')
c2 = parmed.load_file('mutated.pdb')

# Transfer coordinates
for a, b in zip(p1.atoms, c1.atoms):
    a.xx, a.xy, a.xz = b.xx, b.xy, b.xz
for a, b in zip(p2.atoms, c2.atoms):
    a.xx, a.xy, a.xz = b.xx, b.xy, b.xz
    
# Load parameter files if needed
params = parmed.charmm.CharmmParameterSet('parm10.prm', 'frcmod.DNA.OL15.prm')
p1.load_parameters(params)
p2.load_parameters(params)

# Create MDAnalysis universes for base pair detection (optional)
u1 = mda.Universe('unmutated.psf', 'unmutated.pdb')
u2 = mda.Universe('mutated.psf', 'mutated.pdb')

# Create dual topology model
dual_model = ParmedArbd.create_dual_topology_model(
    p1, p2, u1, u2,
    cutoff=12,
    integrator='MD'
)

# Write restraint files for FEP simulation
dual_model.write_restraint_files(
    fep_file='dual.fep.exb',
    min_fep_file='dual.min.fep.exb',
    bp_file='dual.bp.exb'
)

# Run simulation
dual_model.simulate(
    'fep_simulation',
    num_steps=10000,
    output_period=100
)

