"""
Example script demonstrating how to use the ShapeCGModel for coarse-grained protein simulations.

This script shows how to set up and run simulations with multiple proteins at different
coarse-graining resolutions.

Copy this and run outside the package directory to avoid import errors.
"""

import numpy as np
from pathlib import Path

from arbdmodel.shape_cg import ShapeCGModel

# List of PDB IDs for proteins to simulate
pdb_list = ['1ubq', '1brs', '1pgb']  # Example PDBs (ubiquitin, barnase-barstar complex, protein G)

# Define how many copies of each protein to include
def get_copy_number(pdb_id, total_copies):
    """Simple function to distribute copies among proteins."""
    weights = {'1ubq': 3, '1brs': 2, '1pgb': 1}  # Weight distribution for each protein
    total_weight = sum(weights.values())
    return int(np.round(weights.get(pdb_id, 1) * total_copies / total_weight))

# Create the model
model = ShapeCGModel.from_protein_list(
    pdb_list,
    box_size=500,  # Set a large initial box
    cutoff=200     # Large cutoff for long-range interactions
)

# Example 1: Simple system with default parameters
def example_simple():
    """Run a simple simulation with default parameters."""
    # Define protein counts
    protein_counts = {pdb_id: get_copy_number(pdb_id, 10) for pdb_id in pdb_list}
    
    # Run simulation with dry_run=True (just to demonstrate the setup)
    model.setup_and_run(
        protein_counts,
        radius=100,        # Confinement radius
        num_CG_sites=8,    # 8 CG sites per protein
        salt_concentration=150,
        dry_run=True       # Don't actually run the simulation
    )
    
    print("Example 1: Simple setup completed")

# Example 2: Two-step minimization and production run
def example_minimization_production():
    """Run a two-step minimization followed by production simulation."""
    # Define protein counts
    protein_counts = {pdb_id: get_copy_number(pdb_id, 50) for pdb_id in pdb_list}
    
    # Run the minimization first
    min_file = model.run_minimization(
        protein_counts,
        radius=150,
        gpu=0,
        num_steps=1e4,     # Short minimization for demonstration
        dry_run=True       # Set to False for actual run
    )
    
    # Then run production with a higher resolution model
    model.run_from_minimized(
        protein_counts,
        radius=150,
        num_CG_sites=16,   # More detailed CG model
        salt_concentration=150,
        minimization_file=min_file,
        num_steps=1e6,     # Longer production run
        dry_run=True       # Set to False for actual run
    )
    
    print("Example 2: Minimization and production setup completed")

# Example 3: Testing different CG resolutions
def example_resolution_comparison():
    """Compare different CG resolutions."""
    # Define protein counts
    protein_counts = {pdb_id: get_copy_number(pdb_id, 20) for pdb_id in pdb_list}
    
    # Generate random positions once for consistent comparison
    positions = model.generate_random_protein_positions(protein_counts, radius=120)
    
    # Run simulations with different resolutions
    for num_cg in [2, 8, 32]:
        model_copy = ShapeCGModel.from_protein_list(
            pdb_list,
            box_size=500,
            cutoff=200
        )
        
        model_copy.setup_and_run(
            protein_counts,
            radius=120,
            num_CG_sites=num_cg,
            positions=positions,  # Use same positions for fair comparison
            salt_concentration=150,
            dry_run=True,         # Set to False for actual run
            num_steps=1e5         # Short run for testing
        )
        
    print("Example 3: Resolution comparison setup completed")

# Example 4: Varying concentration
def example_concentration_study():
    """Study the effect of different salt concentrations."""
    # Define protein counts
    protein_counts = {pdb_id: get_copy_number(pdb_id, 30) for pdb_id in pdb_list}
    
    # Generate minimized structure first
    min_file = model.run_minimization(
        protein_counts,
        radius=130,
        gpu=0,
        num_steps=1e4,
        dry_run=True       # Set to False for actual run
    )
    
    # Run with different salt concentrations
    for salt_conc in [50, 150, 300]:
        model_copy = ShapeCGModel.from_protein_list(
            pdb_list,
            box_size=500,
            cutoff=200
        )
        
        model_copy.run_from_minimized(
            protein_counts,
            radius=130,
            num_CG_sites=16,
            salt_concentration=salt_conc,
            minimization_file=min_file,
            num_steps=1e6,
            dry_run=True    # Set to False for actual run
        )
        
    print("Example 4: Concentration study setup completed")

if __name__ == "__main__":
    # Run all examples
    example_simple()
    example_minimization_production()
    example_resolution_comparison()
    example_concentration_study()
    
    print("\nAll examples completed.")
    print("To run actual simulations, set dry_run=False in the example functions.")