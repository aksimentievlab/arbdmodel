#!/usr/bin/env python3
"""
Mesh-Based Diffusion Calculator

This script processes mesh files to calculate diffusion coefficients
using both the traditional ellipsoid approximation and a more accurate
bead shell model approach.

Usage:
    python mesh_diffusion.py <mesh_file> [options]

Arguments:
    mesh_file               Path to input mesh file (.msh format)

Options:
    --density DENSITY       Material density in g/cm³ [default: 1.0]
    --temp TEMP             Temperature in Kelvin [default: 295]
    --visc VISC             Viscosity in poise [default: 0.01]
    --bead-radius RADIUS    Radius of beads in Angstroms [default: 5.0]
    --bead-coverage COV     Coverage factor for bead distribution [default: 1.0]
    --bead-method METHOD    Method for bead placement (vertices|centroids|uniform) [default: uniform]
    --compare-methods       Run comparison between different bead methods
    --optimize              Find optimal bead parameters
    --output-dx FILE        Output potential field to DX file [default: mesh_potential.dx]
    --no-potential          Skip potential field generation
    --ellipsoid-only        Use only ellipsoid approximation (no bead model)
    --help                  Show this help message and exit
"""

import sys
import argparse
import time
from pathlib import Path

# Import the modules
from bead_model_diffusion import BeadModelDiffusion
from mesh_processor_extended import process_mesh_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate diffusion coefficients for mesh structures")
    
    parser.add_argument("mesh_file", type=str, help="Path to input mesh file (.msh format)")
    
    parser.add_argument("--density", type=float, default=1.0,
                        help="Material density in g/cm³ [default: 1.0]")
    
    parser.add_argument("--temp", type=float, default=295,
                        help="Temperature in Kelvin [default: 295]")
    
    parser.add_argument("--visc", type=float, default=0.01,
                        help="Viscosity in poise [default: 0.01]")
    
    parser.add_argument("--bead-radius", type=float, default=5.0,
                        help="Radius of beads in Angstroms [default: 5.0]")
    
    parser.add_argument("--bead-coverage", type=float, default=1.0,
                        help="Coverage factor for bead distribution [default: 1.0]")
    
    parser.add_argument("--bead-method", type=str, default="uniform", 
                        choices=["vertices", "centroids", "uniform"],
                        help="Method for bead placement [default: uniform]")
    
    parser.add_argument("--compare-methods", action="store_true",
                        help="Run comparison between different bead methods")
    
    parser.add_argument("--optimize", action="store_true",
                        help="Find optimal bead parameters")
    
    parser.add_argument("--output-dx", type=str, default="mesh_potential.dx",
                        help="Output potential field to DX file [default: mesh_potential.dx]")
    
    parser.add_argument("--no-potential", action="store_true",
                        help="Skip potential field generation")
    
    parser.add_argument("--ellipsoid-only", action="store_true",
                        help="Use only ellipsoid approximation (no bead model)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if mesh file exists
    mesh_file = Path(args.mesh_file)
    if not mesh_file.exists():
        print(f"Error: Mesh file {args.mesh_file} not found")
        sys.exit(1)
    
    print(f"Processing mesh file: {args.mesh_file}")
    print(f"Parameters:")
    print(f"  Density: {args.density} g/cm³")
    print(f"  Temperature: {args.temp} K")
    print(f"  Viscosity: {args.visc} poise")
    
    # Time the processing
    start_time = time.time()
    
    if args.compare_methods or args.optimize:
        # Use direct BeadModelDiffusion for comparison/optimization
        bead_model = BeadModelDiffusion(
            mesh_file=args.mesh_file,
            temperature=args.temp,
            viscosity=args.visc,
            density=args.density
        )
        
        if args.compare_methods:
            print("\nComparing different bead placement methods...")
            results_df = bead_model.compare_methods(
                bead_radius=args.bead_radius,
                coverage_factors=[0.5, 1.0, 2.0],
                methods=['vertices', 'centroids', 'uniform'],
                visualize=True
            )
            
            # Print comparison results
            print("\nMethod comparison results:")
            print(results_df.to_string())
            
            # Save to CSV
            csv_filename = f"{mesh_file.stem}_method_comparison.csv"
            results_df.to_csv(csv_filename)
            print(f"Comparison results saved to {csv_filename}")
            
        if args.optimize:
            print("\nFinding optimal bead parameters...")
            optimal_radius, optimal_coverage = bead_model.find_optimal_parameters(
                bead_radii=[3.0, 5.0, 7.0, 10.0],
                coverage_factors=[0.5, 0.75, 1.0, 1.5, 2.0],
                method=args.bead_method
            )
            
            # Calculate with optimal parameters
            print("\nCalculating with optimal parameters...")
            diffusion_results = bead_model.calculate_diffusion(
                bead_radius=optimal_radius,
                coverage_factor=optimal_coverage,
                method=args.bead_method
            )
            
            # Save results
            bead_model.save_results_to_file(
                diffusion_results, 
                f"{mesh_file.stem}_optimal_bead_diffusion.txt"
            )
            
            # Try visualization
            try:
                fig, ax = bead_model.visualize_beads(
                    diffusion_results["bead_positions"], 
                    optimal_radius, 
                    show_mesh=True
                )
                fig.savefig(f"{mesh_file.stem}_optimal_beads.png")
                print(f"Optimal bead visualization saved to {mesh_file.stem}_optimal_beads.png")
            except Exception as e:
                print(f"Could not visualize optimal beads: {e}")
    else:
        # Use the MeshProcessor for standard processing
        output_dx = None if args.no_potential else args.output_dx
        
        processor = process_mesh_file(
            args.mesh_file,
            density=args.density,
            temperature=args.temp,
            viscosity=args.visc,
            output_dx=output_dx,
            use_bead_model=not args.ellipsoid_only,
            bead_radius=args.bead_radius,
            bead_coverage=args.bead_coverage
        )
        
        # Create a summary file with key results
        summary_file = f"{mesh_file.stem}_diffusion_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Mesh Diffusion Coefficient Summary\n")
            f.write("=================================\n\n")
            
            f.write(f"Input file: {args.mesh_file}\n")
            f.write(f"Shape classification: {processor.shape_type}\n")
            f.write(f"Semi-axes: a={processor.semi_axes[0]:.1f}Å, b={processor.semi_axes[1]:.1f}Å, c={processor.semi_axes[2]:.1f}Å\n\n")
            
            if not args.ellipsoid_only:
                f.write("Bead Model Results:\n")
                f.write(f"  Bead radius: {args.bead_radius}Å\n")
                f.write(f"  Coverage factor: {args.bead_coverage}\n")
                f.write(f"  Translational diffusion: {processor.D_trans_avg:.6e} m²/s\n")
                f.write(f"  Rotational diffusion: {processor.D_rot_avg:.6e} rad²/s\n\n")
                
                f.write("ARBD-Compatible Coefficients:\n")
                f.write(f"  Translational damping: {np.mean(processor.damping_coefficient):.6e} ns/Å²\n")
                f.write(f"  Rotational damping: {np.mean(processor.rotational_damping_coefficient):.6e} ns\n")
            else:
                f.write("Ellipsoid Model Results:\n")
                f.write(f"  Translational diffusion: {processor.D_trans_avg:.6e} m²/s\n")
                f.write(f"  Rotational diffusion: {processor.D_rot_avg:.6e} rad²/s\n\n")
                
                f.write("ARBD-Compatible Coefficients:\n")
                f.write(f"  Translational damping: {np.mean(processor.damping_coefficient):.6e} ns/Å²\n")
                f.write(f"  Rotational damping: {np.mean(processor.rotational_damping_coefficient):.6e} ns\n")
                
            # Compare with expected values for the nanorod example
            expected_trans = 6.87e-12  # m²/s
            expected_rot = 6000  # rad²/s
            
            f.write("\nComparison with Expected Values (nanorod example):\n")
            f.write(f"  Translational ratio: {processor.D_trans_avg/expected_trans:.2f}x expected\n")
            f.write(f"  Rotational ratio: {processor.D_rot_avg/expected_rot:.2f}x expected\n")
            
        print(f"Summary saved to {summary_file}")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Make matplotlib work without display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    main()