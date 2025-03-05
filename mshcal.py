import numpy as np
import gmsh
import math
from scipy.linalg import eigh

def calculate_diffusion_tensors(mesh_file, temperature=293.15, viscosity=0.001, unit_scale=1e-6):
    """
    Calculate diagonal elements of translational and rotational diffusion tensors for a mesh structure.
    
    Parameters:
    -----------
    mesh_file : str
        Path to the mesh file (.msh format)
    temperature : float
        Temperature in Kelvin (default: 293.15 K, room temperature)
    viscosity : float
        Fluid viscosity in Pa·s (default: 0.001 Pa·s, water at 20°C)
    unit_scale : float
        Scale factor to convert mesh units to meters (default: 1e-6 for microns)
        
    Returns:
    --------
    trans_diff : numpy.ndarray
        Diagonal elements of translational diffusion tensor [Dx, Dy, Dz] in m²/s
    rot_diff : numpy.ndarray
        Diagonal elements of rotational diffusion tensor [Drx, Dry, Drz] in rad²/s
    """
    # Initialize gmsh
    gmsh.initialize()
    
    try:
        # Read the mesh file
        gmsh.open(mesh_file)
        
        # Get all nodes in the mesh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        
        # Reshape coordinates into points array
        vertices = np.array(node_coords).reshape(-1, 3)
        
        # Apply unit scale to convert from mesh units to meters
        vertices *= unit_scale
        
        if len(vertices) == 0:
            print("Error: Mesh contains no vertices")
            gmsh.finalize()
            return None, None
            
        print(f"Mesh contains {len(vertices)} vertices")
        
        # Calculate center of mass
        com = np.mean(vertices, axis=0)
        
        # Shift vertices to center of mass
        vertices_centered = vertices - com
        
        # Calculate gyration tensor (used for translational diffusion)
        gyration_tensor = np.zeros((3, 3))
        for v in vertices_centered:
            gyration_tensor += np.outer(v, v)
        gyration_tensor /= len(vertices)
        
        # Calculate inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for v in vertices_centered:
            r_squared = np.sum(v**2)
            for i in range(3):
                # Diagonal terms
                inertia_tensor[i, i] += r_squared - v[i]**2
                # Off-diagonal terms (negative)
                for j in range(i+1, 3):
                    inertia_tensor[i, j] -= v[i] * v[j]
                    inertia_tensor[j, i] = inertia_tensor[i, j]  # Symmetric
        
        # Estimate volume using convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            volume = hull.volume
        except ImportError:
            # Fallback: estimate volume from gyration tensor eigenvalues
            eigenvalues = np.linalg.eigvalsh(gyration_tensor)
            semi_axes = np.sqrt(eigenvalues)
            volume = 4/3 * np.pi * np.prod(semi_axes)
        
        # Estimate mass (assuming water density)
        density = 1000  # kg/m³
        mass = volume * density
        
        # Scale inertia tensor by mass
        inertia_tensor *= mass / len(vertices)
        
        # Get eigenvalues and eigenvectors of gyration tensor
        gyr_eigenvals, gyr_eigenvecs = eigh(gyration_tensor)
        inertia_eigenvals, inertia_eigenvecs = eigh(inertia_tensor)
        
        # Sort eigenvalues and eigenvectors
        sort_idx = np.argsort(gyr_eigenvals)[::-1]  # Descending order
        gyr_eigenvals = gyr_eigenvals[sort_idx]
        gyr_eigenvecs = gyr_eigenvecs[:, sort_idx]
        
        sort_idx = np.argsort(inertia_eigenvals)[::-1]  # Descending order
        inertia_eigenvals = inertia_eigenvals[sort_idx]
        inertia_eigenvecs = inertia_eigenvecs[:, sort_idx]
        
        # Calculate semi-axes of equivalent ellipsoid from gyration tensor
        # Using scaling factor of 5 (empirical value that works well)
        semi_axes = np.sqrt(5.0 * gyr_eigenvals)
        a, b, c = semi_axes  # Largest to smallest
        
        # Determine shape type
        tol = 0.05  # Tolerance for considering axes equal
        if np.isclose(a, b, rtol=tol) and np.isclose(b, c, rtol=tol):
            shape_type = "Sphere"
        elif np.isclose(a, b, rtol=tol):
            shape_type = "Oblate"  # Disk-like (a ≈ b > c)
        elif np.isclose(b, c, rtol=tol):
            shape_type = "Prolate"  # Rod-like (a > b ≈ c)
        else:
            shape_type = "Triaxial"  # General ellipsoid
            
        print(f"Shape classification: {shape_type}")
        print(f"Semi-axes of equivalent ellipsoid: a={a:.6f}, b={b:.6f}, c={c:.6f} m")
        
        # Calculate friction coefficients based on shape type
        if shape_type == "Sphere":
            # Use average radius
            R = (a + b + c) / 3
            
            # Translational friction (kg/s)
            trans_friction = 6 * np.pi * viscosity * R * np.ones(3)
            
            # Rotational friction (kg·m²/s)
            rot_friction = 8 * np.pi * viscosity * R**3 * np.ones(3)
            
        elif shape_type == "Prolate":
            # Prolate ellipsoid (a > b ≈ c)
            e = np.sqrt(1 - (b/a)**2)  # Eccentricity
            
            # Shape factor for prolate ellipsoid
            if e > 0.99:  # Very elongated shapes
                S = 2 * np.log(2*a/b) - 0.5
            else:
                S = 2 * np.log((1 + e)/(1 - e)) / e - 2*e/(1 - e**2)
            
            # Translational friction
            gamma_a = 6 * np.pi * viscosity * b / S  # Along major axis
            gamma_bc = 6 * np.pi * viscosity * b / (0.5 * S + 1)  # Perpendicular
            trans_friction = np.array([gamma_a, gamma_bc, gamma_bc])
            
            # Volume
            V = 4/3 * np.pi * a * b * c
            
            # Rotational friction
            gamma_rot_a = 6 * viscosity * V * ((1 - e**2) / e**2) * \
                         (-2*e/(1-e**2) + np.log((1+e)/(1-e)))  # Around minor axes
            gamma_rot_bc = 6 * viscosity * V * ((1 + e**2) / e**2) * \
                          (2*e/(1-e**2) - (1-e**2)/(2*e) * np.log((1+e)/(1-e)))  # Around major axis
            rot_friction = np.array([gamma_rot_bc, gamma_rot_a, gamma_rot_a])
            
        elif shape_type == "Oblate":
            # Oblate ellipsoid (a ≈ b > c)
            e = np.sqrt(1 - (c/a)**2)  # Eccentricity
            
            # Shape factor
            if e > 0.99:  # Very flat shapes
                S = np.pi * a / (2 * c)
            else:
                S = 2 * np.arctan(e/np.sqrt(1-e**2)) / (e * np.sqrt(1-e**2))
            
            # Translational friction
            gamma_ab = 6 * np.pi * viscosity * a / (1 + 0.5*S*(1-e**2)/e)  # In-plane
            gamma_c = 6 * np.pi * viscosity * a / (S*(1-e**2)/e)  # Normal
            trans_friction = np.array([gamma_ab, gamma_ab, gamma_c])
            
            # Volume
            V = 4/3 * np.pi * a * b * c
            
            # Rotational friction
            gamma_rot_c = 6 * viscosity * V * ((2 - e**2) / e**2) * \
                         (e/(1-e**2) - 0.5 * S)  # Around symmetry axis
            gamma_rot_ab = 6 * viscosity * V * ((2 + e**2) / e**2) * \
                          (0.5 * S - e/(1-e**2))  # Around in-plane axes
            rot_friction = np.array([gamma_rot_ab, gamma_rot_ab, gamma_rot_c])
            
        else:  # Triaxial ellipsoid
            # Equivalent radius
            R_eq = (a * b * c)**(1/3)
            
            # Correction factors based on axis ratios
            alpha_a = 1 - 0.25 * (1 - (a/R_eq)**(-2))
            alpha_b = 1 - 0.25 * (1 - (b/R_eq)**(-2))
            alpha_c = 1 - 0.25 * (1 - (c/R_eq)**(-2))
            
            # Translational friction
            gamma_a = 6 * np.pi * viscosity * R_eq / alpha_a
            gamma_b = 6 * np.pi * viscosity * R_eq / alpha_b
            gamma_c = 6 * np.pi * viscosity * R_eq / alpha_c
            trans_friction = np.array([gamma_a, gamma_b, gamma_c])
            
            # Rotational friction approximation
            beta_a = ((b**2 - c**2)/(b**2 + c**2))**2
            beta_b = ((a**2 - c**2)/(a**2 + c**2))**2
            beta_c = ((a**2 - b**2)/(a**2 + b**2))**2
            
            gamma_rot_a = 8 * np.pi * viscosity * (b**2 + c**2) / 3 * (1 + beta_a)
            gamma_rot_b = 8 * np.pi * viscosity * (a**2 + c**2) / 3 * (1 + beta_b)
            gamma_rot_c = 8 * np.pi * viscosity * (a**2 + b**2) / 3 * (1 + beta_c)
            rot_friction = np.array([gamma_rot_a, gamma_rot_b, gamma_rot_c])
        
        # Apply kBT at the final step to get diffusion coefficients
        kB = 1.380649e-23  # J/K
        kBT = kB * temperature
        
        # Calculate diffusion coefficients
        trans_diff = kBT / trans_friction  # m²/s
        rot_diff = kBT / rot_friction      # rad²/s
        
        print("\nFriction coefficients:")
        print(f"Translational friction (kg/s): {trans_friction}")
        print(f"Rotational friction (kg·m²/s): {rot_friction}")
        
        print("\nDiffusion coefficients:")
        print(f"Translational diffusion (m²/s): {trans_diff}")
        print(f"Rotational diffusion (rad²/s): {rot_diff}")
        
        # Convert to ARBD units for reference
        print("\nDamping coefficients (ARBD units):")
        print(f"Translational damping (1/ns): {trans_friction * 6.02214076e20 / 1000:.3e}")
        print(f"Rotational damping (1/ns): {rot_friction * 6.02214076 / 1000:.3e}")
        
        return trans_diff, rot_diff
        
    except Exception as e:
        print(f"Error processing mesh: {e}")
        return None, None
    
    finally:
        gmsh.finalize()

if __name__ == "__main__":
    import sys
    
    # Default parameters
    mesh_file = "Nanorod.msh"
    temperature = 293.15  # Room temperature in K
    viscosity = 0.001     # Water viscosity at 20°C in Pa·s
    unit_scale = 1e-6     # Microns to meters
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        temperature = float(sys.argv[2])
    if len(sys.argv) > 3:
        viscosity = float(sys.argv[3])
    if len(sys.argv) > 4:
        unit_scale = float(sys.argv[4])
    
    print(f"Calculating diffusion coefficients for mesh: {mesh_file}")
    print(f"Temperature: {temperature} K")
    print(f"Viscosity: {viscosity} Pa·s")
    print(f"Unit scale: {unit_scale} (mesh units to meters)")
    
    # Calculate diffusion tensors
    trans_diff, rot_diff = calculate_diffusion_tensors(
        mesh_file, temperature, viscosity, unit_scale)
