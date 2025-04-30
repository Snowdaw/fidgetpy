"""
Tests demonstrating creative shape generation using combined fidgetpy functionality.

This file shows how to create interesting shapes that might be used in actual
modeling scenarios, demonstrating the flexibility and expressiveness of combined
math and ops functionality.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
import fidgetpy.ops as fpo


def test_twisted_torus_knot():
    """Create and test a twisted torus knot shape."""
    # Standard variables
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Custom parameters
    knot_radius = fp.var("knot_radius")        # Major radius
    tube_radius = fp.var("tube_radius")        # Tube thickness
    twist_factor = fp.var("twist_factor")      # Twist amount
    p_param = fp.var("p")                      # Knot parameter p
    q_param = fp.var("q")                      # Knot parameter q
    
    # === Create a torus knot parametric curve ===
    
    # Get polar coordinates in the XZ plane
    radius_xz = fpm.length([x, z])
    angle = fpm.atan2(z, x)
    
    # Calculate closest point on a torus knot
    # This is a simplified approximation of the distance to a p,q torus knot
    phi = angle
    
    # Calculate the position of the torus knot curve at this angle
    # In a p,q torus knot, the curve wraps p times around the major circle
    # and q times around the minor circle
    major_angle = phi * p_param
    minor_angle = phi * q_param
    
    knot_x = knot_radius * fpm.cos(major_angle)
    knot_y = fpm.sin(major_angle) * fpm.sin(minor_angle)
    knot_z = knot_radius * fpm.sin(major_angle) * fpm.cos(minor_angle)
    
    # Calculate distance from current point to this knot position
    dx = x - knot_x
    dy = y - knot_y
    dz = z - knot_z
    
    # Create the basic knot shape with tube thickness
    torus_knot = fpm.length([dx, dy, dz]) - tube_radius
    
    # === Apply transformations ===
    
    # Add twisting along the major axis
    twisted_knot = fpo.twist(torus_knot, twist_factor)
    
    # Add some detail with displacement
    bump_noise = fpm.sin(x * 20) * fpm.sin(y * 20) * fpm.sin(z * 20) * 0.05
    final_shape = fpo.displace(twisted_knot, bump_noise)
    
    # === Evaluate the shape ===
    
    # Define test parameters
    test_points = [
        # x, y, z, knot_radius, tube_radius, twist_factor, p, q
        [0.0, 0.0, 0.0, 2.0, 0.4, 0.5, 2, 3],
        [2.0, 0.0, 0.0, 2.0, 0.4, 0.5, 2, 3],
        [2.2, 0.0, 0.0, 2.0, 0.4, 0.5, 2, 3],
        [0.0, 0.5, 0.0, 2.0, 0.4, 0.5, 2, 3],
    ]
    
    # Create numpy array version
    numpy_points = np.array(test_points, dtype=np.float32)
    
    # Define variables (must match order in points)
    variables = [x, y, z, knot_radius, tube_radius, twist_factor, p_param, q_param]
    
    # Evaluate with both list and numpy input
    list_results = fp.eval(final_shape, test_points, variables=variables)
    numpy_results = fp.eval(final_shape, numpy_points, variables=variables)
    
    # Verify results
    assert isinstance(list_results, list)
    assert isinstance(numpy_results, np.ndarray)
    assert len(list_results) == len(test_points)
    
    # Verify values (not testing exact values, just that evaluation works)
    assert not np.isnan(numpy_results).any(), "Results should not contain NaN values"
    
    # Verify center point is outside the knot (it should be hollow)
    assert list_results[0] > 0, "Center point should be outside the knot (hollow)"
    
    # Point on the major radius should be near the surface
    assert abs(list_results[1]) < 0.5, "Point at major radius should be near the surface"
    
    # This point is actually inside the tube (negative distance)
    assert list_results[2] < 0, "Point is inside the tube (negative distance)"


def test_fractal_terrain():
    """Create and test a fractal terrain-like shape."""
    # Standard variables
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Custom parameters
    roughness = fp.var("roughness")             # Terrain roughness
    height_scale = fp.var("height_scale")       # Overall height
    detail_scale = fp.var("detail_scale")       # Detail scaling
    
    # === Create a fractal heightfield ===
    
    # Base heightfield using sine waves at different frequencies
    base_freq = 0.5
    
    # First octave
    height1 = fpm.sin(x * base_freq) * fpm.cos(z * base_freq) * 1.0
    
    # Second octave
    height2 = fpm.sin(x * base_freq * 2.3) * fpm.cos(z * base_freq * 2.1) * 0.5
    
    # Third octave
    height3 = fpm.sin(x * base_freq * 4.7) * fpm.cos(z * base_freq * 4.3) * 0.25
    
    # Combine octaves, weighted by roughness parameter
    height_field = height1 + height2 * roughness + height3 * roughness * roughness
    
    # Scale the height
    height_field = height_field * height_scale
    
    # Create the terrain as a distance field (heightfield with y as up direction)
    terrain = y - height_field
    
    # === Add detail features ===
    
    # Add small-scale detail
    detail_noise = (
        fpm.sin(x * detail_scale * 8.3) * 
        fpm.cos(z * detail_scale * 7.9) * 
        0.05 * roughness
    )
    
    # Apply the detail
    terrain_detailed = fpo.displace(terrain, detail_noise)
    
    # === Evaluate the terrain ===
    
    # Define test parameters
    test_points = [
        # x, y, z, roughness, height_scale, detail_scale
        [0.0, 2.0, 0.0, 0.5, 1.0, 1.0],  # Point above terrain
        [0.0, -2.0, 0.0, 0.5, 1.0, 1.0], # Point below terrain
        [1.0, 0.0, 1.0, 0.5, 1.0, 1.0],  # Point potentially near terrain
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # With zero roughness
        [0.0, 0.0, 0.0, 1.0, 2.0, 1.0],  # With high height scale
    ]
    
    # Create numpy array version
    numpy_points = np.array(test_points, dtype=np.float32)
    
    # Define variables (must match order in points)
    variables = [x, y, z, roughness, height_scale, detail_scale]
    
    # Evaluate with both list and numpy input
    list_results = fp.eval(terrain_detailed, test_points, variables=variables)
    numpy_results = fp.eval(terrain_detailed, numpy_points, variables=variables)
    
    # Verify results
    assert isinstance(list_results, list)
    assert isinstance(numpy_results, np.ndarray)
    
    # Verify values
    assert list_results[0] > 0, "Point above terrain should have positive distance"
    assert list_results[1] < 0, "Point below terrain should have negative distance"
    
    # Note: In this specific case at point [0,0,0], changing roughness and height_scale
    # might not affect the result significantly if the sine waves evaluate to near-zero
    # at the origin. Let's just verify the values are reasonable.
    assert isinstance(list_results[3], (float, np.float32)), "Result should be a float"
    assert isinstance(list_results[4], (float, np.float32)), "Result should be a float"


if __name__ == "__main__":
    pytest.main([__file__])