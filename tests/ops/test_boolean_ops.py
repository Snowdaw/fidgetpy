"""
Tests for boolean operations in fidgetpy.ops.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
import fidgetpy.ops as fpo

def setup_test_shapes():
    """Set up common test shapes"""
    # Create two basic shapes for testing
    x, y, z = fp.x(), fp.y(), fp.z()
    p = [x, y, z]
    
    # Define a sphere and a cube
    sphere = fpm.length(p) - 1.0
    cube = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
    
    # Define test points
    points = np.array([
        [0.0, 0.0, 0.0],    # Inside both
        [0.9, 0.0, 0.0],    # Inside sphere, inside cube
        [1.1, 0.0, 0.0],    # Outside sphere, inside cube
        [0.0, 0.9, 0.0],    # Inside sphere, inside cube
        [0.0, 0.0, 0.9],    # Inside sphere, inside cube
        [0.9, 0.9, 0.0],    # Inside sphere, inside cube
        [1.1, 1.1, 0.0],    # Outside sphere, outside cube
    ], dtype=np.float32)
    
    return sphere, cube, points

def test_union():
    """Test union operation"""
    sphere, cube, points = setup_test_shapes()
    union_sdf = fpo.union(sphere, cube)
    result = fp.eval(union_sdf, points)
    
    # Union should be the minimum of the two SDFs
    sphere_values = fp.eval(sphere, points)
    cube_values = fp.eval(cube, points)
    expected = np.minimum(sphere_values, cube_values)
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
def test_intersection():
    """Test intersection operation"""
    sphere, cube, points = setup_test_shapes()
    intersection_sdf = fpo.intersection(sphere, cube)
    result = fp.eval(intersection_sdf, points)
    
    # Intersection should be the maximum of the two SDFs
    sphere_values = fp.eval(sphere, points)
    cube_values = fp.eval(cube, points)
    expected = np.maximum(sphere_values, cube_values)
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
def test_difference():
    """Test difference operation"""
    sphere, cube, points = setup_test_shapes()
    difference_sdf = fpo.difference(sphere, cube)
    result = fp.eval(difference_sdf, points)
    
    # Difference should be max(a, -b)
    sphere_values = fp.eval(sphere, points)
    cube_values = fp.eval(cube, points)
    expected = np.maximum(sphere_values, -cube_values)
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_smooth_union():
    """Test smooth union operation"""
    sphere, cube, points = setup_test_shapes()
    # Just verify it runs without errors; exact values depend on implementation
    smooth_union_sdf = fpo.smooth_union(sphere, cube, 0.2)
    result = fp.eval(smooth_union_sdf, points)
    assert len(result) == len(points)
    
def test_complement():
    """Test complement operation"""
    sphere, cube, points = setup_test_shapes()
    complement_sdf = fpo.complement(sphere)
    result = fp.eval(complement_sdf, points)
    
    # Complement should negate the SDF
    sphere_values = fp.eval(sphere, points)
    expected = -sphere_values
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_smooth_intersection():
    """Test smooth intersection operation"""
    sphere, cube, points = setup_test_shapes()
    # Verify it runs without errors
    smooth_intersection_sdf = fpo.smooth_intersection(sphere, cube, 0.2)
    result = fp.eval(smooth_intersection_sdf, points)
    assert len(result) == len(points)
    
    # Basic validation: with k=0, should match regular intersection
    regular_intersection = fp.eval(fpo.intersection(sphere, cube), points)
    zero_smooth = fp.eval(fpo.smooth_intersection(sphere, cube, 0.000001), points)
    np.testing.assert_allclose(regular_intersection, zero_smooth, atol=1e-5)

def test_smooth_difference():
    """Test smooth difference operation"""
    sphere, cube, points = setup_test_shapes()
    # Verify it runs without errors
    smooth_difference_sdf = fpo.smooth_difference(sphere, cube, 0.2)
    result = fp.eval(smooth_difference_sdf, points)
    assert len(result) == len(points)
    
    # Basic validation: with k=0, should match regular difference
    regular_difference = fp.eval(fpo.difference(sphere, cube), points)
    zero_smooth = fp.eval(fpo.smooth_difference(sphere, cube, 0.000001), points)
    np.testing.assert_allclose(regular_difference, zero_smooth, atol=1e-5)

def test_boolean_aliases():
    """Test boolean operation aliases"""
    sphere, cube, points = setup_test_shapes()
    
    # Test boolean_and (alias for intersection)
    intersection_sdf = fpo.intersection(sphere, cube)
    boolean_and_sdf = fpo.boolean_and(sphere, cube)
    np.testing.assert_allclose(
        fp.eval(intersection_sdf, points),
        fp.eval(boolean_and_sdf, points)
    )
    
    # Test boolean_or (alias for union)
    union_sdf = fpo.union(sphere, cube)
    boolean_or_sdf = fpo.boolean_or(sphere, cube)
    np.testing.assert_allclose(
        fp.eval(union_sdf, points),
        fp.eval(boolean_or_sdf, points)
    )
    
    # Test boolean_not (alias for complement)
    complement_sdf = fpo.complement(sphere)
    boolean_not_sdf = fpo.boolean_not(sphere)
    np.testing.assert_allclose(
        fp.eval(complement_sdf, points),
        fp.eval(boolean_not_sdf, points)
    )

def test_boolean_xor():
    """Test boolean XOR operation"""
    sphere, cube, points = setup_test_shapes()
    xor_sdf = fpo.boolean_xor(sphere, cube)
    result = fp.eval(xor_sdf, points)
    
    # XOR can be defined as (A ∪ B) \ (A ∩ B)
    union_sdf = fpo.union(sphere, cube)
    intersection_sdf = fpo.intersection(sphere, cube)
    expected_xor = fp.eval(fpo.difference(union_sdf, intersection_sdf), points)
    
    np.testing.assert_allclose(result, expected_xor)