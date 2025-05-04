"""
Tests for blending operations in fidgetpy.ops.
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
    
    # Define a sphere and a cube
    sphere = fpm.length([x, y, z]) - 1.0
    cube = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
    
    # Define test points
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [1.1, 0.0, 0.0],
    ], dtype=np.float32)
    
    return sphere, cube, points

def test_blend():
    """Test blend operation"""
    sphere, cube, points = setup_test_shapes()
    blend_sdf = fpo.blend(sphere, cube, 0.5)
    result = fp.eval(blend_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Blend should be a linear interpolation
    sphere_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    cube_values = fp.eval(cube, points, [fp.x(), fp.y(), fp.z()])
    expected = sphere_values * 0.5 + cube_values * 0.5
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)
        
def test_smooth_min():
    """Test smooth min operation"""
    sphere, cube, points = setup_test_shapes()
    # Just verify it runs without errors
    smooth_min_sdf = fpo.smooth_min(sphere, cube, 0.2)
    result = fp.eval(smooth_min_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
        
def test_exponential_smooth_min():
    """Test exponential smooth min operation"""
    sphere, cube, points = setup_test_shapes()
    # Just verify it runs without errors
    exp_smooth_min_sdf = fpo.exponential_smooth_min(sphere, cube, 0.2)
    result = fp.eval(exp_smooth_min_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
        
def test_power_smooth_min():
    """Test power smooth min operation"""
    sphere, cube, points = setup_test_shapes()
    # Just verify it runs without errors
    power_smooth_min_sdf = fpo.power_smooth_min(sphere, cube, 8.0)
    result = fp.eval(power_smooth_min_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)

def test_smooth_max():
    """Test smooth max operation"""
    sphere, cube, points = setup_test_shapes()
    # Verify it runs without errors
    smooth_max_sdf = fpo.smooth_max(sphere, cube, 0.2)
    result = fp.eval(smooth_max_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
    
    # Basic validation: with k=0, should match regular max
    sphere_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    cube_values = fp.eval(cube, points, [fp.x(), fp.y(), fp.z()])
    expected_max = np.maximum(sphere_values, cube_values)
    zero_smooth = fp.eval(fpo.smooth_max(sphere, cube, 0.000001), points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(expected_max, zero_smooth, atol=1e-5)

def test_exponential_smooth_max():
    """Test exponential smooth max operation"""
    sphere, cube, points = setup_test_shapes()
    # Verify it runs without errors
    exp_smooth_max_sdf = fpo.exponential_smooth_max(sphere, cube, 0.2)
    result = fp.eval(exp_smooth_max_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
    
def test_power_smooth_max():
    """Test power smooth max operation"""
    sphere, cube, points = setup_test_shapes()
    # Verify it runs without errors
    power_smooth_max_sdf = fpo.power_smooth_max(sphere, cube, 8.0)
    result = fp.eval(power_smooth_max_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
    
def test_mix_lerp_aliases():
    """Test mix and lerp as aliases for blend"""
    sphere, cube, points = setup_test_shapes()
    
    # Create reference blend
    blend_sdf = fpo.blend(sphere, cube, 0.5)
    blend_values = fp.eval(blend_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Test mix
    mix_sdf = fpo.mix(sphere, cube, 0.5)
    mix_values = fp.eval(mix_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(blend_values, mix_values)
    
    # Test lerp
    lerp_sdf = fpo.lerp(sphere, cube, 0.5)
    lerp_values = fp.eval(lerp_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(blend_values, lerp_values)

def test_soft_clamp():
    """Test soft_clamp operation"""
    sphere, cube, points = setup_test_shapes()
    
    # Verify it runs without errors
    soft_clamp_sdf = fpo.soft_clamp(sphere, -0.5, 0.5, 0.1)
    result = fp.eval(soft_clamp_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
    
    # Basic validation: with k=0, should match hard clamp
    sphere_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    expected_clamped = np.clip(sphere_values, -0.5, 0.5)
    zero_smooth = fp.eval(fpo.soft_clamp(sphere, -0.5, 0.5, 0.000001), points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(expected_clamped, zero_smooth, atol=1e-5)

def test_quad_bezier_blend():
    """Test quadratic bezier blending"""
    sphere, cube, points = setup_test_shapes()
    
    # Create a third shape for the control point
    x, y, z = fp.x(), fp.y(), fp.z()
    cylinder = fpm.max(fpm.length([x, y]) - 0.7, fpm.abs(z) - 1.2)
    
    # Test with t=0.5
    bezier_sdf = fpo.quad_bezier_blend(sphere, cylinder, cube, 0.5)
    result = fp.eval(bezier_sdf, points, [fp.x(), fp.y(), fp.z()])
    assert len(result) == len(points)
    
    # Test with custom t value
    bezier_sdf_t = fpo.quad_bezier_blend(sphere, cylinder, cube, 0.75)
    result_t = fp.eval(bezier_sdf_t, points, [fp.x(), fp.y(), fp.z()])
    assert len(result_t) == len(points)