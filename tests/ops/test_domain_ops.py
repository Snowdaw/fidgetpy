"""
Tests for domain operations in fidgetpy.ops.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
import fidgetpy.ops as fpo

def setup_test_shapes():
    """Set up common test shapes"""
    # Create a basic shape for testing
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Define a sphere
    sphere = fpm.length([x, y, z]) - 1.0
    
    # Define test points
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    
    return sphere, points

def test_onion():
    """Test onion operation"""
    sphere, points = setup_test_shapes()
    thickness = 0.1
    onion_sdf = fpo.onion(sphere, thickness)
    result = fp.eval(onion_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Onion should be the absolute value minus thickness
    sphere_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    expected = np.abs(sphere_values) - thickness
    
    np.testing.assert_allclose(result, expected, rtol=1e-5)
        
def test_mirror_x():
    """Test mirror_x operation"""
    sphere, points = setup_test_shapes()
    mirrored_sdf = fpo.mirror_x(sphere)
    
    # Create points with negative x to test mirroring
    mirror_test_points = np.array([
        [-0.5, 0.3, 0.2],
        [0.5, 0.3, 0.2]  # Same but with positive x
    ], dtype=np.float32)
    
    result = fp.eval(mirrored_sdf, mirror_test_points, [fp.x(), fp.y(), fp.z()])
    
    # Both points should give the same result after mirroring
    assert result[0] == pytest.approx(result[1], rel=1e-5)

def test_mirror_y():
    """Test mirror_y operation"""
    sphere, points = setup_test_shapes()
    mirrored_sdf = fpo.mirror_y(sphere)
    
    # Create points with negative y to test mirroring
    mirror_test_points = np.array([
        [0.3, -0.5, 0.2],
        [0.3, 0.5, 0.2]  # Same but with positive y
    ], dtype=np.float32)
    
    result = fp.eval(mirrored_sdf, mirror_test_points, [fp.x(), fp.y(), fp.z()])
    
    # Both points should give the same result after mirroring
    assert result[0] == pytest.approx(result[1], rel=1e-5)

def test_mirror_z():
    """Test mirror_z operation"""
    sphere, points = setup_test_shapes()
    mirrored_sdf = fpo.mirror_z(sphere)
    
    # Create points with negative z to test mirroring
    mirror_test_points = np.array([
        [0.3, 0.2, -0.5],
        [0.3, 0.2, 0.5]  # Same but with positive z
    ], dtype=np.float32)
    
    result = fp.eval(mirrored_sdf, mirror_test_points, [fp.x(), fp.y(), fp.z()])
    
    # Both points should give the same result after mirroring
    assert result[0] == pytest.approx(result[1], rel=1e-5)

def test_elongate():
    """Test elongate operation"""
    sphere, points = setup_test_shapes()
    
    # Elongate along z-axis
    elongate_sdf = fpo.elongate(sphere, (0.0, 0.0, 1.0))
    result = fp.eval(elongate_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Check that elongating by zero gives the original shape
    no_elongate_sdf = fpo.elongate(sphere, (0.0, 0.0, 0.0))
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    no_elongate_values = fp.eval(no_elongate_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(original_values, no_elongate_values, rtol=1e-5)

def test_twist():
    """Test twist operation"""
    sphere, points = setup_test_shapes()
    
    # Twist around the z-axis
    twist_sdf = fpo.twist(sphere, 1.0)
    result = fp.eval(twist_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Check that twisting by zero gives the original shape
    no_twist_sdf = fpo.twist(sphere, 0.0)
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    no_twist_values = fp.eval(no_twist_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(original_values, no_twist_values, rtol=1e-5)

def test_bend():
    """Test bend operation"""
    sphere, points = setup_test_shapes()
    
    # Bend along the x-axis
    bend_sdf = fpo.bend(sphere, 1.0)
    result = fp.eval(bend_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Check that bending by zero gives the original shape
    no_bend_sdf = fpo.bend(sphere, 0.0)
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    no_bend_values = fp.eval(no_bend_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(original_values, no_bend_values, rtol=1e-5)

def test_round():
    """Test round operation"""
    sphere, points = setup_test_shapes()
    
    # Round the shape
    radius = 0.1
    rounded_sdf = fpo.round(sphere, radius)
    result = fp.eval(rounded_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Rounding should decrease the distance by the radius
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    expected = original_values - radius
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_shell():
    """Test shell operation"""
    sphere, points = setup_test_shapes()
    
    # Create shell of the shape
    thickness = 0.1
    shell_sdf = fpo.shell(sphere, thickness)
    result = fp.eval(shell_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Shell should be the absolute value minus thickness
    # Similar to onion in this case of a sphere
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    expected = np.abs(original_values) - thickness
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_displace():
    """Test displace operation"""
    sphere, points = setup_test_shapes()
    
    # Create a displacement field (simple sinusoidal displacement)
    x, y, z = fp.x(), fp.y(), fp.z()
    displacement = fpm.sin(x * 5.0) * 0.1
    
    # Apply displacement
    displaced_sdf = fpo.displace(sphere, displacement)
    result = fp.eval(displaced_sdf, points, [fp.x(), fp.y(), fp.z()])
    
    # Verify it runs without errors
    assert len(result) == len(points)
    
    # Zero displacement should give the original shape
    no_displace_sdf = fpo.displace(sphere, 0.0)
    original_values = fp.eval(sphere, points, [fp.x(), fp.y(), fp.z()])
    no_displace_values = fp.eval(no_displace_sdf, points, [fp.x(), fp.y(), fp.z()])
    np.testing.assert_allclose(original_values, no_displace_values, rtol=1e-5)