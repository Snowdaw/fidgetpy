"""
Tests for domain manipulation functions in fidgetpy.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm

# Define common variables and constants for tests
X = fp.x()
Y = fp.y()
Z = fp.z()

# Sample points for evaluation (x, y, z)
SAMPLE_POINTS_NP = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, -0.5, 1.5],
    [-1.0, -1.0, -1.0],
], dtype=np.float32)

def evaluate(expr, points=SAMPLE_POINTS_NP, variables=None):
    """Helper to evaluate expressions, handling default/custom vars."""
    if variables:
        # Assume points includes values for custom vars if variables are provided
        return fp.eval(expr, points, variables=variables)
    else:
        # Default: evaluate using x, y, z from the points array
        return fp.eval(expr, points)  # Rely on default variables=[x,y,z]

def test_mirror():
    """Test mirroring functions."""
    # mirror_x
    expr_abs_x = fpm.abs(X)  # |x|
    mirrored_expr = fpm.mirror_x(X)  # Should behave like |x|
    
    np.testing.assert_allclose(evaluate(expr_abs_x), evaluate(mirrored_expr))
    
    # Also test as a method
    mirrored_expr_method = X.mirror_x()
    np.testing.assert_allclose(evaluate(expr_abs_x), evaluate(mirrored_expr_method))
    
    # Test mirror_y
    expr_abs_y = fpm.abs(Y)  # |y|
    mirrored_expr_y = fpm.mirror_y(Y)  # Should behave like |y|
    
    np.testing.assert_allclose(evaluate(expr_abs_y), evaluate(mirrored_expr_y))
    
    # Test mirror_z
    expr_abs_z = fpm.abs(Z)  # |z|
    mirrored_expr_z = fpm.mirror_z(Z)  # Should behave like |z|
    
    np.testing.assert_allclose(evaluate(expr_abs_z), evaluate(mirrored_expr_z))

def test_repeat():
    """Test repeat function."""
    # repeat - test with a simple function
    # Creating points specifically for this test
    repeat_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.9, 0.0, 0.0],  # Just before 2.0, should be same as 1.9 - 2
        [2.1, 0.0, 0.0],  # Just after 2.0, should wrap back
        [3.0, 0.0, 0.0],  # Should wrap back
        [-0.5, 0.0, 0.0]  # Should be mapped within the repeat range
    ], dtype=np.float32)
    
    # Test repeat just in x dimension
    expr_repeat_x = fpm.repeat_x(X, 2.0)
    
    # Calculate expected values based on our implementation
    # x - period * floor(x / period + 0.5)
    expected_x = np.zeros(len(repeat_points))
    for i, point in enumerate(repeat_points):
        x = point[0]
        period = 2.0
        expected_x[i] = x - period * np.floor(x / period + 0.5)
    
    result_repeat_x = evaluate(expr_repeat_x, repeat_points)
    np.testing.assert_allclose(result_repeat_x, expected_x, atol=1e-6)
    # Test method call
    expr_repeat_x_method = X.repeat_x(2.0)
    result_repeat_x_method = evaluate(expr_repeat_x_method, repeat_points)
    np.testing.assert_allclose(result_repeat_x_method, expected_x, atol=1e-6)

    # Test repeat_y
    repeat_points_y = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.9, 0.0],
        [0.0, 2.1, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, -0.5, 0.0]
    ], dtype=np.float32)

    expr_repeat_y = fpm.repeat_y(Y, 2.0)

    # Calculate expected values
    expected_y = np.zeros(len(repeat_points_y))
    for i, point in enumerate(repeat_points_y):
        y = point[1]
        period = 2.0
        expected_y[i] = y - period * np.floor(y / period + 0.5)

    result_repeat_y = evaluate(expr_repeat_y, repeat_points_y)
    np.testing.assert_allclose(result_repeat_y, expected_y, atol=1e-6)
    # Test method call
    expr_repeat_y_method = Y.repeat_y(2.0)
    result_repeat_y_method = evaluate(expr_repeat_y_method, repeat_points_y)
    np.testing.assert_allclose(result_repeat_y_method, expected_y, atol=1e-6)

    # Test repeat_z
    repeat_points_z = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.9],
        [0.0, 0.0, 2.1],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -0.5]
    ], dtype=np.float32)

    expr_repeat_z = fpm.repeat_z(Z, 2.0)

    # Calculate expected values
    expected_z = np.zeros(len(repeat_points_z))
    for i, point in enumerate(repeat_points_z):
        z = point[2]
        period = 2.0
        expected_z[i] = z - period * np.floor(z / period + 0.5)

    result_repeat_z = evaluate(expr_repeat_z, repeat_points_z)
    np.testing.assert_allclose(result_repeat_z, expected_z, atol=1e-6)
    # Test method call
    expr_repeat_z_method = Z.repeat_z(2.0)
    result_repeat_z_method = evaluate(expr_repeat_z_method, repeat_points_z)
    np.testing.assert_allclose(result_repeat_z_method, expected_z, atol=1e-6)


def test_multi_dimensional_repeat():
    """Test multi-dimensional repeat function."""
    # Test repeating in multiple dimensions
    repeat_points_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.1, 0.5, 0.0],   # x past period, y and z within period
        [0.5, 2.1, 0.5],   # y past period, x and z within period
        [0.5, 0.5, 2.1],   # z past period, x and y within period
        [2.1, 2.1, 2.1],   # all past period
        [-0.5, -0.5, -0.5], # all negative within period
    ], dtype=np.float32)

    # Test repeat in xy with z unchanged (repeat_xyz)
    expr_to_repeat = X + Y # Use a simple expression
    expr_repeat_xy = fpm.repeat_xyz(expr_to_repeat, 2.0, 2.0, 0)

    # Calculate expected values based on our implementation
    expected_xy = np.zeros(len(repeat_points_xyz))
    for i, point in enumerate(repeat_points_xyz):
        x = point[0]
        y = point[1]
        z = point[2] # Original z is used
        period_x = period_y = 2.0

        # Apply our repeat algorithm to x and y
        x_repeat = x - period_x * np.floor(x / period_x + 0.5)
        y_repeat = y - period_y * np.floor(y / period_y + 0.5)

        # Evaluate the original expression (X+Y) at the repeated coordinates
        expected_xy[i] = x_repeat + y_repeat

    result_repeat_xy = evaluate(expr_repeat_xy, repeat_points_xyz)
    np.testing.assert_allclose(result_repeat_xy, expected_xy, atol=1e-6)
    # Test method call for repeat_xyz
    expr_repeat_xy_method = expr_to_repeat.repeat_xyz(2.0, 2.0, 0)
    result_repeat_xy_method = evaluate(expr_repeat_xy_method, repeat_points_xyz)
    np.testing.assert_allclose(result_repeat_xy_method, expected_xy, atol=1e-6)


    # Test repeat in all dimensions (uniform repeat)
    expr_to_repeat_all = X + Y + Z
    expr_repeat_all = fpm.repeat(expr_to_repeat_all, 2.0)

    # Calculate expected values
    expected_all = np.zeros(len(repeat_points_xyz))
    for i, point in enumerate(repeat_points_xyz):
        x = point[0]
        y = point[1]
        z = point[2]
        period = 2.0

        # Apply repeat to all dimensions
        x_repeat = x - period * np.floor(x / period + 0.5)
        y_repeat = y - period * np.floor(y / period + 0.5)
        z_repeat = z - period * np.floor(z / period + 0.5)

        # Evaluate the original expression (X+Y+Z) at the repeated coordinates
        expected_all[i] = x_repeat + y_repeat + z_repeat

    result_repeat_all = evaluate(expr_repeat_all, repeat_points_xyz)
    np.testing.assert_allclose(result_repeat_all, expected_all, atol=1e-6)
    # Test method call for repeat
    expr_repeat_all_method = expr_to_repeat_all.repeat(2.0)
    result_repeat_all_method = evaluate(expr_repeat_all_method, repeat_points_xyz)
    np.testing.assert_allclose(result_repeat_all_method, expected_all, atol=1e-6)


def test_symmetry():
    """Test symmetry aliases."""
    # symmetry_x should be same as mirror_x
    expr_mirror_x = fpm.mirror_x(X)
    expr_symmetry_x = fpm.symmetry_x(X)
    np.testing.assert_allclose(evaluate(expr_mirror_x), evaluate(expr_symmetry_x))
    expr_symmetry_x_method = X.symmetry_x()
    np.testing.assert_allclose(evaluate(expr_mirror_x), evaluate(expr_symmetry_x_method))

    # symmetry_y should be same as mirror_y
    expr_mirror_y = fpm.mirror_y(Y)
    expr_symmetry_y = fpm.symmetry_y(Y)
    np.testing.assert_allclose(evaluate(expr_mirror_y), evaluate(expr_symmetry_y))
    expr_symmetry_y_method = Y.symmetry_y()
    np.testing.assert_allclose(evaluate(expr_mirror_y), evaluate(expr_symmetry_y_method))

    # symmetry_z should be same as mirror_z
    expr_mirror_z = fpm.mirror_z(Z)
    expr_symmetry_z = fpm.symmetry_z(Z)
    np.testing.assert_allclose(evaluate(expr_mirror_z), evaluate(expr_symmetry_z))
    expr_symmetry_z_method = Z.symmetry_z()
    np.testing.assert_allclose(evaluate(expr_mirror_z), evaluate(expr_symmetry_z_method))


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_mirror()
    # test_repeat()
    # test_multi_dimensional_repeat()
    # test_symmetry()
    # print("All domain manipulation tests passed!")