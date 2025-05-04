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
    """Test mirroring function."""
    x_points = SAMPLE_POINTS_NP[:, 0:1]

    # Mirror X
    expr_abs_x = fpm.abs(X)  # |x|
    mirrored_expr_x = fpm.mirror(X, mx=True)
    mirrored_expr_x_m = X.mirror(mx=True)
    np.testing.assert_allclose(evaluate(expr_abs_x, x_points, [X]), evaluate(mirrored_expr_x, x_points, [X]))
    np.testing.assert_allclose(evaluate(expr_abs_x, x_points, [X]), evaluate(mirrored_expr_x_m, x_points, [X]))

    # Mirror Y
    expr_abs_y = fpm.abs(Y)  # |y|
    mirrored_expr_y = fpm.mirror(Y, my=True)
    mirrored_expr_y_m = Y.mirror(my=True)
    np.testing.assert_allclose(evaluate(expr_abs_y, x_points, [Y]), evaluate(mirrored_expr_y, x_points, [Y]))
    np.testing.assert_allclose(evaluate(expr_abs_y, x_points, [Y]), evaluate(mirrored_expr_y_m, x_points, [Y]))

    # Mirror Z
    expr_abs_z = fpm.abs(Z)  # |z|
    mirrored_expr_z = fpm.mirror(Z, mz=True)
    mirrored_expr_z_m = Z.mirror(mz=True)
    np.testing.assert_allclose(evaluate(expr_abs_z, x_points, [Z]), evaluate(mirrored_expr_z, x_points, [Z]))
    np.testing.assert_allclose(evaluate(expr_abs_z, x_points, [Z]), evaluate(mirrored_expr_z_m, x_points, [Z]))

    # Mirror XY
    expr_abs_xy = fpm.abs(X) + fpm.abs(Y) # |x| + |y|
    mirrored_expr_xy = fpm.mirror(X + Y, mx=True, my=True)
    mirrored_expr_xy_m = (X + Y).mirror(mx=True, my=True)
    # Evaluate at points where x or y might be negative
    points_xy = np.array([
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [0.5, -0.5],
    ], dtype=np.float32)
    np.testing.assert_allclose(evaluate(expr_abs_xy, points_xy, [X,Y]), evaluate(mirrored_expr_xy, points_xy, [X,Y]))
    #np.testing.assert_allclose(evaluate(expr_abs_xy, points_xy), evaluate(mirrored_expr_xy_m, points_xy))

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
    
    repeat_points_x = repeat_points[:, 0:1]
    # Test repeat just in x dimension
    expr_repeat_x = fpm.repeat(X, 2.0, 0, 0)
    
    # Calculate expected values based on our implementation
    # x - period * floor(x / period + 0.5)
    expected_x = np.zeros(len(repeat_points))
    for i, point in enumerate(repeat_points):
        x = point[0]
        period = 2.0
        expected_x[i] = x - period * np.floor(x / period + 0.5)
    
    result_repeat_x = evaluate(expr_repeat_x, repeat_points_x, [X])
    np.testing.assert_allclose(result_repeat_x, expected_x, atol=1e-6)
    # Test method call
    expr_repeat_x_method = X.repeat(2.0, 0, 0)
    result_repeat_x_method = evaluate(expr_repeat_x_method, repeat_points_x, [X])
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

    expr_repeat_y = fpm.repeat(Y, 0, 2.0, 0)

    # Calculate expected values
    expected_y = np.zeros(len(repeat_points_y))
    for i, point in enumerate(repeat_points_y):
        y = point[1]
        period = 2.0
        expected_y[i] = y - period * np.floor(y / period + 0.5)

    result_repeat_y = evaluate(expr_repeat_y, repeat_points_x, [Y])
    np.testing.assert_allclose(result_repeat_y, expected_y, atol=1e-6)
    # Test method call
    expr_repeat_y_method = Y.repeat(0, 2.0, 0)
    result_repeat_y_method = evaluate(expr_repeat_y_method, repeat_points_x, [Y])
    np.testing.assert_allclose(result_repeat_y_method, expected_y, atol=1e-6)

    # Test repeat_z
    repeat_points_z = np.array([
        [0.0],
        [ 1.0],
        [ 1.9],
        [ 2.1],
        [ 3.0],
        [ -0.5]
    ], dtype=np.float32)

    expr_repeat_z = fpm.repeat(Z, 0, 0, 2.0)

    # Calculate expected values
    expected_z = np.zeros(len(repeat_points_z))
    for i, point in enumerate(repeat_points_z):
        z = point[0]
        period = 2.0
        expected_z[i] = z - period * np.floor(z / period + 0.5)

    result_repeat_z = evaluate(expr_repeat_z, repeat_points_z, [Z])
    np.testing.assert_allclose(result_repeat_z, expected_z, atol=1e-6)
    # Test method call
    expr_repeat_z_method = Z.repeat(0, 0, 2.0)
    result_repeat_z_method = evaluate(expr_repeat_z_method, repeat_points_z, [Z])
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
    expr_repeat_xy = fpm.repeat(expr_to_repeat, 2.0, 2.0, 0)

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

    # Only X and Y are used, so specify them in variables and use only first 2 columns
    result_repeat_xy = evaluate(expr_repeat_xy, repeat_points_xyz[:, :2], [X, Y])
    np.testing.assert_allclose(result_repeat_xy, expected_xy, atol=1e-6)
    # Test method call for repeat_xyz
    expr_repeat_xy_method = expr_to_repeat.repeat(2.0, 2.0, 0)
    # Only X and Y are used here too
    result_repeat_xy_method = evaluate(expr_repeat_xy_method, repeat_points_xyz[:, :2], [X, Y])
    np.testing.assert_allclose(result_repeat_xy_method, expected_xy, atol=1e-6)


    # Test repeat in all dimensions (uniform repeat)
    expr_to_repeat_all = X + Y + Z
    expr_repeat_all = fpm.repeat(expr_to_repeat_all, 2.0, 2.0, 2.0)

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

    result_repeat_all = evaluate(expr_repeat_all, repeat_points_xyz, [X, Y, Z])
    np.testing.assert_allclose(result_repeat_all, expected_all, atol=1e-6)
    # Test method call for repeat
    expr_repeat_all_method = expr_to_repeat_all.repeat(2.0, 2.0, 2.0)
    result_repeat_all_method = evaluate(expr_repeat_all_method, repeat_points_xyz, [X, Y, Z])
    np.testing.assert_allclose(result_repeat_all_method, expected_all, atol=1e-6)


def test_symmetry():
    """Test symmetry function (alias for mirror)."""

    points = np.array([
        [0.0],
        [ 1.0],
        [ 1.9],
        [ 2.1],
        [ 3.0],
        [ -0.5]
    ], dtype=np.float32)

    points_xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.1, 0.5, 0.0],   # x past period, y and z within period
        [0.5, 2.1, 0.5],   # y past period, x and z within period
        [0.5, 0.5, 2.1],   # z past period, x and y within period
        [2.1, 2.1, 2.1],   # all past period
        [-0.5, -0.5, -0.5], # all negative within period
    ], dtype=np.float32)

    # Symmetry X should be same as mirror X
    expr_mirror_x = fpm.mirror(X, mx=True)
    expr_symmetry_x = fpm.symmetry(X, sx=True)
    expr_symmetry_x_m = X.symmetry(sx=True)
    # Both expressions only use X since we're only mirroring X
    # Use the same points array for both evaluations
    np.testing.assert_allclose(evaluate(expr_mirror_x, points, [X]), evaluate(expr_symmetry_x, points, [X]))
    # Both expressions only use X since we're only mirroring X
    # Use the same points array for both evaluations
    np.testing.assert_allclose(evaluate(expr_mirror_x, points, [X]), evaluate(expr_symmetry_x_m, points, [X]))

    # Symmetry Y should be same as mirror Y
    expr_mirror_y = fpm.mirror(Y, my=True)
    expr_symmetry_y = fpm.symmetry(Y, sy=True)
    expr_symmetry_y_m = Y.symmetry(sy=True)
    # Both expressions only use Y since we're only mirroring Y
    # Use the same points array for both evaluations
    np.testing.assert_allclose(evaluate(expr_mirror_y, points, [Y]), evaluate(expr_symmetry_y, points, [Y]))
    np.testing.assert_allclose(evaluate(expr_mirror_y, points, [Y]), evaluate(expr_symmetry_y_m, points, [Y]))

    # Symmetry Z should be same as mirror Z
    expr_mirror_z = fpm.mirror(Z, mz=True)
    expr_symmetry_z = fpm.symmetry(Z, sz=True)
    expr_symmetry_z_m = Z.symmetry(sz=True)
    # Both expressions only use Z since we're only mirroring Z
    # Use the same points array for both evaluations
    np.testing.assert_allclose(evaluate(expr_mirror_z, points, [Z]), evaluate(expr_symmetry_z, points, [Z]))
    np.testing.assert_allclose(evaluate(expr_mirror_z, points, [Z]), evaluate(expr_symmetry_z_m, points, [Z]))

    # Symmetry XYZ should be same as mirror XYZ
    expr_mirror_xyz = fpm.mirror(X + Y + Z, mx=True, my=True, mz=True)
    expr_symmetry_xyz = fpm.symmetry(X + Y + Z, sx=True, sy=True, sz=True)
    expr_symmetry_xyz_m = (X + Y + Z).symmetry(sx=True, sy=True, sz=True)
    # Expressions using all three variables
    np.testing.assert_allclose(evaluate(expr_mirror_xyz, points_xyz, [X, Y, Z]), evaluate(expr_symmetry_xyz, points_xyz, [X, Y, Z]))
    np.testing.assert_allclose(evaluate(expr_mirror_xyz, points_xyz, [X, Y, Z]), evaluate(expr_symmetry_xyz_m, points_xyz, [X, Y, Z]))


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_mirror()
    # test_repeat()
    # test_multi_dimensional_repeat()
    # test_symmetry()
    # print("All domain manipulation tests passed!")