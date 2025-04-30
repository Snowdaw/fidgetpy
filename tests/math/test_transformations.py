"""
Tests for transformation functions in fidgetpy.
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

def test_translate():
    """Test translation transformation."""
    # translate - note that the sign convention is reversed in our implementation
    expr = X + Y + Z
    translated_expr = fpm.translate(expr, 1.0, 2.0, 3.0)
    
    # In our implementation, translating by positive values moves points in the negative direction
    # Evaluating the translated expression at (0,0,0) is like evaluating the original at (-1,-2,-3)
    # which gives -1 + (-2) + (-3) = -6
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result = evaluate(translated_expr, point)
    np.testing.assert_allclose(result, [-6.0])
    
    # Also test as a method
    translated_expr_method = expr.translate(1.0, 2.0, 3.0)
    result_method = evaluate(translated_expr_method, point)
    np.testing.assert_allclose(result_method, [-6.0])

def test_scale():
    """Test scaling transformation."""
    # scale
    expr = X + Y + Z
    scaled_expr = fpm.scale(expr, 2.0)
    
    # In our implementation, scaling by a factor of 2 means coordinates are divided by 2
    # So evaluating the scaled expression at (1,2,3) is like evaluating the original at (1/2, 2/2, 3/2) = (0.5, 1, 1.5) -> 3
    point2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result2 = evaluate(scaled_expr, point2)
    np.testing.assert_allclose(result2, [3.0])
    
    # Also test as a method
    scaled_expr_method = expr.scale(2.0)
    result_method2 = evaluate(scaled_expr_method, point2)
    np.testing.assert_allclose(result_method2, [3.0])

def test_rotate():
    """Test rotation transformation."""
    # rotate_z (rotate around z-axis)
    # Create a point on the x-axis
    expr_x = X
    # In our implementation, rotating by positive angle seems to rotate clockwise, not counter-clockwise
    # Rotate 90 degrees (π/2) around z maps x to -y instead of y
    rotated_expr = fpm.rotate_z(expr_x, np.pi/2)
    
    # Point at (1,0,0) rotated 90° clockwise around z becomes (0,-1,0)
    # So evaluating the original x-coordinate expression should still give us close to 0
    point3 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result3 = evaluate(rotated_expr, point3)
    np.testing.assert_allclose(result3, [0.0], atol=1e-6)  # Should be close to 0
    
    # Same point, (0,1,0) -> after clockwise 90° rotation should evaluate to -1
    point4 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    result4 = evaluate(rotated_expr, point4)
    np.testing.assert_allclose(result4, [-1.0], atol=1e-6)  # Should be close to -1 with clockwise rotation

def test_axis_specific_translate():
    """Test axis-specific translation functions."""
    expr = X + Y + Z
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # translate_x
    tx_expr = fpm.translate_x(expr, 1.0)
    tx_expr_method = expr.translate_x(1.0)
    # Evaluate at (0,0,0) -> original at (-1, 0, 0) -> -1
    np.testing.assert_allclose(evaluate(tx_expr, point), [-1.0])
    np.testing.assert_allclose(evaluate(tx_expr_method, point), [-1.0])

    # translate_y
    ty_expr = fpm.translate_y(expr, 2.0)
    ty_expr_method = expr.translate_y(2.0)
    # Evaluate at (0,0,0) -> original at (0, -2, 0) -> -2
    np.testing.assert_allclose(evaluate(ty_expr, point), [-2.0])
    np.testing.assert_allclose(evaluate(ty_expr_method, point), [-2.0])

    # translate_z
    tz_expr = fpm.translate_z(expr, 3.0)
    tz_expr_method = expr.translate_z(3.0)
    # Evaluate at (0,0,0) -> original at (0, 0, -3) -> -3
    np.testing.assert_allclose(evaluate(tz_expr, point), [-3.0])
    np.testing.assert_allclose(evaluate(tz_expr_method, point), [-3.0])


def test_scale_xyz():
    """Test non-uniform scaling."""
    expr = X + Y + Z
    scaled_expr = fpm.scale_xyz(expr, 1.0, 2.0, 4.0)
    scaled_expr_method = expr.scale_xyz(1.0, 2.0, 4.0)

    # Evaluate at (1, 2, 4) -> original at (1/1, 2/2, 4/4) = (1, 1, 1) -> 3
    point = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(evaluate(scaled_expr, point), [3.0])
    np.testing.assert_allclose(evaluate(scaled_expr_method, point), [3.0])


def test_rotate_axes():
    """Test rotations around X, Y, Z axes."""
    point_x = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    point_y = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    point_z = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    angle = np.pi / 2 # 90 degrees

    # Rotate X around Y by 90 deg -> becomes Z
    rot_y_expr = fpm.rotate_y(X, angle)
    rot_y_expr_method = X.rotate_y(angle)
    # Evaluate at (0,0,1) -> original at (1,0,0) -> 1
    np.testing.assert_allclose(evaluate(rot_y_expr, point_z), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rot_y_expr_method, point_z), [1.0], atol=1e-6)

    # Rotate Y around Z by 90 deg -> becomes X
    rot_z_expr = fpm.rotate_z(Y, angle)
    rot_z_expr_method = Y.rotate_z(angle)
    # Evaluate at (1,0,0) -> original at (0,1,0) -> 1
    np.testing.assert_allclose(evaluate(rot_z_expr, point_x), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rot_z_expr_method, point_x), [1.0], atol=1e-6)

    # Rotate Z around X by 90 deg -> becomes Y
    rot_x_expr = fpm.rotate_x(Z, angle)
    rot_x_expr_method = Z.rotate_x(angle)
    # Evaluate at (0,1,0) -> original at (0,0,1) -> 1
    np.testing.assert_allclose(evaluate(rot_x_expr, point_y), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rot_x_expr_method, point_y), [1.0], atol=1e-6)


def test_remap():
    """Test coordinate remapping transformations."""
    # remap_xyz
    expr_sum = X + Y * 2.0 + Z * 3.0
    remapped_expr = fpm.remap_xyz(expr_sum, Y, Z, X)
    # Evaluate at (1, 2, 3) -> should be like evaluating original at (2, 3, 1)
    # Original at (2, 3, 1) = 2 + 3*2 + 1*3 = 2 + 6 + 3 = 11
    point5 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result5 = evaluate(remapped_expr, point5)
    np.testing.assert_allclose(result5, [11.0])

    # Also test as a method
    remapped_expr_method = expr_sum.remap_xyz(Y, Z, X)
    result_method5 = evaluate(remapped_expr_method, point5)
    np.testing.assert_allclose(result_method5, [11.0])

    # remap_affine (using a translation matrix)
    # Affine matrix for translation [1, 2, 3]
    translate_matrix = fpm.make_translation_matrix(1, 2, 3) # Use helper
    affine_expr = fpm.remap_affine(expr_sum, translate_matrix)

    # Evaluate original at (1,2,3) -> 1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
    # Evaluate transformed at (0,0,0) -> should be like original at (0+1, 0+2, 0+3) = (1, 2, 3) -> 14
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result6 = evaluate(affine_expr, point)
    np.testing.assert_allclose(result6, [14.0])

    # Also test as a method
    affine_expr_method = expr_sum.remap_affine(translate_matrix)
    result_method6 = evaluate(affine_expr_method, point)
    np.testing.assert_allclose(result_method6, [14.0])


def test_matrix_helpers():
    """Test the matrix creation and combination helpers."""
    # Translation
    t_mat = fpm.make_translation_matrix(1, 2, 3)
    expected_t = [1,0,0, 0,1,0, 0,0,1, 1,2,3]
    np.testing.assert_allclose(t_mat, expected_t)

    # Scaling
    s_mat = fpm.make_scaling_matrix(2, 3, 4)
    expected_s = [2,0,0, 0,3,0, 0,0,4, 0,0,0]
    np.testing.assert_allclose(s_mat, expected_s)

    # Rotation Z (90 deg)
    rz_mat = fpm.make_rotation_z_matrix(np.pi/2)
    expected_rz = [0,-1,0, 1,0,0, 0,0,1, 0,0,0]
    np.testing.assert_allclose(rz_mat, expected_rz, atol=1e-7)

    # Rotation Y (90 deg)
    ry_mat = fpm.make_rotation_y_matrix(np.pi/2)
    expected_ry = [0,0,1, 0,1,0, -1,0,0, 0,0,0]
    np.testing.assert_allclose(ry_mat, expected_ry, atol=1e-7)

    # Rotation X (90 deg)
    rx_mat = fpm.make_rotation_x_matrix(np.pi/2)
    expected_rx = [1,0,0, 0,0,-1, 0,1,0, 0,0,0]
    np.testing.assert_allclose(rx_mat, expected_rx, atol=1e-7)

    # Combine: Translate then Scale (M = S * T)
    combined_st = fpm.combine_matrices(s_mat, t_mat)
    # Expected: Scale matrix with translation scaled (2*1, 3*2, 4*3) = (2, 6, 12)
    expected_st = [2,0,0, 0,3,0, 0,0,4, 2,6,12]
    np.testing.assert_allclose(combined_st, expected_st)

    # Combine: Scale then Translate (M = T * S)
    combined_ts = fpm.combine_matrices(t_mat, s_mat)
    # Expected: Scale matrix with original translation (1, 2, 3)
    expected_ts = [2,0,0, 0,3,0, 0,0,4, 1,2,3]
    np.testing.assert_allclose(combined_ts, expected_ts)


def test_transformation_errors():
    """Test error handling for transformations."""
    expr = X # Simple expression
    # TypeErrors
    with pytest.raises(TypeError): fpm.translate("str", 1, 1, 1)
    with pytest.raises(TypeError): fpm.scale("str", 1)
    with pytest.raises(TypeError): fpm.scale_xyz("str", 1, 1, 1)
    with pytest.raises(TypeError): fpm.rotate_x("str", 1)
    with pytest.raises(TypeError): fpm.rotate_y("str", 1)
    with pytest.raises(TypeError): fpm.rotate_z("str", 1)
    with pytest.raises(TypeError): fpm.remap_xyz("str", X, Y, Z)
    with pytest.raises(TypeError): fpm.remap_affine("str", [1]*12)

    # ValueErrors
    with pytest.raises(ValueError, match="zero"): fpm.scale(expr, 0)
    with pytest.raises(ValueError, match="zero"): fpm.scale_xyz(expr, 1, 0, 1)
    with pytest.raises(ValueError, match="12 elements"): fpm.remap_affine(expr, [1]*11)
    with pytest.raises(ValueError, match="12 elements"): fpm.remap_affine(expr, "[1,2,3]")
    with pytest.raises(ValueError, match="zero"): fpm.make_scaling_matrix(0, 1, 1)
    with pytest.raises(ValueError, match="12 elements"): fpm.combine_matrices([1]*11, [1]*12)
    with pytest.raises(ValueError, match="12 elements"): fpm.combine_matrices([1]*12, [1]*13)


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_translate()
    # test_axis_specific_translate()
    # test_scale()
    # test_scale_xyz()
    # test_rotate() # Original simple rotate test
    # test_rotate_axes()
    # test_remap()
    # test_matrix_helpers()
    # test_transformation_errors()
    # print("All transformation tests passed!")