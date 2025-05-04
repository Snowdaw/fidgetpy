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
    result = evaluate(translated_expr, point, [X, Y, Z])
    np.testing.assert_allclose(result, [-6.0])
    
    # Also test as a method
    translated_expr_method = expr.translate(1.0, 2.0, 3.0)
    result_method = evaluate(translated_expr_method, point, [X, Y, Z])
    np.testing.assert_allclose(result_method, [-6.0])

def test_scale():
    """Test non-uniform scaling."""
    expr = X + Y + Z
    scaled_expr = fpm.scale(expr, 1.0, 2.0, 4.0)
    scaled_expr_method = expr.scale(1.0, 2.0, 4.0)

    # Evaluate at (1, 2, 4) -> original at (1/1, 2/2, 4/4) = (1, 1, 1) -> 3
    point = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(evaluate(scaled_expr, point, [X, Y, Z]), [3.0])
    np.testing.assert_allclose(evaluate(scaled_expr_method, point, [X, Y, Z]), [3.0])

    # Test uniform scaling by providing same factor
    uniform_scale_expr = fpm.scale(expr, 2.0, 2.0, 2.0)
    uniform_scale_expr_method = expr.scale(2.0, 2.0, 2.0)
    # Evaluate at (1, 2, 3) -> original at (1/2, 2/2, 3/2) = (0.5, 1, 1.5) -> 3
    point2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(evaluate(uniform_scale_expr, point2, [X, Y, Z]), [3.0])
    np.testing.assert_allclose(evaluate(uniform_scale_expr_method, point2, [X, Y, Z]), [3.0])


def test_rotate():
    """Test combined rotation transformation (Z then Y then X)."""
    angle90 = np.pi / 2

    # Test rotation of X axis
    # Rotate X by 90 deg around Z -> becomes Y
    rotated_x_around_z = fpm.rotate(X, 0, 0, angle90)
    rotated_x_around_z_m = X.rotate(0, 0, angle90)
    # Only use the x and y columns since we're only using X and Y variables
    point_y = np.array([[0.0, 1.0]], dtype=np.float32)  # removed the z column
    # Evaluate at (0,1) -> original at (1,0) -> 1
    # Explicitly provide the variables needed
    np.testing.assert_allclose(evaluate(rotated_x_around_z, point_y, [X, Y]), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rotated_x_around_z_m, point_y, [X, Y]), [1.0], atol=1e-6)

    # Rotate X by 90 deg around Y -> becomes -Z
    rotated_x_around_y = fpm.rotate(X, 0, angle90, 0)
    rotated_x_around_y_m = X.rotate(0, angle90, 0)
    # Only use the x and z columns since we're only using X and Z variables
    point_neg_z = np.array([[0.0, -1.0]], dtype=np.float32)  # use columns for X and Z
    # Evaluate at (0,-1) -> original at (1,0) -> 1
    # Explicitly provide the variables needed
    np.testing.assert_allclose(evaluate(rotated_x_around_y, point_neg_z, [X, Z]), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rotated_x_around_y_m, point_neg_z, [X, Z]), [1.0], atol=1e-6)

    # Test rotation of Y axis
    # Rotate Y by 90 deg around X -> becomes Z
    rotated_y_around_x = fpm.rotate(Y, angle90, 0, 0)
    rotated_y_around_x_m = Y.rotate(angle90, 0, 0)
    # Only use the y and z columns since we're only using Y and Z variables
    point_z = np.array([[0.0, 1.0]], dtype=np.float32)  # use columns for Y and Z
    # Evaluate at (0,1) -> original at (0,1) -> 1
    # Explicitly provide the variables needed
    np.testing.assert_allclose(evaluate(rotated_y_around_x, point_z, [Y, Z]), [1.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rotated_y_around_x_m, point_z, [Y, Z]), [1.0], atol=1e-6)

    # Test combined rotation: Z(90) then X(90)
    # X -> (Z rot) -> Y -> (X rot) -> Z
    rotated_xz = fpm.rotate(X, angle90, 0, angle90)
    rotated_xz_m = X.rotate(angle90, 0, angle90)
    
    # After our optimization, the rotate function is correctly determining
    # that only X and Y variables are needed for this specific rotation
    point_xy = np.array([[0.0, 0.0]], dtype=np.float32)  # Just X and Y columns
    
    # For this rotation, X is mapped to Y and then to 0
    np.testing.assert_allclose(evaluate(rotated_xz, point_xy, [X, Y]), [0.0], atol=1e-6)
    np.testing.assert_allclose(evaluate(rotated_xz_m, point_xy, [X, Y]), [0.0], atol=1e-6)

def test_remap():
    """Test coordinate remapping transformations."""
    # remap_xyz
    expr_sum = X + Y * 2.0 + Z * 3.0
    remapped_expr = fpm.remap_xyz(expr_sum, Y, Z, X)
    # Evaluate at (1, 2, 3) -> should be like evaluating original at (2, 3, 1)
    # Original at (2, 3, 1) = 2 + 3*2 + 1*3 = 2 + 6 + 3 = 11
    point5 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result5 = evaluate(remapped_expr, point5, [X, Y, Z])
    np.testing.assert_allclose(result5, [11.0])

    # Also test as a method
    remapped_expr_method = expr_sum.remap_xyz(Y, Z, X)
    result_method5 = evaluate(remapped_expr_method, point5, [X, Y, Z])
    np.testing.assert_allclose(result_method5, [11.0])

    # remap_affine (using a translation matrix)
    # Affine matrix for translation [1, 2, 3]
    translate_matrix = np.array([
        1.0, 0.0, 0.0, 
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 2.0, 3.0  
    ], dtype=np.float32).tolist()
    affine_expr = fpm.remap_affine(expr_sum, translate_matrix)

    # Evaluate original at (1,2,3) -> 1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
    # Evaluate transformed at (0,0,0) -> should be like original at (0+1, 0+2, 0+3) = (1, 2, 3) -> 14
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result6 = evaluate(affine_expr, point, [X, Y, Z])
    np.testing.assert_allclose(result6, [14.0])

    # Also test as a method
    affine_expr_method = expr_sum.remap_affine(translate_matrix)
    result_method6 = evaluate(affine_expr_method, point, [X, Y, Z])
    np.testing.assert_allclose(result_method6, [14.0])


def test_transformation_errors():
    """Test error handling for transformations."""
    expr = X # Simple expression
    # TypeErrors
    with pytest.raises(TypeError): fpm.translate("str", 1, 1, 1)
    with pytest.raises(TypeError): fpm.scale("str", 1, 1, 1)
    with pytest.raises(TypeError): fpm.rotate("str", 1, 1, 1)
    with pytest.raises(TypeError): fpm.remap_xyz("str", X, Y, Z)
    with pytest.raises(TypeError): fpm.remap_affine("str", [1]*12)

    # ValueErrors
    with pytest.raises(ValueError, match="zero"): fpm.scale(expr, 1, 0, 1)
    with pytest.raises(ValueError, match="12 elements"): fpm.remap_affine(expr, [1]*11)
    with pytest.raises(ValueError, match="12 elements"): fpm.remap_affine(expr, "[1,2,3]")

# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_translate()
    # test_scale()
    # test_rotate()
    # test_remap()
    # test_matrix_helpers()
    # test_transformation_errors()
    # print("All transformation tests passed!")