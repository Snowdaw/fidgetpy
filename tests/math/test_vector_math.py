"""
Tests for vector math functions in fidgetpy.
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

# Helper vectors for tests
VEC_SDF_3D = [X, Y, Z]
VEC_SDF_2D = [X, Y]
VEC_NUM_3D = [1.0, 2.0, 3.0]
VEC_NUM_2D = [3.0, 4.0]
VEC_NUM_ZERO = [0.0, 0.0, 0.0]
SCALAR_SDF = X
SCALAR_NUM = -5.0

def test_length():
    """Test vector length calculation for various types."""
    # Scalar numeric
    np.testing.assert_allclose(fpm.length(SCALAR_NUM), 5.0)
    # Scalar SDF
    expr_len_scalar_sdf = fpm.length(SCALAR_SDF)
    expected_len_scalar_sdf = np.abs(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_len_scalar_sdf), expected_len_scalar_sdf)

    # List of numbers (2D, 3D)
    np.testing.assert_allclose(fpm.length(VEC_NUM_2D), 5.0) # sqrt(3^2 + 4^2)
    np.testing.assert_allclose(fpm.length(VEC_NUM_3D), np.sqrt(14.0)) # sqrt(1^2 + 2^2 + 3^2)

    # List of SDFs (2D, 3D)
    expr_len_sdf_2d = fpm.length(VEC_SDF_2D)
    expected_len_sdf_2d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2)
    np.testing.assert_allclose(evaluate(expr_len_sdf_2d), expected_len_sdf_2d)

    expr_len_sdf_3d = fpm.length(VEC_SDF_3D)
    expected_len_sdf_3d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2)
    np.testing.assert_allclose(evaluate(expr_len_sdf_3d), expected_len_sdf_3d)

    # Potential method call (assuming SDF vector type might have .length())
    # This might fail if no such method exists via extension
    # expr_len_sdf_3d_method = VEC_SDF_3D.length() # How to represent SDF vector?
    # np.testing.assert_allclose(evaluate(expr_len_sdf_3d_method), expected_len_sdf_3d)


def test_distance():
    """Test distance calculation."""
    p1_num = [1.0, 1.0, 1.0]
    p2_num = [1.0, 5.0, 1.0] # Distance should be 4
    p3_num = [4.0, 5.0, 1.0] # Distance from p2 should be 3
    p1_sdf = [X, Y, Z]
    p2_sdf = [X+1, Y+1, Z+1]

    # Numeric lists
    np.testing.assert_allclose(fpm.distance(p1_num, p2_num), 4.0)
    np.testing.assert_allclose(fpm.distance(p2_num, p3_num), 3.0)

    # SDF lists
    expr_dist_sdf = fpm.distance(p1_sdf, p2_sdf)
    # Expected distance is sqrt((-1)^2 + (-1)^2 + (-1)^2) = sqrt(3)
    expected_dist_sdf = np.full(len(SAMPLE_POINTS_NP), np.sqrt(3.0))
    np.testing.assert_allclose(evaluate(expr_dist_sdf), expected_dist_sdf)

    # Scalar distance
    np.testing.assert_allclose(fpm.distance(5.0, 2.0), 3.0)
    expr_dist_scalar_sdf = fpm.distance(X, Y)
    expected_dist_scalar_sdf = np.abs(SAMPLE_POINTS_NP[:, 0] - SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_dist_scalar_sdf), expected_dist_scalar_sdf)


def test_dot_product():
    """Test dot product calculation."""
    # Scalar numeric
    np.testing.assert_allclose(fpm.dot(3.0, 4.0), 12.0)
    # Scalar SDF
    expr_dot_scalar_sdf = fpm.dot(X, Y)
    expected_dot_scalar_sdf = SAMPLE_POINTS_NP[:, 0] * SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr_dot_scalar_sdf), expected_dot_scalar_sdf)

    # List numeric (2D, 3D)
    np.testing.assert_allclose(fpm.dot([1.0, 2.0], [3.0, 4.0]), 3.0 + 8.0) # 11
    np.testing.assert_allclose(fpm.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 4.0 + 10.0 + 18.0) # 32

    # List SDF (3D)
    expr_dot_sdf_3d = fpm.dot(VEC_SDF_3D, VEC_SDF_3D) # dot with self
    expected_dot_sdf_3d = SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2
    np.testing.assert_allclose(evaluate(expr_dot_sdf_3d), expected_dot_sdf_3d)

    # List SDF (dot with numeric) - assumes broadcasting/scalar mult
    expr_dot_sdf_num = fpm.dot(VEC_SDF_3D, [1.0, 2.0, 3.0])
    expected_dot_sdf_num = SAMPLE_POINTS_NP[:, 0]*1.0 + SAMPLE_POINTS_NP[:, 1]*2.0 + SAMPLE_POINTS_NP[:, 2]*3.0
    np.testing.assert_allclose(evaluate(expr_dot_sdf_num), expected_dot_sdf_num)

    # Potential method call (assuming SDF vector type might have .dot())
    # expr_dot_sdf_3d_method = VEC_SDF_3D.dot(VEC_SDF_3D)
    # np.testing.assert_allclose(evaluate(expr_dot_sdf_3d_method), expected_dot_sdf_3d)


def test_dot2():
    """Test dot2 (dot product with self)."""
    # Scalar numeric
    np.testing.assert_allclose(fpm.dot2(SCALAR_NUM), 25.0)
    # Scalar SDF
    expr_dot2_scalar_sdf = fpm.dot2(SCALAR_SDF)
    expected_dot2_scalar_sdf = SAMPLE_POINTS_NP[:, 0]**2
    np.testing.assert_allclose(evaluate(expr_dot2_scalar_sdf), expected_dot2_scalar_sdf)

    # List numeric (2D, 3D)
    np.testing.assert_allclose(fpm.dot2(VEC_NUM_2D), 25.0) # 3^2 + 4^2
    np.testing.assert_allclose(fpm.dot2(VEC_NUM_3D), 14.0) # 1^2 + 2^2 + 3^2

    # List SDF (3D)
    expr_dot2_sdf_3d = fpm.dot2(VEC_SDF_3D)
    expected_dot2_sdf_3d = SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2
    np.testing.assert_allclose(evaluate(expr_dot2_sdf_3d), expected_dot2_sdf_3d)


def test_ndot():
    """Test ndot (2D only)."""
    v1 = [1.0, 2.0]
    v2 = [3.0, 4.0]
    # Expected: 1*3 - 2*4 = 3 - 8 = -5
    np.testing.assert_allclose(fpm.ndot(v1, v2), -5.0)

    # SDF list
    expr_ndot_sdf = fpm.ndot([X, Y], [X+1, Y+1])
    expected_ndot_sdf = SAMPLE_POINTS_NP[:, 0]*(SAMPLE_POINTS_NP[:, 0]+1) - SAMPLE_POINTS_NP[:, 1]*(SAMPLE_POINTS_NP[:, 1]+1)
    np.testing.assert_allclose(evaluate(expr_ndot_sdf), expected_ndot_sdf)


def test_cross():
    """Test cross product (3D only)."""
    v1 = [1.0, 0.0, 0.0] # x-axis
    v2 = [0.0, 1.0, 0.0] # y-axis
    # Expected: [0, 0, 1] (z-axis)
    np.testing.assert_allclose(fpm.cross(v1, v2), [0.0, 0.0, 1.0])

    # SDF list
    expr_cross_sdf = fpm.cross([X, Y, Z], [1.0, 0.0, 0.0]) # Cross with x-axis
    # Expected: [Y*0 - Z*0, Z*1 - X*0, X*0 - Y*1] = [0, Z, -Y]
    expected_cross_z = SAMPLE_POINTS_NP[:, 2]
    expected_cross_neg_y = -SAMPLE_POINTS_NP[:, 1]
    # result_cross = evaluate(expr_cross_sdf) # Cannot evaluate list directly
    expected_cross_x = np.zeros_like(expected_cross_z) # Expect 0 for x component

    # Evaluate each component expression from the list returned by fpm.cross
    result_list = expr_cross_sdf
    assert isinstance(result_list, list) and len(result_list) == 3
    result_x = evaluate(result_list[0])
    result_y = evaluate(result_list[1])
    result_z = evaluate(result_list[2])

    # Assert each component
    np.testing.assert_allclose(result_x, expected_cross_x, atol=1e-7)
    np.testing.assert_allclose(result_y, expected_cross_z, atol=1e-7) # y component should be Z
    np.testing.assert_allclose(result_z, expected_cross_neg_y, atol=1e-7) # z component should be -Y

    # The existing length test is still valuable
    expr_cross_len = fpm.length(fpm.cross([X, Y, Z], [1.0, 0.0, 0.0]))
    expected_cross_len = np.sqrt(expected_cross_z**2 + expected_cross_neg_y**2)
    np.testing.assert_allclose(evaluate(expr_cross_len), expected_cross_len)

    # Potential method call
    # expr_cross_sdf_method = VEC_SDF_3D.cross([1.0, 0.0, 0.0])
    # np.testing.assert_allclose(evaluate(fpm.length(expr_cross_sdf_method)), expected_cross_len)


def test_normalize():
    """Test vector normalization."""
    # Scalar numeric
    np.testing.assert_allclose(fpm.normalize(5.0), 1.0)
    np.testing.assert_allclose(fpm.normalize(-5.0), -1.0)
    np.testing.assert_allclose(fpm.normalize(0.0), 0.0) # Zero length case

    # Scalar SDF
    expr_norm_scalar_sdf = fpm.normalize(X)
    expected_norm_scalar_sdf = np.sign(SAMPLE_POINTS_NP[:, 0])
    # Adjust expected for zero case based on implementation (returns 0.0)
    expected_norm_scalar_sdf[SAMPLE_POINTS_NP[:, 0] == 0] = 0.0
    np.testing.assert_allclose(evaluate(expr_norm_scalar_sdf), expected_norm_scalar_sdf)

    # List numeric (3D)
    norm_vec_num = fpm.normalize(VEC_NUM_3D)
    len_vec_num = np.sqrt(14.0)
    np.testing.assert_allclose(norm_vec_num, [1.0/len_vec_num, 2.0/len_vec_num, 3.0/len_vec_num])
    # Zero length case numeric
    np.testing.assert_allclose(fpm.normalize(VEC_NUM_ZERO), [0.0, 0.0, 0.0])

    # List SDF (3D)
    expr_norm_sdf_3d = fpm.normalize(VEC_SDF_3D)
    # Evaluate the length first
    lengths = evaluate(fpm.length(VEC_SDF_3D))
    non_zero_mask = lengths > 1e-9
    points_nz = SAMPLE_POINTS_NP[non_zero_mask]
    lengths_nz = lengths[non_zero_mask]

    expected_norm_x = points_nz[:, 0] / lengths_nz
    expected_norm_y = points_nz[:, 1] / lengths_nz
    expected_norm_z = points_nz[:, 2] / lengths_nz

    # Similar issue to cross product: how to evaluate the normalized vector?
    # Evaluate length of normalized vector - should be 1 (or 0 for zero input)
    expr_norm_len = fpm.length(expr_norm_sdf_3d)
    expected_norm_len = np.ones(len(SAMPLE_POINTS_NP))
    # Handle zero length input point
    zero_len_mask = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2) < 1e-9
    expected_norm_len[zero_len_mask] = 0.0
    np.testing.assert_allclose(evaluate(expr_norm_len), expected_norm_len, atol=1e-6)

    # Potential method call
    # expr_norm_sdf_3d_method = VEC_SDF_3D.normalize()
    # np.testing.assert_allclose(evaluate(fpm.length(expr_norm_sdf_3d_method)), expected_norm_len, atol=1e-6)


def test_vector_errors():
    """Test error handling for vector math functions."""
    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension"): fpm.distance([1,2], [1,2,3])
    with pytest.raises(ValueError, match="dimension"): fpm.dot([1,2], [1,2,3])

    # Unsupported dimension
    with pytest.raises(ValueError, match="supports 2D or 3D"): fpm.length([1,2,3,4])
    with pytest.raises(ValueError, match="supports 2D or 3D"): fpm.distance([1,2,3,4], [1,2,3,4])
    with pytest.raises(ValueError, match="supports 2D or 3D"): fpm.dot([1,2,3,4], [1,2,3,4])
    with pytest.raises(ValueError, match="supports 2D or 3D"): fpm.dot2([1,2,3,4])
    with pytest.raises(ValueError, match="supports 2D or 3D"): fpm.normalize([1,2,3,4])

    # ndot requires 2D
    with pytest.raises(ValueError, match="ndot requires 2D"): fpm.ndot([1,2,3], [4,5,6])
    with pytest.raises(TypeError, match="ndot requires 2D"): fpm.ndot(1, 2) # Non-vectors

    # cross requires 3D
    with pytest.raises(ValueError, match="Cross product requires 3D"): fpm.cross([1,2], [3,4])
    with pytest.raises(TypeError, match="Cross product requires 3D vector inputs"): fpm.cross(1, [1,2,3])

    # Type errors
    with pytest.raises(TypeError): fpm.length("abc")
    with pytest.raises(TypeError): fpm.distance("a", "b")
    with pytest.raises(TypeError): fpm.dot("a", "b")
    with pytest.raises(TypeError): fpm.dot2("a")
    with pytest.raises(TypeError): fpm.normalize("a")


def test_length_2d_3d():
    """Test explicit parameter length functions."""
    # Test length_2d
    expr_len_2d = fpm.length_2d(X, Y)
    expected_len_2d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2)
    np.testing.assert_allclose(evaluate(expr_len_2d), expected_len_2d)
    
    # Test length_3d
    expr_len_3d = fpm.length_3d(X, Y, Z)
    expected_len_3d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2)
    np.testing.assert_allclose(evaluate(expr_len_3d), expected_len_3d)
    
    # Compare with legacy length function
    expr_len_legacy_2d = fpm.length([X, Y])
    expr_len_legacy_3d = fpm.length([X, Y, Z])
    np.testing.assert_allclose(evaluate(expr_len_2d), evaluate(expr_len_legacy_2d))
    np.testing.assert_allclose(evaluate(expr_len_3d), evaluate(expr_len_legacy_3d))

def test_distance_2d_3d():
    """Test explicit parameter distance functions."""
    # Test distance_2d
    expr_dist_2d = fpm.distance_2d(X, Y, X+1, Y+1)
    expected_dist_2d = np.sqrt(1**2 + 1**2)  # Distance from (x,y) to (x+1,y+1) is sqrt(2)
    np.testing.assert_allclose(evaluate(expr_dist_2d), np.full(len(SAMPLE_POINTS_NP), expected_dist_2d))
    
    # Test distance_3d
    expr_dist_3d = fpm.distance_3d(X, Y, Z, X+1, Y+1, Z+1)
    expected_dist_3d = np.sqrt(1**2 + 1**2 + 1**2)  # Distance from (x,y,z) to (x+1,y+1,z+1) is sqrt(3)
    np.testing.assert_allclose(evaluate(expr_dist_3d), np.full(len(SAMPLE_POINTS_NP), expected_dist_3d))
    
    # Compare with legacy distance function
    expr_dist_legacy_2d = fpm.distance([X, Y], [X+1, Y+1])
    expr_dist_legacy_3d = fpm.distance([X, Y, Z], [X+1, Y+1, Z+1])
    np.testing.assert_allclose(evaluate(expr_dist_2d), evaluate(expr_dist_legacy_2d))
    np.testing.assert_allclose(evaluate(expr_dist_3d), evaluate(expr_dist_legacy_3d))

def test_dot_2d_3d():
    """Test explicit parameter dot product functions."""
    # Test dot_2d
    expr_dot_2d = fpm.dot_2d(X, Y, X+1, Y+1)
    expected_dot_2d = SAMPLE_POINTS_NP[:, 0]*(SAMPLE_POINTS_NP[:, 0]+1) + SAMPLE_POINTS_NP[:, 1]*(SAMPLE_POINTS_NP[:, 1]+1)
    np.testing.assert_allclose(evaluate(expr_dot_2d), expected_dot_2d)
    
    # Test dot_3d
    expr_dot_3d = fpm.dot_3d(X, Y, Z, X+1, Y+1, Z+1)
    expected_dot_3d = SAMPLE_POINTS_NP[:, 0]*(SAMPLE_POINTS_NP[:, 0]+1) + SAMPLE_POINTS_NP[:, 1]*(SAMPLE_POINTS_NP[:, 1]+1) + SAMPLE_POINTS_NP[:, 2]*(SAMPLE_POINTS_NP[:, 2]+1)
    np.testing.assert_allclose(evaluate(expr_dot_3d), expected_dot_3d)
    
    # Compare with legacy dot function
    expr_dot_legacy_2d = fpm.dot([X, Y], [X+1, Y+1])
    expr_dot_legacy_3d = fpm.dot([X, Y, Z], [X+1, Y+1, Z+1])
    np.testing.assert_allclose(evaluate(expr_dot_2d), evaluate(expr_dot_legacy_2d))
    np.testing.assert_allclose(evaluate(expr_dot_3d), evaluate(expr_dot_legacy_3d))

def test_dot2_2d_3d():
    """Test explicit parameter dot2 functions (dot product with self)."""
    # Test dot2_2d
    expr_dot2_2d = fpm.dot2_2d(X, Y)
    expected_dot2_2d = SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2
    np.testing.assert_allclose(evaluate(expr_dot2_2d), expected_dot2_2d)
    
    # Test dot2_3d
    expr_dot2_3d = fpm.dot2_3d(X, Y, Z)
    expected_dot2_3d = SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2
    np.testing.assert_allclose(evaluate(expr_dot2_3d), expected_dot2_3d)
    
    # Compare with legacy dot2 function
    expr_dot2_legacy_2d = fpm.dot2([X, Y])
    expr_dot2_legacy_3d = fpm.dot2([X, Y, Z])
    np.testing.assert_allclose(evaluate(expr_dot2_2d), evaluate(expr_dot2_legacy_2d))
    np.testing.assert_allclose(evaluate(expr_dot2_3d), evaluate(expr_dot2_legacy_3d))

def test_ndot_2d():
    """Test explicit parameter ndot function (2D only)."""
    # Test ndot_2d
    expr_ndot_2d = fpm.ndot_2d(X, Y, X+1, Y+1)
    expected_ndot_2d = SAMPLE_POINTS_NP[:, 0]*(SAMPLE_POINTS_NP[:, 0]+1) - SAMPLE_POINTS_NP[:, 1]*(SAMPLE_POINTS_NP[:, 1]+1)
    np.testing.assert_allclose(evaluate(expr_ndot_2d), expected_ndot_2d)
    
    # Compare with legacy ndot function
    expr_ndot_legacy = fpm.ndot([X, Y], [X+1, Y+1])
    np.testing.assert_allclose(evaluate(expr_ndot_2d), evaluate(expr_ndot_legacy))

def test_cross_3d():
    """Test explicit parameter cross product function (3D only)."""
    # Test cross_3d
    result = fpm.cross_3d(X, Y, Z, 1, 0, 0)  # Cross with x-axis
    
    # Expected: [0, Z, -Y]
    expected_x = np.zeros_like(SAMPLE_POINTS_NP[:, 0])
    expected_y = SAMPLE_POINTS_NP[:, 2]
    expected_z = -SAMPLE_POINTS_NP[:, 1]
    
    # Get the components of the cross product
    assert isinstance(result, tuple) and len(result) == 3
    result_x = evaluate(result[0])
    result_y = evaluate(result[1])
    result_z = evaluate(result[2])
    
    # Assert each component
    np.testing.assert_allclose(result_x, expected_x, atol=1e-7)
    np.testing.assert_allclose(result_y, expected_y, atol=1e-7)
    np.testing.assert_allclose(result_z, expected_z, atol=1e-7)
    
    # Compare with legacy cross function
    legacy_result = fpm.cross([X, Y, Z], [1, 0, 0])
    legacy_x = evaluate(legacy_result[0])
    legacy_y = evaluate(legacy_result[1])
    legacy_z = evaluate(legacy_result[2])
    
    np.testing.assert_allclose(result_x, legacy_x, atol=1e-7)
    np.testing.assert_allclose(result_y, legacy_y, atol=1e-7)
    np.testing.assert_allclose(result_z, legacy_z, atol=1e-7)

def test_normalize_2d_3d():
    """Test explicit parameter normalize functions."""
    # Test normalize_2d
    result = fpm.normalize_2d(X, Y)
    
    # Calculate expected values
    lengths_2d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2)
    non_zero_mask_2d = lengths_2d > 1e-9
    
    # Get the components of the normalized vector
    assert isinstance(result, tuple) and len(result) == 2
    result_x = evaluate(result[0])
    result_y = evaluate(result[1])
    
    # For non-zero vectors, check normalization
    for i in range(len(SAMPLE_POINTS_NP)):
        if non_zero_mask_2d[i]:
            expected_x = SAMPLE_POINTS_NP[i, 0] / lengths_2d[i]
            expected_y = SAMPLE_POINTS_NP[i, 1] / lengths_2d[i]
            np.testing.assert_allclose(result_x[i], expected_x, atol=1e-6)
            np.testing.assert_allclose(result_y[i], expected_y, atol=1e-6)
        else:
            # For zero vectors, should return zero
            np.testing.assert_allclose(result_x[i], 0.0, atol=1e-6)
            np.testing.assert_allclose(result_y[i], 0.0, atol=1e-6)
    
    # Test normalize_3d
    result = fpm.normalize_3d(X, Y, Z)
    
    # Calculate expected values
    lengths_3d = np.sqrt(SAMPLE_POINTS_NP[:, 0]**2 + SAMPLE_POINTS_NP[:, 1]**2 + SAMPLE_POINTS_NP[:, 2]**2)
    non_zero_mask_3d = lengths_3d > 1e-9
    
    # Get the components of the normalized vector
    assert isinstance(result, tuple) and len(result) == 3
    result_x = evaluate(result[0])
    result_y = evaluate(result[1])
    result_z = evaluate(result[2])
    
    # For non-zero vectors, check normalization
    for i in range(len(SAMPLE_POINTS_NP)):
        if non_zero_mask_3d[i]:
            expected_x = SAMPLE_POINTS_NP[i, 0] / lengths_3d[i]
            expected_y = SAMPLE_POINTS_NP[i, 1] / lengths_3d[i]
            expected_z = SAMPLE_POINTS_NP[i, 2] / lengths_3d[i]
            np.testing.assert_allclose(result_x[i], expected_x, atol=1e-6)
            np.testing.assert_allclose(result_y[i], expected_y, atol=1e-6)
            np.testing.assert_allclose(result_z[i], expected_z, atol=1e-6)
        else:
            # For zero vectors, should return zero
            np.testing.assert_allclose(result_x[i], 0.0, atol=1e-6)
            np.testing.assert_allclose(result_y[i], 0.0, atol=1e-6)
            np.testing.assert_allclose(result_z[i], 0.0, atol=1e-6)

def test_explicit_vector_errors():
    """Test error handling for explicit parameter vector functions."""
    # Type errors
    with pytest.raises(TypeError):
        fpm.length_2d("string", Y)
    with pytest.raises(TypeError):
        fpm.length_3d(X, "string", Z)
    with pytest.raises(TypeError):
        fpm.distance_2d(X, Y, "string", 0)
    with pytest.raises(TypeError):
        fpm.dot_2d(X, "string", 0, 0)
    with pytest.raises(TypeError):
        fpm.cross_3d(X, Y, "string", 0, 0, 0)


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_length()
    # test_distance()
    # test_dot_product()
    # test_dot2()
    # test_ndot()
    # test_cross()
    # test_normalize()
    # test_vector_errors()
    # test_length_2d_3d()
    # test_distance_2d_3d()
    # test_dot_2d_3d()
    # test_dot2_2d_3d()
    # test_ndot_2d()
    # test_cross_3d()
    # test_normalize_2d_3d()
    # test_explicit_vector_errors()
    # print("All vector math tests passed!")