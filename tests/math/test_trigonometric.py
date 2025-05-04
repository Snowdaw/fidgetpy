"""
Tests for trigonometric functions in fidgetpy.
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

def test_sin_cos_tan():
    """Test sine, cosine, and tangent functions."""
    # Scale points for trig functions to avoid large values
    points_for_trig = SAMPLE_POINTS_NP.copy()
    points_for_trig[:, 0] *= (np.pi / 4)  # Scale x
    
    # sin
    expr_sin = fpm.sin(X)
    expected_sin = np.sin(points_for_trig[:, 0])
    np.testing.assert_allclose(evaluate(expr_sin, points_for_trig[:, [0]], [X]), expected_sin)
    
    # Also test as a method
    expr_sin_method = X.sin()
    np.testing.assert_allclose(evaluate(expr_sin_method, points_for_trig[:, [0]], [X]), expected_sin)
    
    # cos
    expr_cos = fpm.cos(X)
    expected_cos = np.cos(points_for_trig[:, 0])
    np.testing.assert_allclose(evaluate(expr_cos, points_for_trig[:, [0]], [X]), expected_cos)
    
    # Also test as a method
    expr_cos_method = X.cos()
    np.testing.assert_allclose(evaluate(expr_cos_method, points_for_trig[:, [0]], [X]), expected_cos)
    
    # tan (avoid points where cos is zero)
    points_for_tan = points_for_trig.copy()
    # Simple check, might need refinement if points_for_trig hits exact multiples
    points_for_tan[np.abs(np.cos(points_for_tan[:, 0])) < 1e-6, 0] += 0.1
    expr_tan = fpm.tan(X)
    expected_tan = np.tan(points_for_tan[:, 0])
    np.testing.assert_allclose(evaluate(expr_tan, points_for_tan[:, [0]], [X]), expected_tan, atol=1e-6)
    
    # Also test as a method
    expr_tan_method = X.tan()
    np.testing.assert_allclose(evaluate(expr_tan_method, points_for_tan[:, [0]], [X]), expected_tan, atol=1e-6)

def test_asin_acos():
    """Test arc sine and arc cosine functions."""
    # asin (input must be between -1 and 1)
    points_scaled_01 = SAMPLE_POINTS_NP.copy()
    points_scaled_01[:, 0] = points_scaled_01[:, 0] * 0.5  # Scale x to [-0.5, 0.5]
    expr_asin = fpm.asin(X)
    expected_asin = np.arcsin(points_scaled_01[:, 0])
    np.testing.assert_allclose(evaluate(expr_asin, points_scaled_01[:, [0]], [X]), expected_asin)
    
    # Also test as a method
    expr_asin_method = X.asin()
    np.testing.assert_allclose(evaluate(expr_asin_method, points_scaled_01[:, [0]], [X]), expected_asin)
    
    # acos (input must be between -1 and 1)
    expr_acos = fpm.acos(X)
    expected_acos = np.arccos(points_scaled_01[:, 0])
    np.testing.assert_allclose(evaluate(expr_acos, points_scaled_01[:, [0]], [X]), expected_acos)
    
    # Also test as a method
    expr_acos_method = X.acos()
    np.testing.assert_allclose(evaluate(expr_acos_method, points_scaled_01[:, [0]], [X]), expected_acos)

def test_atan_atan2():
    """Test arc tangent functions."""
    # atan
    expr_atan = fpm.atan(X)
    expected_atan = np.arctan(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_atan, SAMPLE_POINTS_NP[:, [0]], [X]), expected_atan)
    
    # Also test as a method
    expr_atan_method = X.atan()
    np.testing.assert_allclose(evaluate(expr_atan_method, SAMPLE_POINTS_NP[:, [0]], [X]), expected_atan)
    
    # atan2
    expr_atan2 = fpm.atan2(Y, X)
    points_nonzero = SAMPLE_POINTS_NP.copy()
    # Avoid (0, 0) case for atan2
    points_nonzero[(points_nonzero[:, 0] == 0) & (points_nonzero[:, 1] == 0), 0] = 1e-9
    expected_atan2 = np.arctan2(points_nonzero[:, 1], points_nonzero[:, 0])
    np.testing.assert_allclose(evaluate(expr_atan2, points_nonzero[:, [0, 1]], [X, Y]), expected_atan2, atol=1e-6)
    
    # Also test as a method
    expr_atan2_method = Y.atan2(X)
    np.testing.assert_allclose(evaluate(expr_atan2_method, points_nonzero[:, [0, 1]], [X, Y]), expected_atan2, atol=1e-6)

def test_trig_errors():
    """Test error handling for trigonometric functions."""
    # Domain errors for asin/acos (numeric)
    with pytest.raises(ValueError, match="asin domain error"):
        fpm.asin(1.1)
    with pytest.raises(ValueError, match="asin domain error"):
        fpm.asin(-1.1)
    with pytest.raises(ValueError, match="acos domain error"):
        fpm.acos(1.1)
    with pytest.raises(ValueError, match="acos domain error"):
        fpm.acos(-1.1)

    # Type errors (numeric inputs)
    with pytest.raises(TypeError): fpm.sin("a")
    with pytest.raises(TypeError): fpm.cos("a")
    with pytest.raises(TypeError): fpm.tan("a")
    with pytest.raises(TypeError): fpm.asin("a")
    with pytest.raises(TypeError): fpm.acos("a")
    with pytest.raises(TypeError): fpm.atan("a")
    with pytest.raises(TypeError): fpm.atan2("a", 1)
    with pytest.raises(TypeError): fpm.atan2(1, "a")


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_sin_cos_tan()
    # test_asin_acos()
    # test_atan_atan2()
    # test_trig_errors()
    # print("All trigonometric tests passed!")