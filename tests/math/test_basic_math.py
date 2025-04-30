"""
Tests for basic math functions in fidgetpy.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm

# Define common variables and constants for tests
X = fp.x()
Y = fp.y()
Z = fp.z()
C1 = 1.0
C2 = 2.0
C_NEG = -1.0
A = fp.var("a")
B = fp.var("b")

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

def test_min_max():
    """Test min and max functions."""
    # min
    expr_min = fpm.min(X, Y)
    expected_min = np.minimum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_min), expected_min)
    
    # Also test as a method
    expr_min_method = X.min(Y)
    np.testing.assert_allclose(evaluate(expr_min_method), expected_min)
    
    # max
    expr_max = fpm.max(X, Y)
    expected_max = np.maximum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_max), expected_max)
    
    # Also test as a method
    expr_max_method = X.max(Y)
    np.testing.assert_allclose(evaluate(expr_max_method), expected_max)

def test_clamp_abs_sign():
    """Test clamp, abs, and sign functions."""
    # clamp
    expr_clamp = fpm.clamp(X, C_NEG, C1)
    expected_clamp = np.clip(SAMPLE_POINTS_NP[:, 0], -1.0, 1.0)
    np.testing.assert_allclose(evaluate(expr_clamp), expected_clamp)
    
    # abs
    expr_abs = fpm.abs(X)
    expected_abs = np.abs(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_abs), expected_abs)

    # Also test as a method
    expr_abs_method = X.abs()
    np.testing.assert_allclose(evaluate(expr_abs_method), expected_abs)

    # sign
    expr_sign = fpm.sign(X)
    # Note: fpm.sign for numeric types returns -1.0, 0.0, 1.0.
    # The SDF implementation might differ slightly near zero.
    expected_sign = np.sign(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_sign), expected_sign, atol=1e-6)

    # Also test as a method
    expr_sign_method = X.sign()
    np.testing.assert_allclose(evaluate(expr_sign_method), expected_sign, atol=1e-6)

    # Test clamp method
    expr_clamp_method = X.clamp(C_NEG, C1)
    np.testing.assert_allclose(evaluate(expr_clamp_method), expected_clamp)


def test_floor_ceil_round():
    """Test floor, ceil, and round functions."""
    # floor
    expr_floor = fpm.floor(X)
    expected_floor = np.floor(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_floor), expected_floor)
    expr_floor_method = X.floor()
    np.testing.assert_allclose(evaluate(expr_floor_method), expected_floor)

    # ceil
    expr_ceil = fpm.ceil(X)
    expected_ceil = np.ceil(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_ceil), expected_ceil)
    expr_ceil_method = X.ceil()
    np.testing.assert_allclose(evaluate(expr_ceil_method), expected_ceil)

    # round
    expr_round = fpm.round(X)
    # Note: Python/Numpy round uses round-half-to-even.
    # The SDF implementation appears to round halves away from zero (e.g., 0.5 -> 1.0, -0.5 -> -1.0).
    # We need to calculate the expected result accordingly.
    raw_values = SAMPLE_POINTS_NP[:, 0]
    # Implement round half away from zero: sign(x) * floor(abs(x) + 0.5)
    expected_round = np.sign(raw_values) * np.floor(np.abs(raw_values) + 0.5)
    # Handle the case where raw_value is exactly 0
    expected_round[raw_values == 0] = 0.0
    np.testing.assert_allclose(evaluate(expr_round), expected_round)
    expr_round_method = X.round()
    np.testing.assert_allclose(evaluate(expr_round_method), expected_round)


def test_fract_mod():
    """Test fractional part and modulo functions."""
    # fract
    expr_fract = fpm.fract(X)
    expected_fract = SAMPLE_POINTS_NP[:, 0] - np.floor(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_fract), expected_fract)
    expr_fract_method = X.fract()
    np.testing.assert_allclose(evaluate(expr_fract_method), expected_fract)

    # mod
    expr_mod = fpm.mod(X, C2)
    expected_mod = np.mod(SAMPLE_POINTS_NP[:, 0], 2.0) # Use np.mod for consistency
    np.testing.assert_allclose(evaluate(expr_mod), expected_mod)
    expr_mod_method = X.modulo(C2) # Method name is 'modulo'
    np.testing.assert_allclose(evaluate(expr_mod_method), expected_mod)


def test_pow_sqrt():
    """Test power and square root functions."""
    # pow
    # Test with positive base
    points_pos_base = SAMPLE_POINTS_NP.copy()
    points_pos_base[:, 0] = np.abs(points_pos_base[:, 0]) + 0.1 # Ensure positive
    expr_pow = fpm.pow(X, C2)
    expected_pow = points_pos_base[:, 0] ** 2.0
    np.testing.assert_allclose(evaluate(expr_pow, points_pos_base), expected_pow)
    expr_pow_method = X.pow(C2)
    np.testing.assert_allclose(evaluate(expr_pow_method, points_pos_base), expected_pow)
    expr_pow_op = X ** C2
    np.testing.assert_allclose(evaluate(expr_pow_op, points_pos_base), expected_pow)

    # Test pow with negative base, integer exponent
    expr_pow_neg = fpm.pow(C_NEG, C2) # (-1)^2
    np.testing.assert_allclose(evaluate(expr_pow_neg), 1.0)

    # sqrt (use positive points)
    points_positive = SAMPLE_POINTS_NP.copy()
    points_positive[:, 0] = np.abs(points_positive[:, 0]) # Ensure non-negative
    expr_sqrt = fpm.sqrt(X)
    expected_sqrt = np.sqrt(points_positive[:, 0])
    np.testing.assert_allclose(evaluate(expr_sqrt, points_positive), expected_sqrt)
    expr_sqrt_method = X.sqrt()
    np.testing.assert_allclose(evaluate(expr_sqrt_method, points_positive), expected_sqrt)


def test_exp_ln():
    """Test exponential and natural logarithm functions."""
    # exp
    expr_exp = fpm.exp(X)
    expected_exp = np.exp(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_exp), expected_exp)
    expr_exp_method = X.exp()
    np.testing.assert_allclose(evaluate(expr_exp_method), expected_exp)

    # ln (use positive points)
    points_positive = SAMPLE_POINTS_NP.copy()
    points_positive[:, 0] = np.abs(points_positive[:, 0]) + 1e-6 # Ensure strictly positive
    expr_ln = fpm.ln(X)
    expected_ln = np.log(points_positive[:, 0])
    np.testing.assert_allclose(evaluate(expr_ln, points_positive), expected_ln)
    expr_ln_method = X.ln()
    np.testing.assert_allclose(evaluate(expr_ln_method, points_positive), expected_ln)


def test_error_handling():
    """Test error handling for domain errors and type errors."""
    # sqrt domain error (numeric)
    with pytest.raises(ValueError, match="sqrt domain error"):
        fpm.sqrt(-1.0)

    # ln domain error (numeric)
    with pytest.raises(ValueError, match="ln domain error"):
        fpm.ln(0.0)
    with pytest.raises(ValueError, match="ln domain error"):
        fpm.ln(-1.0)

    # pow domain error (numeric)
    with pytest.raises(ValueError, match="pow domain error"):
        fpm.pow(-1.0, 0.5) # Negative base, non-integer exponent

    # Type errors (numeric inputs)
    with pytest.raises(TypeError):
        fpm.min("a", 1)
    with pytest.raises(TypeError):
        fpm.max(1, "b")
    with pytest.raises(TypeError):
        fpm.clamp("a", 0, 1)
    with pytest.raises(TypeError):
        fpm.abs("a")
    with pytest.raises(TypeError):
        fpm.sign("a")
    with pytest.raises(TypeError):
        fpm.floor("a")
    with pytest.raises(TypeError):
        fpm.ceil("a")
    with pytest.raises(TypeError):
        fpm.round("a")
    with pytest.raises(TypeError):
        fpm.fract("a")
    with pytest.raises(TypeError):
        fpm.mod("a", 1)
    with pytest.raises(TypeError):
        fpm.pow("a", 1)
    with pytest.raises(TypeError):
        fpm.sqrt("a")
    with pytest.raises(TypeError):
        fpm.exp("a")
    with pytest.raises(TypeError):
        fpm.ln("a")


# Run all tests if executed directly
if __name__ == "__main__":
    # Note: Running directly won't capture pytest features like raises
    # It's better to run using 'pytest' command
    pytest.main([__file__])
    # test_min_max()
    # test_clamp_abs_sign()
    # test_floor_ceil_round()
    # test_fract_mod()
    # test_pow_sqrt()
    # test_exp_ln()
    # print("Basic math tests passed (run with pytest for full coverage).")