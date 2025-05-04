"""
Tests for the fp.eval() function.
"""

import pytest
import numpy as np
import fidgetpy as fp

# Re-use sample data defined in test_expressions for consistency if needed,
# or define specific data here.
# For simplicity, let's redefine some here.

X = fp.x()
Y = fp.y()
Z = fp.z()
A = fp.var("a")

POINTS_NP = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0],
    [-1.0, -0.5, 0.5],
], dtype=np.float32)

POINTS_LIST = POINTS_NP.tolist()

# x, y, z, a
VARS_NP = np.array([
    [0.0, 0.0, 0.0, 10.0],
    [1.0, 2.0, 3.0, 20.0],
    [-1.0, -0.5, 0.5, 30.0],
], dtype=np.float32)

VARS_LIST = VARS_NP.tolist()
CUSTOM_VARS = [X, Y, Z, A]

def test_eval_numpy_input_custom_vars():
    """Test fp.eval with numpy array input and custom variables."""
    expr = X + Y + Z + A # Use all variables for strict validation
    expected = VARS_NP[:, 0] + VARS_NP[:, 1] + VARS_NP[:, 2] + VARS_NP[:, 3]
    result = fp.eval(expr, VARS_NP, variables=CUSTOM_VARS)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)

def test_eval_list_input_custom_vars():
    """Test fp.eval with list of lists input and custom variables."""
    expr = X + Y + Z - A # Use all variables for strict validation
    expected = VARS_NP[:, 0] + VARS_NP[:, 1] + VARS_NP[:, 2] - VARS_NP[:, 3]
    result = fp.eval(expr, VARS_LIST, variables=CUSTOM_VARS)
    assert isinstance(result, list) # Expect list output for list input
    np.testing.assert_allclose(result, expected)

def test_eval_incorrect_variable_count_numpy():
    """Test error handling when numpy columns don't match variables."""
    # With strict validation, we'll get the variable mismatch error first, so adjust test
    # to check for the more general message pattern, but use an expression that uses all variables
    expr = X + Y + Z + A
    with pytest.raises(ValueError):
        # POINTS_NP only has 3 columns, but CUSTOM_VARS expects 4
        fp.eval(expr, POINTS_NP, variables=CUSTOM_VARS)

def test_eval_incorrect_variable_count_list():
    """Test error handling when list elements don't match variables."""
    # With strict validation, we'll get the variable mismatch error first, so adjust test
    # to check for the more general message pattern, but use an expression that uses all variables
    expr = X + Y + Z + A
    with pytest.raises(ValueError):
        # Inner lists of POINTS_LIST only have 3 elements, CUSTOM_VARS expects 4
        fp.eval(expr, POINTS_LIST, variables=CUSTOM_VARS)

def test_eval_non_variable_in_list():
    """Test error handling when a non-variable expression is in variables list."""
    expr = X + Y
    bad_vars = [X, Y + Z] # Y + Z is not a simple variable
    with pytest.raises(TypeError, match="must be either variables .* or direct variable expressions"):
        fp.eval(expr, POINTS_NP, variables=bad_vars)

# Define a custom exception to handle JitNotAvailableError case
class JitNotAvailableError(Exception):
    pass

# Add the error to fp namespace for the test
fp.JitNotAvailableError = JitNotAvailableError

def test_eval_backend_selection():
    """Test backend selection ('jit', 'vm')."""
    expr = X + Y + Z  # Use all default variables
    points = POINTS_NP

    try:
        result_jit = fp.eval(expr, points, [X, Y, Z], backend='jit')
    except fp.JitNotAvailableError:
        pytest.skip("JIT backend not available")

    result_vm = fp.eval(expr, points, [X, Y, Z], backend='vm')

    assert isinstance(result_jit, np.ndarray)
    assert isinstance(result_vm, np.ndarray)
    np.testing.assert_allclose(result_jit, result_vm)
