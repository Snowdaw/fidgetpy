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

def test_eval_numpy_input_default_vars():
    """Test fp.eval with numpy array input and default x,y,z variables."""
    expr = X + Y + Z
    expected = POINTS_NP[:, 0] + POINTS_NP[:, 1] + POINTS_NP[:, 2]
    result = fp.eval(expr, POINTS_NP)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)

def test_eval_list_input_default_vars():
    """Test fp.eval with list of lists input and default x,y,z variables."""
    expr = X * Y
    expected = POINTS_NP[:, 0] * POINTS_NP[:, 1]
    result = fp.eval(expr, POINTS_LIST)
    assert isinstance(result, list) # Expect list output for list input
    np.testing.assert_allclose(result, expected)

def test_eval_numpy_input_custom_vars():
    """Test fp.eval with numpy array input and custom variables."""
    expr = X + A # x + a
    expected = VARS_NP[:, 0] + VARS_NP[:, 3]
    result = fp.eval(expr, VARS_NP, variables=CUSTOM_VARS)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)

def test_eval_list_input_custom_vars():
    """Test fp.eval with list of lists input and custom variables."""
    expr = Y - A # y - a
    expected = VARS_NP[:, 1] - VARS_NP[:, 3]
    result = fp.eval(expr, VARS_LIST, variables=CUSTOM_VARS)
    assert isinstance(result, list) # Expect list output for list input
    np.testing.assert_allclose(result, expected)

def test_eval_incorrect_variable_count_numpy():
    """Test error handling when numpy columns don't match variables."""
    expr = X + A
    with pytest.raises(ValueError, match="Number of columns"):
        # POINTS_NP only has 3 columns, but CUSTOM_VARS expects 4
        fp.eval(expr, POINTS_NP, variables=CUSTOM_VARS)

def test_eval_incorrect_variable_count_list():
    """Test error handling when list elements don't match variables."""
    expr = X + A
    with pytest.raises(ValueError, match="Number of elements"):
        # Inner lists of POINTS_LIST only have 3 elements, CUSTOM_VARS expects 4
        fp.eval(expr, POINTS_LIST, variables=CUSTOM_VARS)

def test_eval_non_variable_in_list():
    """Test error handling when a non-variable expression is in variables list."""
    expr = X + Y
    bad_vars = [X, Y + Z] # Y + Z is not a simple variable
    with pytest.raises(TypeError, match="must be either variables .* or direct variable expressions"):
        fp.eval(expr, POINTS_NP, variables=bad_vars)

def test_eval_backend_selection():
    """Test backend selection ('jit', 'vm')."""
    expr = X + Y
    points = POINTS_NP

    try:
        result_jit = fp.eval(expr, points, backend='jit')
    except fp.JitNotAvailableError:
        pytest.skip("JIT backend not available")

    result_vm = fp.eval(expr, points, backend='vm')

    assert isinstance(result_jit, np.ndarray)
    assert isinstance(result_vm, np.ndarray)
    np.testing.assert_allclose(result_jit, result_vm)

def test_eval_empty_input():
    """Test edge cases (empty input lists/arrays)."""
    expr = X + Y
    empty_points_np = np.array([], dtype=np.float32).reshape(0, 3)
    empty_points_list = []

    result_np = fp.eval(expr, empty_points_np)
    result_list = fp.eval(expr, empty_points_list)

    assert isinstance(result_np, np.ndarray)
    assert isinstance(result_list, list)
    assert result_np.size == 0
    assert len(result_list) == 0