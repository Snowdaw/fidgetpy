"""
Tests for the logical operations functions.
"""

import numpy as np
import pytest

import fidgetpy as fp
from fidgetpy.math import (
    logical_and, logical_or, logical_not, logical_xor, logical_if,
    python_and, python_or
)

# Common variables
X = fp.x()
Y = fp.y()
Z = fp.z()
C1 = 1.0
C0 = 0.0
SPHERE1 = fp.shape.sphere(1.0)
SPHERE2 = fp.shape.sphere(1.0).translate(2, 0, 0)

# Sample points
POINTS = np.array([
    [0.0, 0.0, 0.0],  # Inside sphere1, outside sphere2
    [1.0, 0.0, 0.0],  # On surface of sphere1, outside sphere2
    [2.0, 0.0, 0.0],  # Outside sphere1, on surface of sphere2
    [3.0, 0.0, 0.0],  # Outside both
    [0.5, 0.5, 0.0],  # Inside sphere1, outside sphere2
], dtype=np.float32)

def evaluate(expr, points=POINTS, variables=[X, Y, Z]):
    """Helper to evaluate expressions."""
    return fp.eval(expr, points, variables)

def test_logical_functions():
    """Test that logical functions match their operator equivalents."""
    # Test logical_and function
    and_op = SPHERE1 & SPHERE2
    and_func = logical_and(SPHERE1, SPHERE2)
    np.testing.assert_allclose(evaluate(and_op), evaluate(and_func))

    # Test logical_or function
    or_op = SPHERE1 | SPHERE2
    or_func = logical_or(SPHERE1, SPHERE2)
    np.testing.assert_allclose(evaluate(or_op), evaluate(or_func))

    # Test logical_not function
    not_op = ~SPHERE1
    not_func = logical_not(SPHERE1)
    np.testing.assert_allclose(evaluate(not_op), evaluate(not_func))

    # Test logical_xor function
    # XOR = (a | b) & ~(a & b)
    xor_op = (SPHERE1 | SPHERE2) & ~(SPHERE1 & SPHERE2)
    xor_func = logical_xor(SPHERE1, SPHERE2)
    np.testing.assert_allclose(evaluate(xor_op), evaluate(xor_func))


def test_python_style_logical():
    """Test that python-style logical functions match their method equivalents."""
    # Create array of test points with different values
    py_points = np.array([
        [0.0, 0.0],  # Both 0
        [1.0, 0.0],  # x=1, y=0
        [0.0, 1.0],  # x=0, y=1
        [1.0, 1.0],  # Both 1
        [0.5, 0.3],  # Fractional values
        [2.0, 3.0],  # Values > 1
    ], dtype=np.float32)

    # Test python_and function
    and_method = X.python_and(Y)
    and_func = python_and(X, Y)
    np.testing.assert_allclose(evaluate(and_method, py_points, [X, Y]), evaluate(and_func, py_points, [X, Y]))

    # Test python_or function
    or_method = X.python_or(Y)
    or_func = python_or(X, Y)
    np.testing.assert_allclose(evaluate(or_method, py_points, [X, Y]), evaluate(or_func, py_points, [X, Y]))


def test_logical_if():
    """Test the logical_if function."""
    # Condition: x > 0.5
    condition = X > 0.5
    true_val = Y  # Use Y if true
    false_val = Z # Use Z if false

    if_expr = logical_if(condition, true_val, false_val)

    # Points for testing logical_if
    if_points = np.array([
        [0.0, 10.0, 20.0], # x <= 0.5 -> should be Z (20.0)
        [0.4, 11.0, 21.0], # x <= 0.5 -> should be Z (21.0)
        [0.5, 12.0, 22.0], # x <= 0.5 -> should be Z (22.0)
        [0.6, 13.0, 23.0], # x > 0.5 -> should be Y (13.0)
        [1.0, 14.0, 24.0], # x > 0.5 -> should be Y (14.0)
    ], dtype=np.float32)

    expected = np.where(if_points[:, 0] > 0.5, if_points[:, 1], if_points[:, 2])
    result = evaluate(if_expr, if_points, [X, Y, Z])
    np.testing.assert_allclose(result, expected)

    # Test with constant values
    if_expr_const = logical_if(condition, 100.0, 200.0)
    expected_const = np.where(if_points[:, 0] > 0.5, 100.0, 200.0)
    result_const = evaluate(if_expr_const, if_points[:, [0]], [X])
    np.testing.assert_allclose(result_const, expected_const)


def test_logical_errors():
    """Test error handling for logical functions."""
    # TypeErrors for standard logical ops
    with pytest.raises(TypeError):
        logical_and(SPHERE1, "string")
    with pytest.raises(TypeError):
        logical_or("string", SPHERE2)
    with pytest.raises(TypeError):
        logical_not("string")
    with pytest.raises(TypeError):
        logical_xor(SPHERE1, "string")
    with pytest.raises(TypeError):
        logical_if("string", C1, C0) # Condition must be SDF
    with pytest.raises(TypeError):
        logical_if(X > 0, "string", C0) # Values must be SDF or numeric
    with pytest.raises(TypeError):
        logical_if(X > 0, C1, "string")

    # AttributeErrors for python_and/or
    class Dummy: pass
    dummy = Dummy()
    with pytest.raises(AttributeError, match="python_and"):
        python_and(dummy, X)
    with pytest.raises(AttributeError, match="python_or"):
        python_or(dummy, X)


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])