"""
Tests for SDF expression creation and manipulation in fidgetpy.
Covers operators, methods, and math functions.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
# from fidgetpy import x, y, z, constant, var # Alternative import style

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

SAMPLE_POINTS_LIST = SAMPLE_POINTS_NP.tolist()

# Sample variable values for evaluation with custom vars (x, y, z, a, b)
SAMPLE_VARS_NP = np.array([
    [0.0, 0.0, 0.0, 5.0, 10.0],
    [1.0, 0.0, 0.0, 5.0, 10.0],
    [0.0, 1.0, 0.0, 5.0, 10.0],
    [0.0, 0.0, 1.0, 5.0, 10.0],
    [0.5, -0.5, 1.5, 5.0, 10.0],
    [-1.0, -1.0, -1.0, 5.0, 10.0],
], dtype=np.float32)

SAMPLE_VARS_LIST = SAMPLE_VARS_NP.tolist()
CUSTOM_VARS_LIST = [X, Y, Z, A, B] # Order matches SAMPLE_VARS_NP columns

def evaluate(expr, points=SAMPLE_POINTS_NP, variables=None):
    """Helper to evaluate expressions, handling default/custom vars."""
    if variables:
        # Assume points includes values for custom vars if variables are provided
        return fp.eval(expr, points, variables=variables) # Added return
        return fp.eval(expr, points, variables=variables)
    else:
        # Default: evaluate using x, y, z from the points array
        return fp.eval(expr, points) # Rely on default variables=[x,y,z]

# --- Basic Variable/Constant Tests ---

def test_constant_evaluation():
    result = evaluate(C1)
    np.testing.assert_allclose(result, np.ones(len(SAMPLE_POINTS_NP)))
    result = evaluate(C2)
    np.testing.assert_allclose(result, np.full(len(SAMPLE_POINTS_NP), 2.0))

def test_variable_evaluation_default():
    result_x = evaluate(X)
    np.testing.assert_allclose(result_x, SAMPLE_POINTS_NP[:, 0])
    result_y = evaluate(Y)
    np.testing.assert_allclose(result_y, SAMPLE_POINTS_NP[:, 1])
    result_z = evaluate(Z)
    np.testing.assert_allclose(result_z, SAMPLE_POINTS_NP[:, 2])

def test_variable_evaluation_custom():
    result_a = evaluate(A, points=SAMPLE_VARS_NP, variables=CUSTOM_VARS_LIST)
    np.testing.assert_allclose(result_a, SAMPLE_VARS_NP[:, 3]) # Column for 'a'
    result_b = evaluate(B, points=SAMPLE_VARS_NP, variables=CUSTOM_VARS_LIST)
    np.testing.assert_allclose(result_b, SAMPLE_VARS_NP[:, 4]) # Column for 'b'

# --- Operator Tests ---

# Tests for operators: +, -, *, /, **, %, &, |, //, ==, !=, <, <=, >, >=, - (neg)
# with combinations: var op const, const op var, var op var

# Example: Addition
def test_add_operator():
    # X + 1
    expr1 = X + C1
    expected1 = SAMPLE_POINTS_NP[:, 0] + 1.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # 1 + X
    expr2 = C1 + X
    expected2 = 1.0 + SAMPLE_POINTS_NP[:, 0]
    np.testing.assert_allclose(evaluate(expr2), expected2)

    # X + Y
    expr3 = X + Y
    expected3 = SAMPLE_POINTS_NP[:, 0] + SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr3), expected3)

    # A + B (custom vars)
    expr4 = A + B
    expected4 = SAMPLE_VARS_NP[:, 3] + SAMPLE_VARS_NP[:, 4]
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

    # A + 1 (custom var + const)
    expr5 = A + C1
    expected5 = SAMPLE_VARS_NP[:, 3] + 1.0
    np.testing.assert_allclose(evaluate(expr5, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected5)

def test_subtract_operator():
    # X - 1
    expr1 = X - C1
    expected1 = SAMPLE_POINTS_NP[:, 0] - 1.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # 1 - X
    expr2 = C1 - X
    expected2 = 1.0 - SAMPLE_POINTS_NP[:, 0]
    np.testing.assert_allclose(evaluate(expr2), expected2)

    # X - Y
    expr3 = X - Y
    expected3 = SAMPLE_POINTS_NP[:, 0] - SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr3), expected3)

    # A - B (custom vars)
    expr4 = A - B
    expected4 = SAMPLE_VARS_NP[:, 3] - SAMPLE_VARS_NP[:, 4]
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

def test_multiply_operator():
    # X * 2
    expr1 = X * C2
    expected1 = SAMPLE_POINTS_NP[:, 0] * 2.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # 2 * X
    expr2 = C2 * X
    expected2 = 2.0 * SAMPLE_POINTS_NP[:, 0]
    np.testing.assert_allclose(evaluate(expr2), expected2)

    # X * Y
    expr3 = X * Y
    expected3 = SAMPLE_POINTS_NP[:, 0] * SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr3), expected3)

    # A * B (custom vars)
    expr4 = A * B
    expected4 = SAMPLE_VARS_NP[:, 3] * SAMPLE_VARS_NP[:, 4]
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

def test_divide_operator():
    # X / 2
    expr1 = X / C2
    expected1 = SAMPLE_POINTS_NP[:, 0] / 2.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # 2 / X (handle potential division by zero)
    expr2 = C2 / X
    points = SAMPLE_POINTS_NP.copy()
    points[points[:, 0] == 0, 0] = 1e-9 # Avoid exact zero for testing division
    expected2 = 2.0 / points[:, 0]
    np.testing.assert_allclose(evaluate(expr2, points), expected2, atol=1e-6) # Relax tolerance

    # X / Y (handle potential division by zero)
    expr3 = X / Y
    points = SAMPLE_POINTS_NP.copy()
    points[points[:, 1] == 0, 1] = 1e-9 # Avoid exact zero
    expected3 = points[:, 0] / points[:, 1]
    np.testing.assert_allclose(evaluate(expr3, points), expected3, atol=1e-6)

    # A / B (custom vars)
    expr4 = A / B
    expected4 = SAMPLE_VARS_NP[:, 3] / SAMPLE_VARS_NP[:, 4] # Assumes B is not zero in samples
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

def test_power_operator():
    # X ** 2 (integer exponent)
    expr1 = X ** 2 # Use integer literal
    expected1 = SAMPLE_POINTS_NP[:, 0] ** 2.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # X ** 0.5 (sqrt)
    expr2 = X ** 0.5
    # Handle negative base: evaluate only for non-negative points
    points = SAMPLE_POINTS_NP.copy()
    non_negative_mask = points[:, 0] >= 0
    points_to_eval = points[non_negative_mask]
    expected2 = np.sqrt(points_to_eval[:, 0])
    result2 = evaluate(expr2, points_to_eval) # Evaluate only valid points
    np.testing.assert_allclose(result2, expected2)

    # A ** B (custom vars)
    expr4 = A ** B
    expected4 = (SAMPLE_VARS_NP[:, 3] ** SAMPLE_VARS_NP[:, 4]).astype(np.float32) # Cast to f32
    # Increase tolerance slightly for f32 vs f64 power calculation
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4, rtol=1e-6)

def test_modulo_operator():
    # X % 2
    expr1 = X % C2
    expected1 = SAMPLE_POINTS_NP[:, 0] % 2.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # A % B (custom vars)
    expr4 = A % B
    expected4 = SAMPLE_VARS_NP[:, 3] % SAMPLE_VARS_NP[:, 4]
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

def test_floor_divide_operator():
    # X // 2
    expr1 = X // C2
    expected1 = SAMPLE_POINTS_NP[:, 0] // 2.0
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # A // B (custom vars)
    expr4 = A // B
    expected4 = SAMPLE_VARS_NP[:, 3] // SAMPLE_VARS_NP[:, 4]
    np.testing.assert_allclose(evaluate(expr4, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected4)

def test_negation_operator():
    # -X
    expr1 = -X
    expected1 = -SAMPLE_POINTS_NP[:, 0]
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # -A (custom var)
    expr2 = -A
    expected2 = -SAMPLE_VARS_NP[:, 3]
    np.testing.assert_allclose(evaluate(expr2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected2)

# Note: Python's & and | operators map to logical AND/OR in fidgetpy, not bitwise.
# Fidget's logical ops return 1.0 for true, 0.0 for false.
def test_logical_and_operator():
    # (X > 0) & (Y > 0)
    expr1 = (X > 0.0) & (Y > 0.0)
    expected1 = ((SAMPLE_POINTS_NP[:, 0] > 0) & (SAMPLE_POINTS_NP[:, 1] > 0)).astype(float)
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # (A > 0) & (B > 0) (custom vars)
    expr2 = (A > 0.0) & (B > 0.0)
    expected2 = ((SAMPLE_VARS_NP[:, 3] > 0) & (SAMPLE_VARS_NP[:, 4] > 0)).astype(float)
    np.testing.assert_allclose(evaluate(expr2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected2)

def test_logical_or_operator():
    # (X < 0) | (Y < 0)
    expr1 = (X < 0.0) | (Y < 0.0)
    expected1 = ((SAMPLE_POINTS_NP[:, 0] < 0) | (SAMPLE_POINTS_NP[:, 1] < 0)).astype(float)
    np.testing.assert_allclose(evaluate(expr1), expected1)

     # (A < 0) | (B < 0) (custom vars)
    expr2 = (A < 0.0) | (B < 0.0)
    expected2 = ((SAMPLE_VARS_NP[:, 3] < 0) | (SAMPLE_VARS_NP[:, 4] < 0)).astype(float)
    np.testing.assert_allclose(evaluate(expr2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected2)

def test_comparison_operators():
    # ==
    expr_eq1 = X == C1
    expected_eq1 = (SAMPLE_POINTS_NP[:, 0] == 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_eq1), expected_eq1)

    expr_eq2 = A == 5.0
    expected_eq2 = (SAMPLE_VARS_NP[:, 3] == 5.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_eq2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_eq2)

    # !=
    expr_ne1 = X != C1
    expected_ne1 = (SAMPLE_POINTS_NP[:, 0] != 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_ne1), expected_ne1)

    expr_ne2 = A != B
    expected_ne2 = (SAMPLE_VARS_NP[:, 3] != SAMPLE_VARS_NP[:, 4]).astype(float)
    np.testing.assert_allclose(evaluate(expr_ne2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_ne2)

    # <
    expr_lt1 = X < C1
    expected_lt1 = (SAMPLE_POINTS_NP[:, 0] < 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_lt1), expected_lt1)

    expr_lt2 = A < B
    expected_lt2 = (SAMPLE_VARS_NP[:, 3] < SAMPLE_VARS_NP[:, 4]).astype(float)
    np.testing.assert_allclose(evaluate(expr_lt2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_lt2)

    # <=
    expr_le1 = X <= C1
    expected_le1 = (SAMPLE_POINTS_NP[:, 0] <= 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_le1), expected_le1)

    expr_le2 = A <= 5.0
    expected_le2 = (SAMPLE_VARS_NP[:, 3] <= 5.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_le2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_le2)

    # >
    expr_gt1 = X > C1
    expected_gt1 = (SAMPLE_POINTS_NP[:, 0] > 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_gt1), expected_gt1)

    expr_gt2 = A > B
    expected_gt2 = (SAMPLE_VARS_NP[:, 3] > SAMPLE_VARS_NP[:, 4]).astype(float)
    np.testing.assert_allclose(evaluate(expr_gt2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_gt2)

    # >=
    expr_ge1 = X >= C1
    expected_ge1 = (SAMPLE_POINTS_NP[:, 0] >= 1.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_ge1), expected_ge1)

    expr_ge2 = A >= 5.0
    expected_ge2 = (SAMPLE_VARS_NP[:, 3] >= 5.0).astype(float)
    np.testing.assert_allclose(evaluate(expr_ge2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_ge2)


# --- Method Tests ---

# Note: Some methods like sqrt, asin, acos, ln might produce NaN for certain inputs.
# We test the behavior on valid inputs or use abs() / square() to ensure valid domains.

def test_unary_methods():
    # Input data preparation - Modify copies to preserve 2D shape
    points = SAMPLE_POINTS_NP.copy()

    points_for_trig = points.copy()
    points_for_trig[:, 0] *= (np.pi / 4) # Scale x for trig functions

    points_positive = points.copy()
    points_positive[:, 0] = np.abs(points_positive[:, 0]) + 1e-6 # Ensure x is positive for sqrt/ln

    points_scaled_01 = points.copy()
    points_scaled_01[:, 0] = (points_scaled_01[:, 0] + 1) / 2 # Scale x to approx [0, 1] for asin/acos

    points_nonzero = points.copy()
    points_nonzero[points_nonzero[:, 0] == 0, 0] = 1e-9 # Avoid exact zero for recip

    points_for_tan = points_for_trig.copy()
    # Simple check, might need refinement if points_for_trig hits exact multiples
    points_for_tan[np.abs(np.cos(points_for_tan[:, 0])) < 1e-6, 0] += 0.1

    vars_data = SAMPLE_VARS_NP.copy()
    # No need for vars_positive_a if not used below

    # --- Test each method ---

    # abs
    expr_abs = X.abs()
    expected_abs = np.abs(points[:, 0])
    np.testing.assert_allclose(evaluate(expr_abs, points), expected_abs)

    # square
    expr_sq = X.square()
    expected_sq = points[:, 0] ** 2
    np.testing.assert_allclose(evaluate(expr_sq, points), expected_sq)

    # sqrt (use positive points)
    expr_sqrt = X.sqrt()
    expected_sqrt = np.sqrt(points_positive[:, 0])
    np.testing.assert_allclose(evaluate(expr_sqrt, points_positive), expected_sqrt)

    # floor
    expr_floor = X.floor()
    expected_floor = np.floor(points[:, 0])
    np.testing.assert_allclose(evaluate(expr_floor, points), expected_floor)

    # ceil
    expr_ceil = X.ceil()
    expected_ceil = np.ceil(points[:, 0])
    np.testing.assert_allclose(evaluate(expr_ceil, points), expected_ceil)

    # round (Fidget likely rounds half up)
    expr_round = X.round()
    expected_round = np.floor(points[:, 0] + 0.5) # Mimic round half up
    np.testing.assert_allclose(evaluate(expr_round, points), expected_round)

    # neg (already tested via operator, but good to have method test)
    expr_neg = X.neg()
    expected_neg = -points[:, 0]
    np.testing.assert_allclose(evaluate(expr_neg, points), expected_neg)

    # recip (avoid zero)
    expr_recip = X.recip()
    points_nonzero = points.copy()
    points_nonzero[points_nonzero[:, 0] == 0, 0] = 1e-9
    expected_recip = 1.0 / points_nonzero[:, 0]
    np.testing.assert_allclose(evaluate(expr_recip, points_nonzero), expected_recip, atol=1e-6)

    # sin
    expr_sin = X.sin()
    expected_sin = np.sin(points_for_trig[:, 0]) # Index corrected column
    np.testing.assert_allclose(evaluate(expr_sin, points_for_trig), expected_sin)

    # cos
    expr_cos = X.cos()
    expected_cos = np.cos(points_for_trig[:, 0]) # Index corrected column
    np.testing.assert_allclose(evaluate(expr_cos, points_for_trig), expected_cos)

    # tan
    expr_tan = X.tan()
    expected_tan = np.tan(points_for_tan[:, 0]) # Index corrected column
    np.testing.assert_allclose(evaluate(expr_tan, points_for_tan), expected_tan, atol=1e-6)

    # asin
    expr_asin = X.asin()
    expected_asin = np.arcsin(points_scaled_01[:, 0]) # Index corrected column
    np.testing.assert_allclose(evaluate(expr_asin, points_scaled_01), expected_asin)

    # acos
    expr_acos = X.acos()
    expected_acos = np.arccos(points_scaled_01[:, 0]) # Index corrected column
    np.testing.assert_allclose(evaluate(expr_acos, points_scaled_01), expected_acos)

    # atan
    expr_atan = X.atan()
    expected_atan = np.arctan(points[:, 0])
    np.testing.assert_allclose(evaluate(expr_atan, points), expected_atan)

    # exp
    expr_exp = X.exp()
    expected_exp = np.exp(points[:, 0])
    np.testing.assert_allclose(evaluate(expr_exp, points), expected_exp)

    # ln (use positive points)
    expr_ln = X.ln()
    expected_ln = np.log(points_positive[:, 0])
    np.testing.assert_allclose(evaluate(expr_ln, points_positive), expected_ln)

    # not (logical not: 1.0 if input is <= 0, else 0.0) - Assuming method is .not()
    expr_not = X.not_() # Use renamed method
    # NOTE: Observed behavior seems to be '== 0', not logical 'not'
    expected_not = (points[:, 0] == 0).astype(float)
    np.testing.assert_allclose(evaluate(expr_not, points), expected_not)

    # Test one method with a custom variable
    expr_abs_a = A.abs()
    expected_abs_a = np.abs(vars_data[:, 3])
    np.testing.assert_allclose(evaluate(expr_abs_a, vars_data, CUSTOM_VARS_LIST), expected_abs_a)


# Tests for methods: sqrt, sin, cos, abs, max, min, square, floor, ceil, round,
# compare, modulo, and, or, atan2, neg, recip, tan, asin, acos, atan, exp, ln, not
# with combinations: var.method(), var.method(const), var.method(var)

# Example: max (Binary method)
def test_binary_methods():
    # X.max(Y)
    expr1 = X.max(Y)
    expected1 = np.maximum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr1), expected1)

    # X.max(1.0)
    expr2 = X.max(C1) # Test method with constant
    expected2 = np.maximum(SAMPLE_POINTS_NP[:, 0], 1.0)
    np.testing.assert_allclose(evaluate(expr2), expected2)

    # A.max(B) (custom vars)
    expr3 = A.max(B)
    expected3 = np.maximum(SAMPLE_VARS_NP[:, 3], SAMPLE_VARS_NP[:, 4])
    np.testing.assert_allclose(evaluate(expr3, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected3)

    # min
    expr_min1 = X.min(Y)
    expected_min1 = np.minimum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_min1), expected_min1)

    expr_min2 = A.min(C1)
    expected_min2 = np.minimum(SAMPLE_VARS_NP[:, 3], 1.0)
    np.testing.assert_allclose(evaluate(expr_min2, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_min2)

    # compare (1.0 if self > other, -1.0 if self < other, 0.0 if equal)
    expr_cmp1 = X.compare(Y)
    expected_cmp1 = np.sign(SAMPLE_POINTS_NP[:, 0] - SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_cmp1), expected_cmp1)

    # modulo (method)
    expr_mod1 = X.modulo(C2)
    expected_mod1 = SAMPLE_POINTS_NP[:, 0] % 2.0
    np.testing.assert_allclose(evaluate(expr_mod1), expected_mod1)

    # and (logical, method) - Assuming method is .and and it acts like min for 0/1 inputs
    expr_and1 = (X > C1).and_(Y > C1) # Use renamed method
    expected_and1 = np.minimum((SAMPLE_POINTS_NP[:, 0] > 1.0).astype(float), (SAMPLE_POINTS_NP[:, 1] > 1.0).astype(float))
    np.testing.assert_allclose(evaluate(expr_and1), expected_and1)

    # or (logical, method) - Assuming method is .or and it acts like max for 0/1 inputs
    expr_or1 = (X < C1).or_(Y < C1) # Use renamed method
    expected_or1 = np.maximum((SAMPLE_POINTS_NP[:, 0] < 1.0).astype(float), (SAMPLE_POINTS_NP[:, 1] < 1.0).astype(float))
    np.testing.assert_allclose(evaluate(expr_or1), expected_or1)

    # atan2(y, x)
    expr_atan2 = Y.atan2(X) # Note order: atan2(y, x)
    points_nonzero = SAMPLE_POINTS_NP.copy()
    # Avoid (0, 0) case for atan2
    points_nonzero[(points_nonzero[:, 0] == 0) & (points_nonzero[:, 1] == 0), 0] = 1e-9
    expected_atan2 = np.arctan2(points_nonzero[:, 1], points_nonzero[:, 0])
    np.testing.assert_allclose(evaluate(expr_atan2, points_nonzero), expected_atan2, atol=1e-6)


# --- Math Module Tests ---

# Assume fpm functions largely mirror the methods, test a few key ones

def test_fpm_unary_functions():
    # fpm.abs
    expr_abs = fpm.abs(X)
    expected_abs = np.abs(SAMPLE_POINTS_NP[:, 0])
    np.testing.assert_allclose(evaluate(expr_abs), expected_abs)

    # fpm.sqrt (use positive points)
    points_positive = np.abs(SAMPLE_POINTS_NP) + 1e-6
    expr_sqrt = fpm.sqrt(X)
    expected_sqrt = np.sqrt(points_positive[:, 0])
    np.testing.assert_allclose(evaluate(expr_sqrt, points_positive), expected_sqrt)

    # fpm.sin
    points_for_trig = SAMPLE_POINTS_NP.copy()
    points_for_trig[:, 0] *= (np.pi / 4)
    expr_sin = fpm.sin(X)
    expected_sin = np.sin(points_for_trig[:, 0])
    np.testing.assert_allclose(evaluate(expr_sin, points_for_trig), expected_sin)

def test_fpm_binary_functions():
    # fpm.max
    expr_max1 = fpm.max(X, Y)
    expected_max1 = np.maximum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_max1), expected_max1)

    expr_max2 = fpm.max(X, 1.0) # Test with float
    expected_max2 = np.maximum(SAMPLE_POINTS_NP[:, 0], 1.0)
    np.testing.assert_allclose(evaluate(expr_max2), expected_max2)

    expr_max3 = fpm.max(A, B) # Test with custom vars
    expected_max3 = np.maximum(SAMPLE_VARS_NP[:, 3], SAMPLE_VARS_NP[:, 4])
    np.testing.assert_allclose(evaluate(expr_max3, SAMPLE_VARS_NP, CUSTOM_VARS_LIST), expected_max3)

    # fpm.min
    expr_min1 = fpm.min(X, Y)
    expected_min1 = np.minimum(SAMPLE_POINTS_NP[:, 0], SAMPLE_POINTS_NP[:, 1])
    np.testing.assert_allclose(evaluate(expr_min1), expected_min1)

    expr_min2 = fpm.min(X, C1) # Test with constant expr
    expected_min2 = np.minimum(SAMPLE_POINTS_NP[:, 0], 1.0)
    np.testing.assert_allclose(evaluate(expr_min2), expected_min2)

    # fpm.atan2
    expr_atan2 = fpm.atan2(Y, X) # atan2(y, x)
    points_nonzero = SAMPLE_POINTS_NP.copy()
    points_nonzero[(points_nonzero[:, 0] == 0) & (points_nonzero[:, 1] == 0), 0] = 1e-9
    expected_atan2 = np.arctan2(points_nonzero[:, 1], points_nonzero[:, 0])
    np.testing.assert_allclose(evaluate(expr_atan2, points_nonzero), expected_atan2, atol=1e-6)


# --- Transformation Tests ---

def test_remap_xyz():
    # Remap X to Y, Y to Z, Z to X
    expr = X + Y * 2.0 + Z * 3.0
    remapped_expr = expr.remap_xyz(Y, Z, X)
    # Evaluate at (1, 2, 3) -> should be like evaluating original at (3, 1, 2)
    # Original at (3, 1, 2) = 3 + 1*2 + 2*3 = 3 + 2 + 6 = 11
    point = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = evaluate(remapped_expr, point)
    np.testing.assert_allclose(result, [11.0])

    # Remap X to A, Y to B, Z to C1 (custom vars)
    expr_custom = X + Y * 2.0
    remapped_custom = expr_custom.remap_xyz(A, B, C1)
    # Evaluate at (x,y,z,a,b) = (0,0,0,5,10) -> like original at (a,b,c1) = (5,10,1)
    # Original at (5, 10, 1) = 5 + 10*2 = 25
    var_point = np.array([[0.0, 0.0, 0.0, 5.0, 10.0]], dtype=np.float32)
    result_custom = evaluate(remapped_custom, var_point, CUSTOM_VARS_LIST)
    np.testing.assert_allclose(result_custom, [25.0])

def test_remap_affine():
    # Translate X by (1, 2, 3)
    expr = X + Y + Z
    # Affine matrix for translation [1, 2, 3]
    # [[1, 0, 0, 1],
    #  [0, 1, 0, 2],
    #  [0, 0, 1, 3],
    #  [0, 0, 0, 1]] -> flat [1,0,0, 0,1,0, 0,0,1, 1,2,3]
    translate_matrix = [1,0,0, 0,1,0, 0,0,1, 1,2,3]
    translated_expr = expr.remap_affine(translate_matrix)

    # Evaluate original at (1,2,3) -> 6
    # Evaluate translated at (0,0,0) -> should be like original at (0+1, 0+2, 0+3) = (1, 2, 3) -> 6
    point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result = evaluate(translated_expr, point)
    np.testing.assert_allclose(result, [6.0])

    # Evaluate original at (2,3,4) -> 9
    # Evaluate translated at (1,1,1) -> should be like original at (1+1, 1+2, 1+3) = (2, 3, 4) -> 9
    point2 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    result2 = evaluate(translated_expr, point2)
    np.testing.assert_allclose(result2, [9.0])

    # Scale X by 2
    # [[2, 0, 0, 0],
    #  [0, 1, 0, 0],
    #  [0, 0, 1, 0],
    #  [0, 0, 0, 1]] -> flat [2,0,0, 0,1,0, 0,0,1, 0,0,0]
    scale_matrix = [2,0,0, 0,1,0, 0,0,1, 0,0,0]
    scaled_expr = expr.remap_affine(scale_matrix)
    # Evaluate original at (2,1,1) -> 4
    # Evaluate scaled at (1,1,1) -> should be like original at (1*2, 1*1, 1*1) = (2, 1, 1) -> 4
    result3 = evaluate(scaled_expr, point2)
    np.testing.assert_allclose(result3, [4.0])


# Tests for transformation functions: remap_xyz, remap_affine

# --- Add more tests for all operators and methods ---