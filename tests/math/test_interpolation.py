"""
Tests for interpolation functions in fidgetpy.
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

def test_mix_lerp():
    """Test linear interpolation functions."""
    # mix (lerp)
    xy_points = SAMPLE_POINTS_NP[:, 0:2]
    expr_mix = fpm.mix(X, Y, 0.5)
    expected_mix = 0.5 * SAMPLE_POINTS_NP[:, 0] + 0.5 * SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr_mix, xy_points, [X, Y]), expected_mix)
    
    # Also test as a method
    expr_mix_method = X.mix(Y, 0.5)
    np.testing.assert_allclose(evaluate(expr_mix_method, xy_points, [X, Y]), expected_mix)
    
    # Test with expression as t parameter
    expr_mix_expr_t = fpm.mix(X, Y, Z)
    expected_mix_expr_t = (1.0 - SAMPLE_POINTS_NP[:, 2]) * SAMPLE_POINTS_NP[:, 0] + SAMPLE_POINTS_NP[:, 2] * SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr_mix_expr_t, SAMPLE_POINTS_NP[:, 0:3], [X, Y, Z]), expected_mix_expr_t)
    
    # Test lerp (alias for mix)
    expr_lerp = fpm.lerp(X, Y, 0.5)
    np.testing.assert_allclose(evaluate(expr_lerp, xy_points, [X, Y]), expected_mix)
    # Test lerp method call
    expr_lerp_method = X.lerp(Y, 0.5)
    np.testing.assert_allclose(evaluate(expr_lerp_method, xy_points, [X, Y]), expected_mix)

    # Different t value
    expr_lerp_25 = fpm.lerp(X, Y, 0.25)
    expected_lerp_25 = 0.75 * SAMPLE_POINTS_NP[:, 0] + 0.25 * SAMPLE_POINTS_NP[:, 1]
    np.testing.assert_allclose(evaluate(expr_lerp_25, xy_points, [X, Y]), expected_lerp_25)


def test_smoothstep():
    """Test smoothstep interpolation function."""
    # Create points with x values between -0.5 and 1.5 for testing edges
    smooth_points = np.array([
        [-0.5], # Below edge0
        [0.0],  # At edge0
        [0.25], # Between edges
        [0.5],  # Between edges
        [0.75], # Between edges
        [1.0],  # At edge1
        [1.5],  # Above edge1
    ], dtype=np.float32)

    expr_smoothstep = fpm.smoothstep(0.0, 1.0, X)
    # smoothstep formula: t^2 * (3 - 2t) where t is clamped normalized value
    t_smooth = np.clip(smooth_points[:, 0], 0.0, 1.0)
    expected_smoothstep = t_smooth * t_smooth * (3.0 - 2.0 * t_smooth)
    np.testing.assert_allclose(evaluate(expr_smoothstep, smooth_points, [X]), expected_smoothstep)
    # Note: Method call (X.smoothstep) not supported due to parameter order issues

    # Test with different edge values
    expr_smoothstep_edges = fpm.smoothstep(0.25, 0.75, X)
    # Adjust the calculation for different edge values
    t_edges = np.clip((smooth_points[:, 0] - 0.25) / (0.75 - 0.25), 0.0, 1.0)
    expected_smoothstep_edges = t_edges * t_edges * (3.0 - 2.0 * t_edges)
    np.testing.assert_allclose(evaluate(expr_smoothstep_edges, smooth_points, [X]), expected_smoothstep_edges)
    # Note: Method call (X.smoothstep) not supported due to parameter order issues


def test_smootherstep():
    """Test smootherstep interpolation function (higher order polynomial)."""
    # Use the same points as smoothstep
    smooth_points = np.array([
        [-0.5], # Below edge0
        [0.0],  # At edge0
        [0.25], # Between edges
        [0.5],  # Between edges
        [0.75], # Between edges
        [1.0],  # At edge1
        [1.5],  # Above edge1
    ], dtype=np.float32)

    expr_smootherstep = fpm.smootherstep(0.0, 1.0, X)
    t_smooth = np.clip(smooth_points[:, 0], 0.0, 1.0)
    expected_smootherstep = t_smooth * t_smooth * t_smooth * (t_smooth * (t_smooth * 6.0 - 15.0) + 10.0)
    np.testing.assert_allclose(evaluate(expr_smootherstep, smooth_points, [X]), expected_smootherstep)
    # Note: Method call (X.smootherstep) not supported due to parameter order issues

    # Test with different edge values
    expr_smootherstep_edges = fpm.smootherstep(0.25, 0.75, X)
    # Adjust the calculation for different edge values
    t_edges = np.clip((smooth_points[:, 0] - 0.25) / (0.75 - 0.25), 0.0, 1.0)
    expected_smootherstep_edges = t_edges * t_edges * t_edges * (t_edges * (t_edges * 6.0 - 15.0) + 10.0)
    np.testing.assert_allclose(evaluate(expr_smootherstep_edges, smooth_points, [X]), expected_smootherstep_edges)
    # Note: Method call (X.smootherstep) not supported due to parameter order issues


def test_step():
    """Test the step function."""
    step_points = np.array([
        [0.0],  # Below edge
        [0.49], # Below edge
        [0.5],  # At edge
        [0.51], # Above edge
        [1.0],  # Above edge
    ], dtype=np.float32)

    edge = 0.5
    expr_step = fpm.step(edge, X)
    expected_step = np.where(step_points[:, 0] < edge, 0.0, 1.0)
    np.testing.assert_allclose(evaluate(expr_step, step_points, [X]), expected_step)
    # Note: Method call (X.step) not supported due to parameter order issues

    step_points = np.array([
        [0.0, 0.0],  # Below edge
        [0.49, 0.49], # Below edge
        [0.5, 0.5],  # At edge
        [0.51, 0.51], # Above edge
        [1.0, 1.0],  # Above edge
    ], dtype=np.float32)
    # Test with expression edge
    edge_expr = Y + 0.5
    expr_step_expr_edge = fpm.step(edge_expr, X)
    expected_step_expr_edge = np.where(step_points[:, 0] < (step_points[:, 1] + 0.5), 0.0, 1.0)
    np.testing.assert_allclose(evaluate(expr_step_expr_edge, step_points, [X, Y]), expected_step_expr_edge)
    # Note: Method call (X.step) not supported due to parameter order issues


def test_interpolate():
    """Test the interpolate function."""
    # Create points with x values between -0.5 and 1.5 for testing edges
    interp_points = np.array([
        [-0.5], # Below edge0
        [0.0],  # At edge0
        [0.5],  # Between edges
        [1.0],  # At edge1
        [1.5],  # Above edge1
    ], dtype=np.float32)
    
    expr_interp = fpm.interpolate(X, 0.0, 1.0)
    # Expected: clamped (x - edge0) / (edge1 - edge0)
    expected_interp = np.clip((interp_points[:, 0] - 0.0) / (1.0 - 0.0), 0.0, 1.0)
    np.testing.assert_allclose(evaluate(expr_interp, interp_points, [X]), expected_interp)
    
    # Test with different edge values
    expr_interp_edges = fpm.interpolate(X, -0.5, 0.5)
    # Expected: clamped (x - edge0) / (edge1 - edge0)
    expected_interp_edges = np.clip((interp_points[:, 0] - (-0.5)) / (0.5 - (-0.5)), 0.0, 1.0)
    np.testing.assert_allclose(evaluate(expr_interp_edges, interp_points, [X]), expected_interp_edges)
    
    # Test with expression edges - only test non-zero denominators
    # Create a filtered set of points where z != y
    filtered_points = []
    filtered_expected = []
    for i, point in enumerate(SAMPLE_POINTS_NP):
        x, y, z = point
        if z != y:
            filtered_points.append(point)
            filtered_expected.append(np.clip((x - y) / (z - y), 0.0, 1.0))
    
    if filtered_points:
        filtered_points_np = np.array(filtered_points, dtype=np.float32)
        filtered_expected_np = np.array(filtered_expected)
        expr_interp_expr_edges = fpm.interpolate(X, Y, Z)
        np.testing.assert_allclose(
            evaluate(expr_interp_expr_edges, filtered_points_np, [X, Y, Z]),
            filtered_expected_np,
            atol=1e-6
        )


def test_threshold():
    """Test the threshold function."""
    # Create points with x values between -0.5 and 1.5 for testing edges
    threshold_points = np.array([
        [-0.5], # Below threshold
        [0.0],  # At threshold
        [0.25], # Above threshold
        [0.5],  # Above threshold
        [1.0],  # Above threshold
    ], dtype=np.float32)
    
    # Test with default low_value=0.0, high_value=1.0
    expr_threshold = fpm.threshold(X, 0.0)
    expected_threshold = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(evaluate(expr_threshold, threshold_points, [X]), expected_threshold)
    
    # Test with custom low_value and high_value
    expr_threshold_custom = fpm.threshold(X, 0.0, -1.0, 2.0)
    expected_threshold_custom = np.array([-1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(evaluate(expr_threshold_custom, threshold_points, [X]), expected_threshold_custom)
    
    # Test with method call
    expr_threshold_method = X.threshold(0.0, -1.0, 2.0)
    np.testing.assert_allclose(evaluate(expr_threshold_method, threshold_points, [X]), expected_threshold_custom)


def test_pulse():
    """Test the pulse function."""
    # Create points with x values between -0.5 and 1.5 for testing edges
    pulse_points = np.array([
        [-0.5], # Below edge0
        [0.0],  # At edge0
        [0.25], # Between edges
        [0.5],  # At edge1
        [1.0],  # Above edge1
    ], dtype=np.float32)
    
    # Test pulse function (rectangular pulse)
    expr_pulse = fpm.pulse(0.0, 0.5, X)
    # Expected: 1 when edge0 <= x < edge1, 0 otherwise
    expected_pulse = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(evaluate(expr_pulse, pulse_points, [X]), expected_pulse)
    
    # Test with different edge values
    expr_pulse_edges = fpm.pulse(-0.5, 0.0, X)
    expected_pulse_edges = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(evaluate(expr_pulse_edges, pulse_points, [X]), expected_pulse_edges)


def test_new_interpolation_errors():
    """Test error handling for new interpolation functions."""
    # Type errors
    with pytest.raises(TypeError):
        fpm.interpolate("string", 0, 1)
    with pytest.raises(TypeError):
        fpm.threshold("string", 0)
    with pytest.raises(TypeError):
        fpm.pulse(0, 1, "string")


# Run all tests if executed directly
if __name__ == "__main__":
    # Better to run using 'pytest' command
    pytest.main([__file__])
    # test_mix_lerp()
    # test_smoothstep()
    # test_smootherstep()
    # test_step()
    # test_interpolate()
    # test_threshold()
    # test_pulse()
    # test_new_interpolation_errors()
    # print("All interpolation tests passed!")