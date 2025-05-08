"""
Tests for curve-based shapes in fidgetpy.shape module.

This file tests the curve-based shapes by creating them, meshing them using both
fidgetpy's built-in meshing functionality and the fidget-cli tool, and then
comparing the resulting STL files.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.shape as fps
from pathlib import Path

from .test_utils import shape_dual_meshing

# Common parameters for shape tests
TEST_DEPTH = 5
TEST_SCALE = 1.0


def test_line_segment():
    """Test the line_segment shape."""
    # Create a line segment from (-1,0,0) to (1,0,0) with thickness 0.2
    line = fps.line_segment((-1, 0, 0), (1, 0, 0), 0.2)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        line, "line_segment", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Line segment meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_quadratic_bezier():
    """Test the quadratic_bezier shape."""
    # Create a quadratic Bezier curve with control points and thickness
    start = (-1, 0, 0)
    control = (0, 1, 0)
    end = (1, 0, 0)
    thickness = 0.2
    
    bezier = fps.quadratic_bezier(start, control, end, thickness)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        bezier, "quadratic_bezier", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Quadratic bezier meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_cubic_bezier():
    """Test the cubic_bezier shape."""
    # Create a cubic Bezier curve with control points and thickness
    start = (-1, 0, 0)
    control1 = (-0.5, 1, 0)
    control2 = (0.5, -1, 0)
    end = (1, 0, 0)
    thickness = 0.2
    
    bezier = fps.cubic_bezier(start, control1, control2, end, thickness)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        bezier, "cubic_bezier", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cubic bezier meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_polyline():
    """Test the polyline shape."""
    # Create a polyline with multiple points and thickness
    points = [
        (-1, 0, 0),
        (-0.5, 0.5, 0),
        (0, 0, 0),
        (0.5, -0.5, 0),
        (1, 0, 0)
    ]
    thickness = 0.2
    
    poly = fps.polyline(points, thickness)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        poly, "polyline", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Polyline meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_bezier_spline():
    """Test the bezier_spline shape."""
    # Create points for the spline
    points = [
        (-1.0, 0.0, 0.0),  # First point
        (0.0, 1.0, 0.0),   # Second point
        (1.0, 0.0, 0.0)    # Third point
    ]
    
    # Create left and right handles for each point
    left_handles = [
        (-1.0, 0.0, 0.0),       # Left handle of first point (coincides with point)
        (-0.5, 0.5, 0.0),       # Left handle of second point (influence from first point)
        (0.5, 0.5, 0.0)         # Left handle of third point (influence from second point)
    ]
    
    right_handles = [
        (-0.5, -0.5, 0.0),      # Right handle of first point (downward direction)
        (0.5, 1.5, 0.0),        # Right handle of second point (upward direction)
        (1.0, 0.0, 0.0)         # Right handle of third point (coincides with point)
    ]
    
    thickness = 0.2
    
    # Create the bezier spline
    spline = fps.bezier_spline(points, left_handles, right_handles, thickness)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        spline, "bezier_spline", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Bezier spline meshes differ: {py_stl} vs {cli_stl}"


def test_bezier_spline_variable_thickness():
    """Test the bezier_spline_variable_thickness shape."""
    # Create points for the spline
    points = [
        (-4.0, 0.0, 0.0),     # Left point
        (0.0, 0.0 ,0.0), 
        (4.0, 0.0, 0.0),      # Right point
    ]

    # Define very different thicknesses for clear visual difference
    radii = [
        0.8,     # Left end (thicker)
        1.0,
        0.2      # Right end (thinner)
    ]

    # Create custom handles to make a clear curve between points
    left_handles = [
        (-4.0, 0.0, 0.0),      # Left point (coincides with point)
        (0.0, 1.0, 0.0),
        (0.0, -2.0, 0.0)       # Left handle of right point (curves below)
    ]

    right_handles = [
        (0.0, 2.0, 0.0),       # Right handle of left point (curves above)
        (0.0, -1.0, 0.0),
        (4.0, 0.0, 0.0)        # Right point (coincides with point)
    ]
        
    # Create the bezier spline with variable thickness
    spline = fps.cubic_bezier_spline_variable_radius(points, left_handles, right_handles, radii=radii)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        spline, "bezier_spline_variable", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Variable thickness bezier spline meshes differ: {py_stl} vs {cli_stl}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])