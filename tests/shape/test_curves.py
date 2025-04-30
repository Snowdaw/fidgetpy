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
    
    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])