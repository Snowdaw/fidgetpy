"""
Tests for rounded shapes in fidgetpy.shape module.

This file tests the rounded shapes by creating them, meshing them using both
fidgetpy's built-in meshing functionality and the fidget-cli tool, and then
comparing the resulting STL files.
"""

import pytest
import numpy as np
import os
import fidgetpy as fp
import fidgetpy.shape as fps
from pathlib import Path

from .test_utils import shape_dual_meshing

# Common parameters for shape tests
TEST_DEPTH = 5
TEST_SCALE = 1.0


def test_rounded_box():
    """Test the rounded_box shape."""
    # Create a rounded box with dimensions 1.5, 1.0, 0.75 and corner radius 0.2
    rounded_box = fps.rounded_box(1.5, 1.0, 0.75, 0.2)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        rounded_box, "rounded_box", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Rounded box meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_rounded_cylinder():
    """Test the rounded_cylinder shape."""
    # Create a rounded cylinder with radius 1.0, height 2.0, and corner radius 0.2
    rounded_cylinder = fps.rounded_cylinder(1.0, 2.0, 0.2)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        rounded_cylinder, "rounded_cylinder", depth=TEST_DEPTH, scale=TEST_SCALE, save_output_if_no_cleanup=True
    )

    # Check that both meshes were created and are similar
    assert success, f"Rounded cylinder meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_round_cone():
    """Test the round_cone shape."""
    # Create a round cone with radius 0.2, radius2 1.0, height 2.0
    round_cone = fps.round_cone(0.2, 0.5, 1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        round_cone, "round_cone", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Round cone meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_capped_torus():
    """Test the capped_torus shape."""
    # Create a capped torus with main radius 1.0, tube radius 0.3, and cap radius 0.1
    capped_torus = fps.capped_torus(0.5, 0.3, 0.1)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        capped_torus, "capped_torus", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Capped torus meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_solid_angle():
    """Test the solid_angle shape."""
    # Create a solid angle with angle 45 degrees (π/4 radians) and radius 1.0
    solid_angle = fps.solid_angle(45, 1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        solid_angle, "solid_angle", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Solid angle meshes differ: {py_stl} vs {cli_stl}"
    
    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])