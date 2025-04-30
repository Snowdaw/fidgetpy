"""
Tests for cylinder-based shapes in fidgetpy.shape module.

This file tests the cylinder-based shapes by creating them, meshing them using both
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


def test_cylinder():
    """Test the cylinder shape."""
    # Create a cylinder with radius 1.0 and height 2.0
    cylinder = fps.cylinder(1.0, 2.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cylinder, "cylinder", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cylinder meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_infinite_cylinder():
    """Test the infinite_cylinder shape with a bounding box."""
    # Create an infinite cylinder with radius 1.0
    inf_cylinder = fps.infinite_cylinder(1.0)
    
    # Create a bounding box to make it finite
    box = fps.box(2.0, 2.0, 2.0)
    bounded_cylinder = fp.ops.intersection(inf_cylinder, box)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        bounded_cylinder, "bounded_infinite_cylinder", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Bounded infinite cylinder meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_capsule():
    """Test the capsule shape."""
    # Create a capsule with length 2.0 and radius 0.5
    capsule = fps.capsule(1.0, 0.5)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        capsule, "capsule", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Capsule meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_vertical_capsule():
    """Test the vertical_capsule shape."""
    # Create a vertical capsule with height 0.5 and radius 0.3
    vert_capsule = fps.vertical_capsule(0.5, 0.3)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        vert_capsule, "vertical_capsule", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Vertical capsule meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_cone():
    """Test the cone shape."""
    # Create a cone with radius 0.5 and height 1.0
    cone = fps.cone(45, 0.5)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cone, "cone", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cone meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_capped_cone():
    """Test the capped_cone shape."""
    # Create a capped cone with radius1 0.5, radius2 1.0, and height 2.0
    capped_cone = fps.capped_cone(0.5, 1.0, 2.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        capped_cone, "capped_cone", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Capped cone meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_capped_cylinder():
    """Test the capped_cylinder shape."""
    # Create a capped cylinder with radius 1.0, height 2.0, capHeight 0.3
    capped_cylinder = fps.capped_cylinder(1.0, 2.0, 0.3)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        capped_cylinder, "capped_cylinder", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Capped cylinder meshes differ: {py_stl} vs {cli_stl}"
    
    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])