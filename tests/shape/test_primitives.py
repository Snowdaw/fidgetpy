"""
Tests for primitive shapes in fidgetpy.shape module.

This file tests the primitive shapes by creating them, meshing them using both
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
TEST_SCALE = 1.0  # Smaller scale to ensure the shape fits within internal bounds


def test_sphere():
    """Test the sphere primitive."""
    # Create a sphere with radius 1.0
    sphere = fps.sphere(1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        sphere, "sphere", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    if not success:
        pytest.fail(f"Sphere meshes differ: {py_stl} vs {cli_stl}")
    
    


def test_box():
    """Test the box primitive."""
    # Create a box with different dimensions
    box = fps.box(1.0, 1.5, 0.75)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        box, "box", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Box meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_torus():
    """Test the torus primitive."""
    # Create a torus with major radius 1.0 and minor radius 0.25
    torus = fps.torus(1.0, 0.25)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        torus, "torus", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Torus meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_plane():
    """Test the plane primitive with a bounding box."""
    # Create a plane along with a bounding box to make it finite
    plane = fps.plane((0, 1, 0), 0)  # XZ plane
    box = fps.box(2.0, 1.0, 2.0)  # Bounding box
    bounded_plane = fp.ops.intersection(plane, box)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        bounded_plane, "bounded_plane", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Bounded plane meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_octahedron():
    """Test the octahedron primitive."""
    # Create an octahedron with size 1.0
    octahedron = fps.octahedron(1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        octahedron, "octahedron", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Octahedron meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_hexagonal_prism():
    """Test the hexagonal_prism primitive."""
    # Create a hexagonal prism with radius 0.5 and height 1.0
    hex_prism = fps.hexagonal_prism(0.5, 1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        hex_prism, "hexagonal_prism", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Hexagonal prism meshes differ: {py_stl} vs {cli_stl}"
    
    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])