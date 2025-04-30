"""
Tests for specialized shapes in fidgetpy.shape module.

This file tests the specialized shapes by creating them, meshing them using both
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


def test_ellipsoid():
    """Test the ellipsoid shape."""
    # Create an ellipsoid with radii (1.0, 0.75, 0.5)
    ellipsoid = fps.ellipsoid(1.0, 0.75, 0.5)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        ellipsoid, "ellipsoid", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Ellipsoid meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_triangular_prism():
    """Test the triangular_prism shape."""
    # Create a triangular prism with side length 1.0 and height 2.0
    triangular_prism = fps.triangular_prism(1.0, 2.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        triangular_prism, "triangular_prism", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Triangular prism meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_box_frame():
    """Test the box_frame shape."""
    # Create a box frame with dimensions (1.5, 1.0, 0.75) and edge thickness 0.1
    box_frame = fps.box_frame(1.5, 1.0, 0.75, 0.1)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        box_frame, "box_frame", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Box frame meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_link():
    """Test the link shape."""
    # Create a link with length 1.0, width 0.5, height 0.25, and thickness 0.1
    link = fps.link(0.5, 0.5, 0.25, 0.1)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        link, "link", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Link meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_cut_sphere():
    """Test the cut_sphere shape."""
    # Create a cut sphere with radius 1.0, cut height 0.3
    cut_sphere = fps.cut_sphere(1.0, 0.3)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cut_sphere, "cut_sphere", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cut sphere meshes differ: {py_stl} vs {cli_stl}"
    
    


def test_cut_hollow_sphere():
    """Test the cut_hollow_sphere shape."""
    # Create a cut hollow sphere with radius 1.0, cut height 0.3, thickness 0.1
    cut_hollow_sphere = fps.cut_hollow_sphere(1.0, 0.3, 0.1)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cut_hollow_sphere, "cut_hollow_sphere", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cut hollow sphere meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_death_star():
    """Test the death_star shape."""
    # Create a death star with sphere radius 1.0, cut sphere radius 0.8, and cut distance 0.8
    death_star = fps.death_star(1.0, 0.8, 0.8)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        death_star, "death_star", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Death star meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_pyramid():
    """Test the pyramid shape."""
    # Create a pyramid with side 1.0 and height 1.5
    pyramid = fps.pyramid(0.5, 1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        pyramid, "pyramid", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Pyramid meshes differ: {py_stl} vs {cli_stl}"
    
    


    


def test_rhombus():
    """Test the rhombus shape."""
    # Create a rhombus with length and radius parameters
    rhombus = fps.rhombus(0.3, 0.3, 0.3, 0.1)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        rhombus, "rhombus", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Rhombus meshes differ: {py_stl} vs {cli_stl}"
    
    


    


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])