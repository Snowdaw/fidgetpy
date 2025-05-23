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
import fidgetpy.math as fpm
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
    """Test the box primitive (alias for box_exact)."""
    # Create a box with different dimensions
    box = fps.box_exact(1.0, 1.5, 0.75)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        box, "box", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Box meshes differ: {py_stl} vs {cli_stl}"


def test_box_exact():
    """Test the box_exact primitive."""
    # Create a box with different dimensions
    box = fps.box_exact(1.0, 1.5, 0.75)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        box, "box_exact", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Box exact meshes differ: {py_stl} vs {cli_stl}"


def test_box_mitered():
    """Test the box_mitered primitive."""
    # Create a mitered box with different dimensions
    # For mitered box, we need to ensure the mesh resolution is high enough
    box = fps.box_mitered(1.0, 1.5, 0.75)
    
    # Test meshing with both methods and compare results
    # Use a higher depth for better meshing of the mitered box
    success, py_stl, cli_stl = shape_dual_meshing(
        box, "box_mitered", depth=TEST_DEPTH+1, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Box mitered meshes differ: {py_stl} vs {cli_stl}"


def test_rectangle():
    """Test the rectangle primitive."""
    # Create a rectangle
    rect = fps.rectangle(1.5, 0.75)
    
    # Create a 3D shape by extruding the rectangle
    # We need to use a thicker extrusion to ensure proper meshing
    x, y, z = fp.x(), fp.y(), fp.z()
    extruded_rect = fpm.max(rect, fpm.abs(z) - 0.2)
    
    # Test meshing with both methods and compare results
    # Use a higher depth for better meshing of the thin shape
    success, py_stl, cli_stl = shape_dual_meshing(
        extruded_rect, "rectangle", depth=TEST_DEPTH+1, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Rectangle meshes differ: {py_stl} vs {cli_stl}"
    
    


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
    box = fps.box_exact(2.0, 1.0, 2.0)  # Bounding box
    bounded_plane = fp.ops.intersection(plane, box)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        bounded_plane, "bounded_plane_intersection", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Bounded plane (intersection) meshes differ: {py_stl} vs {cli_stl}"


def test_bounded_plane():
    """Test the bounded_plane primitive."""
    # Create a bounded plane using the new function
    bounded_plane = fps.bounded_plane((0, 1, 0), 0, (2.0, 1.0, 2.0))
    
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
    
    


def test_circle():
    """Test the circle primitive."""
    # Create a circle with radius 1.0
    # Note: circle function expects (radius, center) where center is a tuple
    circle = fps.circle(1.0, (0.0, 0.0))
    
    # Create a 3D shape by extruding the circle
    x, y, z = fp.x(), fp.y(), fp.z()
    extruded_circle = fpm.max(circle, fpm.abs(z) - 0.2)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        extruded_circle, "circle", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Circle meshes differ: {py_stl} vs {cli_stl}"


def test_ring():
    """Test the ring primitive."""
    # Create a ring with outer radius 1.0 and inner radius 0.5
    # Note: ring function expects (outer_radius, inner_radius, center_x, center_y)
    ring = fps.ring(1.0, 0.5, 0.0, 0.0)
    
    # Create a 3D shape by extruding the ring
    x, y, z = fp.x(), fp.y(), fp.z()
    extruded_ring = fpm.max(ring, fpm.abs(z) - 0.2)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        extruded_ring, "ring", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Ring meshes differ: {py_stl} vs {cli_stl}"


def test_cylinder_z():
    """Test the cylinder_z primitive."""
    # Create a cylinder along the z-axis
    # Note: cylinder_z function expects (radius, height, base) where base is a tuple
    cylinder = fps.cylinder_z(0.5, 1.0, (0.0, 0.0, 0.0))
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cylinder, "cylinder_z", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cylinder_z meshes differ: {py_stl} vs {cli_stl}"


def test_cone_z():
    """Test the cone_z primitive."""
    # Create a cone along the z-axis
    # Note: cone_z function expects (radius, height, base) where base is a tuple
    cone = fps.cone_z(0.5, 1.0, (0.0, 0.0, 0.0))
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        cone, "cone_z", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Cone_z meshes differ: {py_stl} vs {cli_stl}"


def test_pyramid_z():
    """Test the pyramid_z primitive."""
    # Create a pyramid with a rectangular base
    # Note: pyramid_z function expects (a, b, zmin, height) where a and b are tuples
    pyramid = fps.pyramid_z((-0.5, -0.5), (0.5, 0.5), 0.0, 1.0)
    
    # Test meshing with both methods and compare results
    success, py_stl, cli_stl = shape_dual_meshing(
        pyramid, "pyramid_z", depth=TEST_DEPTH, scale=TEST_SCALE
    )
    
    # Check that both meshes were created and are similar
    assert success, f"Pyramid_z meshes differ: {py_stl} vs {cli_stl}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])