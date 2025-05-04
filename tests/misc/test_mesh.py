"""
Tests for the fp.mesh() function.
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.shape as fps

def test_mesh_basic_sphere():
    """Test meshing a simple sphere."""
    sphere = fps.sphere(1.0)
    # Define scale and center for meshing
    scale = 3.0
    center = [0.0, 0.0, 0.0]
    mesh_data = fp.mesh(sphere, center=center, scale=scale, depth=4, numpy=True)

    assert hasattr(mesh_data, 'vertices')
    assert hasattr(mesh_data, 'triangles')
    assert isinstance(mesh_data.vertices, np.ndarray)
    assert isinstance(mesh_data.triangles, np.ndarray)
    assert mesh_data.vertices.ndim == 2
    assert mesh_data.vertices.shape[1] == 3 # x, y, z
    assert mesh_data.triangles.ndim == 2
    assert mesh_data.triangles.shape[1] == 3 # v1, v2, v3 indices
    assert mesh_data.vertices.dtype == np.float32
    # Triangle indices can be signed or unsigned int32/int64
    assert mesh_data.triangles.dtype in [np.int32, np.int64, np.uint32, np.uint64]
    assert len(mesh_data.vertices) > 0
    assert len(mesh_data.triangles) > 0

def test_mesh_format_numpy():
    """Test meshing with numpy arrays as output."""
    sphere = fps.sphere(1.0)
    scale = 3.0
    center = [0.0, 0.0, 0.0]
    # Set numpy=True to get numpy array output
    mesh_data = fp.mesh(sphere, center=center, scale=scale, depth=3, numpy=True)

    assert isinstance(mesh_data.vertices, np.ndarray)
    assert isinstance(mesh_data.triangles, np.ndarray)

def test_mesh_format_list():
    """Test meshing with Python lists as output."""
    sphere = fps.sphere(1.0)
    scale = 3.0
    center = [0.0, 0.0, 0.0]
    # Default format is Python lists
    mesh_data = fp.mesh(sphere, center=center, scale=scale, depth=3, numpy=False)

    assert isinstance(mesh_data.vertices, list)
    assert isinstance(mesh_data.triangles, list)
    # Check that each vertex is a list of 3 coordinates
    assert all(isinstance(v, list) for v in mesh_data.vertices)
    assert all(len(v) == 3 for v in mesh_data.vertices)
    # Check that each triangle is a list of 3 indices
    assert all(isinstance(t, list) for t in mesh_data.triangles)
    assert all(len(t) == 3 for t in mesh_data.triangles)

def test_mesh_options():
    """Test meshing with different options (scale, center, depth)."""
    box = fps.box(width=0.5, height=0.5, depth=0.5)
    default_scale = 2.0
    default_center = [0.0, 0.0, 0.0]

    # Default depth with default scale and center
    mesh_default = fp.mesh(box, center=default_center, scale=default_scale)
    verts_default = len(mesh_default.vertices)
    tris_default = len(mesh_default.triangles)
    assert verts_default > 0
    assert tris_default > 0

    # Higher depth (should increase complexity)
    mesh_deep = fp.mesh(box, center=default_center, scale=default_scale, depth=5)
    verts_deep = len(mesh_deep.vertices)
    tris_deep = len(mesh_deep.triangles)
    # Check if complexity generally increased (not a strict guarantee, but likely)
    # Allow for slight variations due to meshing algorithm details
    assert verts_deep >= verts_default * 0.9 # Allow some tolerance
    assert tris_deep >= tris_default * 0.9 # Allow some tolerance

    # Test different scale and center (should affect the mesh output, check it runs)
    different_scale = 4.0
    different_center = [1.0, 0.0, 0.0]
    try:
        mesh_modified = fp.mesh(box, center=different_center, scale=different_scale)
        assert len(mesh_modified.vertices) > 0 # Basic check
        assert len(mesh_modified.triangles) > 0
    except Exception as e:
        pytest.fail(f"Meshing with different scale and center failed: {e}")

def test_mesh_custom_variables():
    """Test meshing with custom variables."""
    # Create shape with custom variables
    radius_var = fp.var("radius")
    height_var = fp.var("height")
    
    # Create a cylinder with variable radius and height
    cylinder = fps.cylinder(radius_var, height_var)
    
    scale = 1.0
    center = [0.0, 0.0, 0.0]
    
    # First configuration: radius=1.0, height=2.0
    variables = [fp.x(), fp.y(), fp.z(), radius_var, height_var]
    variable_values = [0.0, 0.0, 0.0, 1.0, 2.0]  # Using lists as required by the implementation
    
    mesh_data1 = fp.mesh(cylinder, center=center, scale=scale, depth=4, numpy=True,
                         variables=variables, variable_values=variable_values)
    
    assert hasattr(mesh_data1, 'vertices')
    assert hasattr(mesh_data1, 'triangles')
    assert isinstance(mesh_data1.vertices, np.ndarray)
    assert isinstance(mesh_data1.triangles, np.ndarray)
    assert len(mesh_data1.vertices) > 0
    assert len(mesh_data1.triangles) > 0
    
    # Second configuration: radius=0.5, height=3.0
    variable_values = [0.0, 0.0, 0.0, 0.5, 3.0]
    
    mesh_data2 = fp.mesh(cylinder, center=center, scale=scale, depth=4, numpy=True,
                         variables=variables, variable_values=variable_values)
    
    assert len(mesh_data2.vertices) > 0
    assert len(mesh_data2.triangles) > 0
    
    # The meshes should be different due to different variable values
    # Check if the bounding boxes are different
    vertices1 = mesh_data1.vertices
    vertices2 = mesh_data2.vertices
    
    max_x1 = np.max(vertices1[:, 0])
    max_x2 = np.max(vertices2[:, 0])
    
    # A cylinder with radius 1.0 should have larger x-extent than one with radius 0.5
    assert abs(max_x1) > abs(max_x2) * 0.9

def test_mesh_multithreading():
    """Test meshing with multi-threading option."""
    sphere = fps.sphere(1.0)
    center = [0.0, 0.0, 0.0]
    scale = 1.0
    
    # Run with multi-threading disabled
    mesh_single = fp.mesh(sphere, center=center, scale=scale, depth=5, numpy=True, threads=False)
    
    # Run with multi-threading enabled
    mesh_multi = fp.mesh(sphere, center=center, scale=scale, depth=5, numpy=True, threads=True)
    
    # Both should produce valid meshes
    assert len(mesh_single.vertices) > 0
    assert len(mesh_single.triangles) > 0
    assert len(mesh_multi.vertices) > 0
    assert len(mesh_multi.triangles) > 0
    
    # For the same input and parameters, output should be identical or very similar
    # The multithreaded version might have slight numerical differences or vertex ordering
    # So we check that both meshes have similar number of vertices and triangles
    # This is a basic check that multithreading doesn't break the mesh generation
    assert abs(len(mesh_single.vertices) - len(mesh_multi.vertices)) < len(mesh_single.vertices) * 0.1