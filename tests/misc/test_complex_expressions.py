"""
Tests for complex expressions combining math and ops functionality.

This test file demonstrates how to create, evaluate, and mesh complex SDF
expressions that mix different styles of API calls, including:
- Math and ops submodule functions
- Function-style and method-style calls
- Custom variables
- Floating-point literals and expressions
- Python list and NumPy array inputs
"""

import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
import fidgetpy.ops as fpo


def test_complex_expression_creation_and_evaluation():
    """Test creating and evaluating a complex expression."""
    # Create standard variables
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Create custom variables
    radius = fp.var("radius")
    blend_factor = fp.var("blend")
    smooth_k = fp.var("smooth_k")
    
    # === Step 1: Create base shapes using a mix of methods and functions ===
    
    # Create a sphere using method calls
    sphere = (x*x + y*y + z*z).sqrt() - radius
    
    # Create a box using function calls
    box = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
    
    # Create a cylinder using a mix of both
    cylinder_radius = 0.5
    cylinder_height = 2.0
    cylinder = fpm.max(
        (x*x + z*z).sqrt() - cylinder_radius,  # Circular profile
        fpm.abs(y) - cylinder_height/2         # Finite height
    )
    
    # === Step 2: Apply transformations using both styles ===
    
    # Translate sphere using method call
    sphere_translated = sphere.translate(1.5, 0.5, 0.0)
    
    # Translate box using function call
    box_translated = fpm.translate(box, -1.5, -0.5, 0.0)
    
    # Rotate cylinder using method call
    cylinder_rotated = cylinder.rotate_x(fpm.sin(y) * 0.5)  # Dynamic rotation based on y
    
    # === Step 3: Apply domain operations ===
    
    # Mirror the box
    box_mirrored = fpo.mirror_x(box_translated)
    
    # Repeat the sphere
    sphere_repeated = fpo.repeat_limited(
        sphere_translated,
        (3.0, 3.0, 3.0),  # Cell size
        (2, 2, 1)         # Repetition counts
    )
    
    # === Step 4: Combine using boolean and blending operations ===
    
    # Union box and cylinder
    box_cylinder = fpo.union(box_mirrored, cylinder_rotated)
    
    # Mix of function and method for smooth operations
    smooth_union1 = fpo.smooth_union(sphere_repeated, box_cylinder, smooth_k)
    smooth_union2 = sphere_repeated.translate(0.25, 0.25, 0.25)  # Additional transform
    
    # Final blend with dynamic weight
    dynamic_blend = blend_factor.sin() * 0.5 + 0.5  # Oscillates between 0 and 1
    final_shape = fpm.mix(smooth_union1, smooth_union2, dynamic_blend)
    
    # === Step 5: Add some noise and displacement ===
    
    # Create noise function
    noise_scale = 4.0
    noise = fpm.sin(x * noise_scale) * fpm.cos(y * noise_scale) * fpm.sin(z * noise_scale) * 0.05
    
    # Apply displacement
    final_shape_displaced = fpo.displace(final_shape, noise)
    
    # === Step 6: Evaluate with different input types ===
    
    # Define test points
    list_points = [
        [0.0, 0.0, 0.0, 1.0, 0.5, 0.2],  # x, y, z, radius, blend, smooth_k
        [1.0, 1.0, 1.0, 0.8, 0.7, 0.3],
        [-1.0, 0.5, -0.5, 1.2, 0.2, 0.1]
    ]
    
    numpy_points = np.array(list_points, dtype=np.float32)
    
    # Define variables list (must match order in points)
    variables = [x, y, z, radius, blend_factor, smooth_k]
    
    # Evaluate with list input
    list_result = fp.eval(final_shape_displaced, list_points, variables=variables)
    
    # Evaluate with numpy input
    numpy_result = fp.eval(final_shape_displaced, numpy_points, variables=variables)
    
    # Verify results
    assert isinstance(list_result, list), "List input should produce list output"
    assert isinstance(numpy_result, np.ndarray), "NumPy input should produce NumPy output"
    assert len(list_result) == len(list_points), "Output length should match input length"
    assert numpy_result.shape == (len(numpy_points),), "Output shape should match input rows"
    
    # Verify values match between list and numpy versions
    for i in range(len(list_result)):
        assert abs(list_result[i] - numpy_result[i]) < 1e-5, f"Results differ at index {i}"


def test_complex_expression_meshing():
    """Test meshing a complex expression with both list and numpy inputs."""
    # Create a simpler but still complex expression for meshing
    x, y, z = fp.x(), fp.y(), fp.z()
    radius = fp.var("radius")
    
    # Create a base shape: blend of sphere and box
    sphere = (x*x + y*y + z*z).sqrt() - radius
    box = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
    
    # Apply transforms
    sphere_moved = sphere.translate(0.5, 0.0, 0.0)
    box_moved = fpm.translate(box, -0.5, 0.0, 0.0)
    
    # Combine with smooth union
    shape = fpo.smooth_union(sphere_moved, box_moved, 0.3)
    
    # Test with different radius values
    test_radius = 0.7
    
    # Define points for evaluation (simple case: just radius as custom var)
    list_points = [
        [0.0, 0.0, 0.0, test_radius],  # x, y, z, radius
        [1.0, 0.0, 0.0, test_radius],
        [0.0, 1.0, 0.0, test_radius]
    ]
    numpy_points = np.array(list_points, dtype=np.float32)
    # Define the custom variable and its value for meshing
    mesh_variables = [radius]
    mesh_variable_values = [test_radius]

    # Generate mesh using list input (with low resolution for speed)
    list_mesh_data = fp.mesh(
        shape,
        depth=4,  # Low depth for testing
        variables=mesh_variables,
        variable_values=mesh_variable_values,
        numpy=False
    )

    # Generate mesh using numpy input
    numpy_mesh_data = fp.mesh(
        shape,
        depth=4,  # Low depth for testing
        variables=mesh_variables,
        variable_values=mesh_variable_values,
        numpy=True
    )
    
    # Verify mesh outputs (accessing attributes of the PyMesh object)
    assert list_mesh_data is not None, "List-based mesh generation should succeed"
    assert numpy_mesh_data is not None, "NumPy-based mesh generation should succeed"

    # Check vertices
    assert hasattr(list_mesh_data, 'vertices')
    assert hasattr(numpy_mesh_data, 'vertices')
    assert isinstance(list_mesh_data.vertices, list), "List input should produce list vertices"
    assert isinstance(numpy_mesh_data.vertices, np.ndarray), "NumPy input should produce NumPy vertices"
    assert len(list_mesh_data.vertices) > 0, "Should have vertices"
    assert len(list_mesh_data.vertices[0]) == 3, "Vertices should be 3D points"

    # Check triangles
    assert hasattr(list_mesh_data, 'triangles')
    assert hasattr(numpy_mesh_data, 'triangles')
    assert isinstance(list_mesh_data.triangles, list), "List input should produce list triangles"
    assert isinstance(numpy_mesh_data.triangles, np.ndarray), "NumPy input should produce NumPy triangles"
    assert len(list_mesh_data.triangles) > 0, "Should have at least one triangle"
    assert len(list_mesh_data.triangles[0]) == 3, "Triangles should have 3 vertex indices"


if __name__ == "__main__":
    pytest.main([__file__])