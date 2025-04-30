import fidgetpy as fp
import fidgetpy.shape as fps
import numpy as np
import pytest # Import pytest for potential use of fixtures or markers later

# Define a tolerance for floating point comparisons
TOLERANCE = 1e-6

def test_mesh_with_bounds():
    """Test meshing with explicit bounds respects those bounds."""
    # Create a sphere with radius 1.0
    sphere = fps.sphere(1.0)

    # Define bounds that should capture the sphere
    bounds_min = np.array([-1.5, -1.5, -1.5])
    bounds_max = np.array([1.5, 1.5, 1.5])
    depth = 5

    # Generate mesh with bounds
    mesh_data = fp.mesh(
        sphere,
        depth=depth,
        numpy=True,
        bounds_min=bounds_min.tolist(), # Pass as list
        bounds_max=bounds_max.tolist()  # Pass as list
    )

    # Assert that mesh data was generated
    assert mesh_data.vertices.shape[0] > 0, "No vertices generated"
    assert mesh_data.triangles.shape[0] > 0, "No triangles generated"

    # Check if vertices are within the specified bounds (with tolerance)
    min_vertex = np.min(mesh_data.vertices, axis=0)
    max_vertex = np.max(mesh_data.vertices, axis=0)

    assert np.all(min_vertex >= bounds_min - TOLERANCE), f"Min vertex {min_vertex} outside lower bounds {bounds_min}"
    assert np.all(max_vertex <= bounds_max + TOLERANCE), f"Max vertex {max_vertex} outside upper bounds {bounds_max}"

def test_mesh_with_larger_bounds():
    """Test meshing with larger explicit bounds."""
    # Create a sphere with radius 1.0 (smaller than bounds)
    sphere = fps.sphere(1.0)

    # Define larger bounds
    bounds_min = np.array([-3.0, -3.0, -3.0])
    bounds_max = np.array([3.0, 3.0, 3.0])
    depth = 5

    # Generate mesh with bounds
    mesh_data = fp.mesh(
        sphere,
        depth=depth,
        numpy=True,
        bounds_min=bounds_min.tolist(),
        bounds_max=bounds_max.tolist()
    )

    # Assert that mesh data was generated
    assert mesh_data.vertices.shape[0] > 0, "No vertices generated"
    assert mesh_data.triangles.shape[0] > 0, "No triangles generated"

    # Check if vertices are within the specified bounds
    min_vertex = np.min(mesh_data.vertices, axis=0)
    max_vertex = np.max(mesh_data.vertices, axis=0)

    assert np.all(min_vertex >= bounds_min - TOLERANCE), f"Min vertex {min_vertex} outside lower bounds {bounds_min}"
    assert np.all(max_vertex <= bounds_max + TOLERANCE), f"Max vertex {max_vertex} outside upper bounds {bounds_max}"

    # Additionally, check that the mesh is roughly sphere-sized, not filling the whole bounds
    assert np.all(min_vertex > bounds_min + 0.5), "Mesh seems too close to lower bounds" # Expect some margin
    assert np.all(max_vertex < bounds_max - 0.5), "Mesh seems too close to upper bounds" # Expect some margin


def test_mesh_with_offset_bounds():
    """Test meshing with offset bounds."""
    # Define offset bounds (not centered at origin)
    bounds_min = np.array([1.0, 1.0, 1.0])
    bounds_max = np.array([4.0, 4.0, 4.0])
    center = (bounds_min + bounds_max) / 2.0 # Calculate center of bounds

    # Create a sphere centered within the bounds
    sphere = fps.sphere(1.0).translate(center[0], center[1], center[2])
    depth = 5

    # Generate mesh with bounds
    mesh_data = fp.mesh(
        sphere,
        depth=depth,
        numpy=True,
        bounds_min=bounds_min.tolist(),
        bounds_max=bounds_max.tolist()
    )

    # Assert that mesh data was generated
    assert mesh_data.vertices.shape[0] > 0, "No vertices generated"
    assert mesh_data.triangles.shape[0] > 0, "No triangles generated"

    # Check if vertices are within the specified bounds
    min_vertex = np.min(mesh_data.vertices, axis=0)
    max_vertex = np.max(mesh_data.vertices, axis=0)

    assert np.all(min_vertex >= bounds_min - TOLERANCE), f"Min vertex {min_vertex} outside lower bounds {bounds_min}"
    assert np.all(max_vertex <= bounds_max + TOLERANCE), f"Max vertex {max_vertex} outside upper bounds {bounds_max}"

    # Check that the mesh is centered around the expected offset center
    mesh_center = (min_vertex + max_vertex) / 2.0
    assert np.allclose(mesh_center, center, atol=0.5), f"Mesh center {mesh_center} is not close to expected center {center}"

def test_mesh_bounds_precedence_error():
    """Test that providing both bounds and center raises an error."""
    sphere = fps.sphere(1.0)
    bounds_min = [-1.5, -1.5, -1.5]
    bounds_max = [1.5, 1.5, 1.5]
    center = [0.0, 0.0, 0.0] # Explicitly provide center
    depth = 5

    with pytest.raises(ValueError, match="Ambiguous meshing region"):
        fp.mesh(
            sphere,
            depth=depth,
            numpy=True,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            center=center # Provide center along with bounds
        )

# Note: We don't test providing bounds and scale, as scale always has a default
# and the error check focuses on the ambiguity introduced by providing 'center'.
# The error message correctly states that both center and scale are ignored if bounds are used.