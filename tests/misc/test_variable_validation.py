import pytest
import numpy as np
import fidgetpy as fp

# --- Test Data ---
points_np = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32).T # Shape (5, 1) for single var
points_np_xyz = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32) # Shape (2, 3)
points_np_custom = np.array([[0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.8]], dtype=np.float32) # Shape (2, 4) for x,y,z,a

# --- Expressions ---
x = fp.var("X")
y = fp.y()
z = fp.var("z")
a = fp.var("a")
b = fp.var("b")

expr_xyz = x + y * z
expr_a = a * 2.0
expr_ab = a + b
expr_xyz_a = x + y + z + a

# --- Tests for eval() ---

def test_eval_correct_vars():
    """Test eval with the exact required variables."""
    result = fp.eval(expr_xyz, points_np_xyz, variables=[x, y, z])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_eval_correct_vars_custom():
    """Test eval with a custom variable."""
    result = fp.eval(expr_a, points_np, variables=[a])
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    np.testing.assert_allclose(result, points_np.flatten() * 2.0)

def test_eval_missing_var():
    """Test eval raises ValueError when a required variable is missing."""
    with pytest.raises(ValueError, match="Missing or unused variable.*\nMissing: z"):
         fp.eval(expr_xyz, points_np_xyz[:, :2], variables=[x, y]) # Only provide x, y data and vars

def test_eval_missing_var_custom():
    """Test eval raises ValueError for missing custom variable."""
    with pytest.raises(ValueError, match="Missing or unused variable.*\nMissing: b"):
        fp.eval(expr_ab, points_np, variables=[a]) # Only provide 'a'

def test_eval_extra_var():
    """Test eval raises ValueError when an unused variable is provided."""
    points_extra = np.hstack([points_np_xyz, points_np_xyz[:, :1]]) # Add extra column
    with pytest.raises(ValueError, match="Missing or unused variable.*\nUnused: a"):
        fp.eval(expr_xyz, points_extra, variables=[x, y, z, a]) # Provide x, y, z, a

def test_eval_extra_var_custom():
    """Test eval raises ValueError for extra custom variable."""
    points_extra = np.hstack([points_np, points_np]) # Add extra column
    with pytest.raises(ValueError, match="Missing or unused variable.*\nUnused: b"):
        fp.eval(expr_a, points_extra, variables=[a, b]) # Provide a, b

# --- Tests for mesh() ---
# Meshing validation only applies when custom vars are present and substituted

def test_mesh_correct_custom_vars():
    """Test mesh with correct custom variables."""
    # mesh_impl substitutes custom vars, so success means no validation error
    mesh = fp.mesh(expr_xyz_a, variables=[a], variable_values=[1.0])
    assert mesh is not None # Check if meshing ran without error

def test_mesh_missing_custom_var():
    """Test mesh raises ValueError for missing custom variable."""
    with pytest.raises(ValueError, match="Missing or unused variable.*\nMissing: a"):
        # Try to mesh expr_xyz_a, providing only 'b' for substitution
        # Note: check_for_custom_vars requires *both* variables and variable_values
        # to be present if custom vars exist. The validation happens *after* this check.
        # We provide 'b' here, which isn't used by expr_xyz_a, triggering the validation.
        fp.mesh(expr_xyz_a, variables=[b], variable_values=[1.0])

def test_mesh_extra_custom_var():
    """Test mesh raises ValueError for extra custom variable."""
    with pytest.raises(ValueError, match="Missing or unused variable.*\nUnused: b"):
         # Provide 'a' (required) and 'b' (extra)
        fp.mesh(expr_xyz_a, variables=[a, b], variable_values=[1.0, 2.0])

def test_mesh_ignores_xyz_validation():
    """Test that mesh validation ignores x, y, z."""
    # Provide 'a' (required) and 'x' (should be ignored by validation)
    mesh = fp.mesh(expr_xyz_a, variables=[a, x], variable_values=[1.0, 99.0]) # x value doesn't matter
    assert mesh is not None # Check if meshing ran without error

def test_mesh_no_custom_vars_no_validation():
    """Test that mesh doesn't run validation if no custom vars are present."""
    # This expression only uses x, y, z. No variables/variable_values needed.
    mesh = fp.mesh(expr_xyz)
    assert mesh is not None # Check if meshing ran without error
    # Also test providing xyz explicitly (should not trigger validation)
    mesh = fp.mesh(expr_xyz, variables=[x,y,z], variable_values=[0.0, 0.0, 0.0]) # Values don't matter here
    assert mesh is not None