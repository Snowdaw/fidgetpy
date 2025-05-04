import pytest
import numpy as np
import fidgetpy as fp

# --- Test Data ---
points_np = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)  # Shape (1, 4) for x,y,z,a

# --- Expressions ---
x = fp.var("X")
y = fp.y()
z = fp.var("z")
a = fp.var("a")
b = fp.var("b")
c = fp.var("c")

expr_abc = a + b + c
expr_ab = a + b

def test_eval_missing_variable():
    """Test combined error message with missing variables"""
    with pytest.raises(ValueError) as excinfo:
        fp.eval(expr_abc, points_np[:, :2], variables=[a, b])  # Missing c
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing: c" in error_msg
    assert "Unused:" not in error_msg  # No unused variables

def test_eval_unused_variable():
    """Test combined error message with unused variables"""
    with pytest.raises(ValueError) as excinfo:
        fp.eval(expr_ab, np.hstack([points_np[:, :2], points_np[:, :1]]), variables=[a, b, c])  # c is unused
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing:" not in error_msg  # No missing variables
    assert "Unused: c" in error_msg

def test_eval_both_missing_and_unused():
    """Test combined error message with both missing and unused variables"""
    with pytest.raises(ValueError) as excinfo:
        # expr_abc needs a, b, c but we provide a, d
        d = fp.var("d")
        values = points_np[:, :2]  # Only enough columns for a, d
        fp.eval(expr_abc, values, variables=[a, d])  # Missing b, c and unused d
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing: " in error_msg
    assert "b" in error_msg and "c" in error_msg  # Both b and c are missing
    assert "Unused: d" in error_msg

def test_mesh_missing_variable():
    """Test combined error message with missing variables in mesh"""
    with pytest.raises(ValueError) as excinfo:
        # Meshing requires all custom variables to be provided
        fp.mesh(expr_abc, variables=[a, b], variable_values=[1.0, 2.0])  # Missing c
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing: c" in error_msg
    assert "Unused:" not in error_msg  # No unused variables

def test_mesh_unused_variable():
    """Test combined error message with unused variables in mesh"""
    with pytest.raises(ValueError) as excinfo:
        # c is not used in expr_ab
        fp.mesh(expr_ab, variables=[a, b, c], variable_values=[1.0, 2.0, 3.0])
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing:" not in error_msg  # No missing variables
    assert "Unused: c" in error_msg

def test_mesh_both_missing_and_unused():
    """Test combined error message with both missing and unused variables in mesh"""
    with pytest.raises(ValueError) as excinfo:
        # expr_abc needs a, b, c but we provide a, d
        d = fp.var("d")
        fp.mesh(expr_abc, variables=[a, d], variable_values=[1.0, 4.0])
    
    # Check for the new error message format
    error_msg = str(excinfo.value)
    assert "Missing or unused variable(s) found in mapping:" in error_msg
    assert "Missing: " in error_msg
    assert "b" in error_msg and "c" in error_msg  # Both b and c are missing
    assert "Unused: d" in error_msg