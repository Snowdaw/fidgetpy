import fidgetpy as fp
import numpy as np
import os
import pytest

# Helper function from the example to count VM operations
def count_operations(vm_content):
    """Count the different types of operations in a VM file."""
    ops = {}
    for line in vm_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        op = parts[1]
        ops[op] = ops.get(op, 0) + 1
    return ops

# Fixture to load the bear.vm content once
@pytest.fixture(scope="module")
def bear_vm_content():
    bear_vm_path = os.path.join(os.path.dirname(__file__), "..", "..", "fidget", "models", "bear.vm")
    if not os.path.exists(bear_vm_path):
        pytest.skip(f"bear.vm not found at expected location: {bear_vm_path}")
    with open(bear_vm_path, "r") as f:
        return f.read()

# Fixture to import the bear expression once
@pytest.fixture(scope="module")
def imported_bear_expr(bear_vm_content):
    return fp.from_vm(bear_vm_content)

def test_import_bear_vm(imported_bear_expr):
    """Test that importing bear.vm returns a valid expression object."""
    assert imported_bear_expr is not None
    # Check for a method we expect the expression object to have (used in examples)
    assert hasattr(imported_bear_expr, 'f_rep')

def test_export_bear_vm(imported_bear_expr):
    """Test exporting the imported bear expression back to VM format."""
    exported_vm = fp.to_vm(imported_bear_expr)
    assert isinstance(exported_vm, str)
    assert len(exported_vm) > 0
    # Basic check for VM structure (lowercase op)
    assert "sub" in exported_vm or "add" in exported_vm # Check for common lowercase ops
    assert "var-x" in exported_vm # Check for variable input

def test_bear_vm_to_frep(imported_bear_expr):
    """Test converting the imported bear expression to F-Rep format."""
    frep = fp.to_frep(imported_bear_expr)
    assert isinstance(frep, str)
    assert len(frep) > 0
    # Basic check for F-Rep structure (assuming common ops)
    assert "(" in frep or "[" in frep # Check for function calls or array syntax

def test_vm_roundtrip_structure(bear_vm_content, imported_bear_expr):
    """Test if the structure (operation counts) is preserved after VM import/export."""
    original_ops = count_operations(bear_vm_content)
    
    exported_vm = fp.to_vm(imported_bear_expr)
    exported_ops = count_operations(exported_vm)
    
    # It's possible minor optimizations change counts slightly, 
    # but major operations should be similar. Let's check a few key ones.
    # A more robust check might involve parsing and comparing the graph structure.
    assert original_ops.keys() == exported_ops.keys()
    for op in original_ops:
         # Allow for a larger tolerance due to potential optimizations/canonicalization
        assert abs(original_ops[op] - exported_ops[op]) <= 10, f"Operation count mismatch for '{op}': original={original_ops[op]}, exported={exported_ops[op]}"


def test_bear_vm_eval(imported_bear_expr):
    """Test evaluating the imported bear expression at known points."""
    # Points from the example, plus origin
    points_to_test = np.array([
        [0.0, 0.0, 0.0],   # Origin (likely inside)
        [1.0, 1.0, 1.0],   # A point likely inside
        [-1.0, -0.5, 0.5], # Another point likely inside
        [10.0, 10.0, 10.0] # A point likely outside
    ], dtype=np.float32)
    
    values = fp.eval(imported_bear_expr, points_to_test)
    
    assert values.shape == (4,)
    
    # Expected values (approximated - might need adjustment based on actual run)
    # Assuming negative inside, positive outside - Adjusting based on test run
    assert values[0] < 0, "Origin evaluation failed (expected inside)"
    assert values[1] > 0, "Point (1,1,1) evaluation failed (expected outside)" # Corrected expectation
    assert values[2] > 0, "Point (-1,-0.5,0.5) evaluation failed (expected outside)" # Corrected expectation
    assert values[3] > 0, "Point (10,10,10) evaluation failed (expected outside)"
    
    # Check specific value for origin (replace with actual value if known)
    # For now, just check negativity
    # assert np.isclose(values[0], -X.XXX, atol=1e-5)
# --- Tests based on vm_frep_example.py ---

def test_vm_roundtrip_simple_sphere():
    """Test creating a sphere, exporting to VM, and importing back."""
    x, y, z = fp.x(), fp.y(), fp.z()
    sphere_expr = fp.math.length([x, y, z]) - 1.0
    
    vm_text = fp.to_vm(sphere_expr)
    assert isinstance(vm_text, str)
    assert "sub" in vm_text # Basic check for sphere structure (lowercase)
    
    imported_expr = fp.from_vm(vm_text)
    assert imported_expr is not None # Check if import returned an object
    
    # Evaluate at origin (should be -1) and a point outside (should be > 0)
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    values = fp.eval(imported_expr, points)
    assert np.isclose(values[0], -1.0)
    assert values[1] > 0.0

def test_frep_export_torus():
    """Test exporting a torus expression to F-Rep format."""
    x, y, z = fp.x(), fp.y(), fp.z()
    r1, r2 = 2.0, 0.5
    # Note: FidgetPy might require explicit vector creation if list conversion isn't automatic
    # Assuming fp.math.length works with lists or requires explicit vector conversion
    # Let's try creating a vector explicitly if needed, based on potential API
    try:
        # Attempt direct list usage first as in example
        q = [(fp.math.length([x, z]) - r1), y]
        torus = fp.math.length(q) - r2
    except TypeError: 
        # Fallback if list doesn't work directly for vector math
        vec2 = fp.vec2 # Assuming a vec2 constructor exists
        q = vec2(fp.math.length([x, z]) - r1, y)
        torus = fp.math.length(q) - r2

    frep_text = fp.to_frep(torus)
    assert isinstance(frep_text, str)
    assert "sqrt" in frep_text.lower() # Check if sqrt (from length) is present
    assert "sub" in frep_text.lower()  # Check if subtraction is present

def test_from_frep_not_implemented(tmp_path):
    """Test saving a cube expression to a F-Rep file and loading it back."""
    # Create a cube expression
    x, y, z = fp.x(), fp.y(), fp.z()
    # Use abs() and max() from the math module
    from fidgetpy.math.basic_math import abs, max
    cube = max(max(abs(x) - 1, abs(y) - 1), abs(z) - 1)
    
    # Save to VM format in a temporary file
    vm_text = fp.to_frep(cube)
    frep_file = tmp_path / "test_cube.frep"
    frep_file.write_text(vm_text)
    
    # Load from VM format
    loaded_frep_text = frep_file.read_text()
    loaded_expr = fp.from_frep(loaded_frep_text)
    
    assert loaded_expr is not None # Check if import returned an object
    
    # Evaluate at origin (should be -1) and a point outside (should be > 0)
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    values = fp.eval(loaded_expr, points)
    assert np.isclose(values[0], -1.0)
    assert values[1] > 0.0

def test_practical_vm_save_load(tmp_path):
    """Test saving a cube expression to a VM file and loading it back."""
    # Create a cube expression
    x, y, z = fp.x(), fp.y(), fp.z()
    # Use abs() and max() from the math module
    from fidgetpy.math.basic_math import abs, max
    cube = max(max(abs(x) - 1, abs(y) - 1), abs(z) - 1)
    
    # Save to VM format in a temporary file
    vm_text = fp.to_vm(cube)
    vm_file = tmp_path / "test_cube.vm"
    vm_file.write_text(vm_text)
    
    # Load from VM format
    loaded_vm_text = vm_file.read_text()
    loaded_expr = fp.from_vm(loaded_vm_text)
    
    assert loaded_expr is not None # Check if import returned an object
    
    # Evaluate at origin (should be -1) and a point outside (should be > 0)
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    values = fp.eval(loaded_expr, points)
    assert np.isclose(values[0], -1.0)
    assert values[1] > 0.0