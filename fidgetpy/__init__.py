"""
Fidget Python bindings - A Pythonic interface to the Fidget SDF library.

This module provides a clean, Pythonic API for creating and manipulating
Signed Distance Fields (SDFs) using the Fidget library.

Core functions:
- x(), y(), z(): Coordinate variables
- var(name): Custom variables
- constant(value): Constant values
- eval(expr, points): Evaluate expressions at points
- mesh(expr, numpy=False): Generate meshes from expressions, returns a Mesh object with attributes:
  * vertices: Points in 3D space (list or numpy array based on 'numpy' parameter)
  * triangles: Triangle indices (list or numpy array based on 'numpy' parameter)
- save_stl(mesh, filepath): Save a Mesh object to an STL file

Submodules:
- shape: Common shape primitives (sphere, box, cylinder, etc.)
- ops: Operations for combining shapes (smooth_union, etc.)
- math: Math operations and transformations
"""

# Import core functions from the Rust module
from fidgetpy.fidgetpy import (
    # Core variables
    x, y, z, var,
    
    # Evaluation and meshing
    eval, mesh, save_stl,
    
    # Import/Export
    from_vm, to_vm, from_frep, to_frep,
)

# Import submodules
from . import shape
from . import ops
from . import math

# ----------------------------------------------------------------------
# Expression extension functionality (integrated from extend_expr.py)
# ----------------------------------------------------------------------

# Use underscore prefix to hide from tab completion
import inspect as _inspect

def _extend_expressions():
    """
    Add methods to the SDF expression class.
    This function extends the SDF expression class with methods from the math module.
    """
    # Get the SDF expression class
    expr_class = type(x())
    
    # Add a marker attribute to identify SDF expressions
    setattr(expr_class, '_is_sdf_expr', True)
    
    # Get all functions from fidgetpy.math module
    math_functions = _inspect.getmembers(math, _inspect.isfunction)
    
    # For each function, add a method to the expression class that delegates to the function
    count = 0
    for name, func in math_functions:
        # Skip private functions (those starting with _)
        if name.startswith('_'):
            continue
            
        # Skip functions that are already implemented in Rust
        # These are the ones causing recursion issues
        if hasattr(expr_class, name):
            continue
            
        # Create a method that delegates to the function
        # The first argument to the function will be 'self'
        # Skip functions with parameter order issues
        if name in ('step', 'smoothstep', 'smootherstep', 'pulse'):
            # Don't register these as methods due to parameter ordering issues
            continue
            
        def create_method(math_func):
            def method(self, *args, **kwargs):
                return math_func(self, *args, **kwargs)
            return method
        
        # Add the method to the class
        setattr(expr_class, name, create_method(func))
        count += 1

# Initialize expression extensions when the module is imported
_extend_expressions()