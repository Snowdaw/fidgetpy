"""
Math operations for Fidget.

This module provides mathematical operations for SDF expressions, including:
- Basic math functions (min, max, abs, etc.)
- Trigonometric functions (sin, cos, tan, etc.)
- Vector math (length, dot, cross, etc.)
- Transformations (translate, scale, rotate, etc.)
- Domain manipulation (repeat, mirror, etc.)
- Interpolation functions (mix, lerp, smoothstep, etc.)
- Logical operations (logical_and, logical_or, logical_not, etc.)
"""

# Import everything from each module with * to avoid having module references
from .basic_math import (
    min, max, clamp, abs, sign, floor, ceil, round, fract, mod, pow, sqrt, exp, ln
)

from .trigonometry import (
    sin, cos, tan, asin, acos, atan, atan2
)

# Import from improved vector_math module
from .vector_math_improved import (
    # Original functions (for backward compatibility)
    length, distance, dot, dot2, ndot, cross, normalize,
    
    # New 2D vector functions with explicit parameters
    length_2d, distance_2d, dot_2d, dot2_2d, ndot_2d, normalize_2d,
    
    # New 3D vector functions with explicit parameters
    length_3d, distance_3d, dot_3d, dot2_3d, cross_3d, normalize_3d
)

from .transformations import (
    translate, translate_x, translate_y, translate_z,
    scale_xyz, scale, rotate_x, rotate_y, rotate_z,
    remap_xyz, remap_affine, combine_matrices,
    make_translation_matrix, make_scaling_matrix,
    make_rotation_x_matrix, make_rotation_y_matrix, make_rotation_z_matrix
)

# Import from improved domain_manipulation module
from .domain_manipulation_improved import (
    repeat, repeat_xyz, repeat_x, repeat_y, repeat_z, 
    mirror_x, mirror_y, mirror_z,
    symmetry_x, symmetry_y, symmetry_z
)

# Import from improved interpolation module
from .interpolation_improved import (
    # Original functions (for backward compatibility)
    mix, lerp, smoothstep, step, smootherstep,
    
    # New interpolation functions
    interpolate, threshold, pulse
)

from .logical import (
    logical_and, logical_or, logical_not, logical_xor,
    python_and, python_or, logical_if
)

# Define __all__ to control what gets imported with "from fidgetpy.math import *"
__all__ = [
    # Basic Math Functions
    'min', 'max', 'clamp', 'abs', 'sign', 'floor', 'ceil', 'round',
    'fract', 'mod', 'pow', 'sqrt', 'exp', 'ln',
    
    # Trigonometric Functions
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    
    # Vector Math - Original Functions
    'length', 'distance', 'dot', 'dot2', 'ndot', 'cross', 'normalize',
    
    # Vector Math - New 2D Functions
    'length_2d', 'distance_2d', 'dot_2d', 'dot2_2d', 'ndot_2d', 'normalize_2d',
    
    # Vector Math - New 3D Functions
    'length_3d', 'distance_3d', 'dot_3d', 'dot2_3d', 'cross_3d', 'normalize_3d',
    
    # Transformations
    'translate', 'translate_x', 'translate_y', 'translate_z',
    'scale_xyz', 'scale', 'rotate_x', 'rotate_y', 'rotate_z',
    'remap_xyz', 'remap_affine', 'combine_matrices',
    
    # Affine Matrix Helpers
    'make_translation_matrix', 'make_scaling_matrix',
    'make_rotation_x_matrix', 'make_rotation_y_matrix', 'make_rotation_z_matrix',
    
    # Domain Manipulation
    'repeat', 'repeat_xyz', 'repeat_x', 'repeat_y', 'repeat_z', 
    'mirror_x', 'mirror_y', 'mirror_z',
    'symmetry_x', 'symmetry_y', 'symmetry_z',
    
    # Interpolation - Original Functions
    'mix', 'lerp', 'smoothstep', 'step', 'smootherstep',
    
    # Interpolation - New Functions
    'interpolate', 'threshold', 'pulse',
    
    # Logical Operations
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    'python_and', 'python_or', 'logical_if'
]

# Clean up any modules that might have been imported
import sys as _sys
for _module in ['basic_math', 'trigonometry', 'vector_math', 'vector_math_improved',
                'transformations', 'domain_manipulation', 'domain_manipulation_improved', 
                'interpolation', 'interpolation_improved',
                'logical', 'builtins', 'py_math', 'fp']:
    if _module in globals():
        globals().pop(_module)
    if f"fidgetpy.math.{_module}" in _sys.modules:
        _sys.modules.pop(f"fidgetpy.math.{_module}")

# Remove the cleanup variables themselves
del _module
del _sys

# We're keeping all math functions in Python for simplicity and consistency
# This allows us to extend them with our Python extension mechanism