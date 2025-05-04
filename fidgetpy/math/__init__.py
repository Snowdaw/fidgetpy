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
    add, sub, mul, div, min, max,
    clamp, abs, sign, floor, ceil, round,
    fract, mod, pow, sqrt, exp, ln
)

from .trigonometry import (
    sin, cos, tan, asin, acos, atan, atan2
)

from .vector_math import (
    length, distance, dot, dot2, ndot, cross, normalize
)

from .transformations import (
    translate, rotate, scale,
    remap_xyz, remap_affine
)

from .domain_manipulation import (
    repeat, mirror, symmetry
)

from .interpolation import (
    mix, lerp, smoothstep, step, smootherstep,
    interpolate, threshold, pulse
)

from .logical import (
    logical_and, logical_or, logical_not, logical_xor,
    python_and, python_or, logical_if
)

# Define __all__ to control what gets imported with "from fidgetpy.math import *"
__all__ = [
    # Basic Math Functions
    'add', 'sub', 'mul', 'div', 'min', 'max',
    'clamp', 'abs', 'sign', 'floor', 'ceil', 'round',
    'fract', 'mod', 'pow', 'sqrt', 'exp', 'ln',
    
    # Trigonometric Functions
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    
    # Vector Math
    'length', 'distance', 'dot', 'dot2', 'ndot', 'cross', 'normalize',
    
    # Transformations
    'translate', 'rotate', 'scale',
    'remap_xyz', 'remap_affine', 'combine_matrices',
    
    # Domain Manipulation
    'repeat', 'mirror', 'symmetry'
    
    # Interpolation
    'mix', 'lerp', 'smoothstep', 'step', 'smootherstep',
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