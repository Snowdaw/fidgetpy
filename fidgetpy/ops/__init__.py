"""
Operations for combining and manipulating SDF shapes.

This module provides operations for combining and manipulating SDF expressions, including:
- Boolean operations (union, intersection, difference)
- Smooth operations (smooth_union, smooth_intersection, smooth_difference)
- Blending operations (blend, mix, smooth_min, smooth_max)
- Domain operations (elongate, onion, twist, bend)
- Miscellaneous operations (extrusion, revolution, repeat)

IMPORTANT: Unlike math functions, operations can ONLY be used as direct function calls.
They do not support method call syntax. For example, use fpo.union(a, b) instead of a.union(b).
"""

import fidgetpy as fp
import fidgetpy.math as fpm

# Import everything from each module with * to avoid having module references
from .boolean_ops import (
    union, intersection, difference,
    smooth_union, smooth_intersection, smooth_difference,
    complement, boolean_and, boolean_or, boolean_not, boolean_xor
)

from .blending_ops import (
    blend, mix, lerp, 
    smooth_min, smooth_max,
    exponential_smooth_min, exponential_smooth_max,
    power_smooth_min, power_smooth_max,
    soft_clamp, quad_bezier_blend
)

from .domain_ops import (
    onion, elongate, twist, bend,
    round, shell, displace,
    mirror_x, mirror_y, mirror_z
)

from .misc_ops import (
    smooth_step_union, chamfer_union, chamfer_intersection,
    engrave, extrusion, revolution, 
    repeat, repeat_limited, weight_blend
)

# Define __all__ to control what gets imported with "from fidgetpy.ops import *"
__all__ = [
    # Boolean Operations
    'union', 'intersection', 'difference',
    'smooth_union', 'smooth_intersection', 'smooth_difference',
    'complement', 'boolean_and', 'boolean_or', 'boolean_not', 'boolean_xor',
    
    # Blending Operations
    'blend', 'mix', 'lerp',
    'smooth_min', 'smooth_max',
    'exponential_smooth_min', 'exponential_smooth_max',
    'power_smooth_min', 'power_smooth_max',
    'soft_clamp', 'quad_bezier_blend',
    
    # Domain Operations
    'onion', 'elongate', 'twist', 'bend',
    'round', 'shell', 'displace',
    'mirror_x', 'mirror_y', 'mirror_z',
    
    # Miscellaneous Operations
    'smooth_step_union', 'chamfer_union', 'chamfer_intersection',
    'engrave', 'extrusion', 'revolution',
    'repeat', 'repeat_limited', 'weight_blend'
]

# Clean up any modules that might have been imported
import sys as _sys
for _module in ['boolean_ops', 'blending_ops', 'domain_ops', 'misc_ops', 
                'builtins', 'py_math', 'fp', 'fpm']:
    if _module in globals():
        globals().pop(_module)
    if f"fidgetpy.ops.{_module}" in _sys.modules:
        _sys.modules.pop(f"fidgetpy.ops.{_module}")

# Remove the cleanup variables themselves
del _module
del _sys