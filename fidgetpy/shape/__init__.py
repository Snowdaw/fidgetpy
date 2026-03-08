"""
Shape primitives for Fidget.

This module provides a collection of common shape primitives that can be used
to build more complex shapes, including:
- Basic primitives (sphere, box, plane, torus)
- Cylinder-based shapes (cylinder, cone, capsule)
- Rounded shapes (rounded_box, round_cone, capped_torus)
- Curve-based shapes (line_segment, quadratic_bezier, cubic_bezier)
- Specialized shapes (ellipsoid, box_frame, death_star)
- 2D shapes for use with extrude_z (circle, rectangle, ring, polygon)

All functions return SDF expressions that can be combined using the standard
operators (+, -, *, /, etc.), methods (min, max, etc.), and operations from
the fidgetpy.ops module.
"""

# Primary 3D primitives
from .primitives import (
    sphere,
    box, box_exact,       # box_exact is a backward-compat alias for box
    torus,
    plane, half_space,
    octahedron, hexagonal_prism,
    gyroid,
    extrude_z,
    # 2D shapes
    circle, ring, polygon, rectangle,
)

# Rounded shapes (rounded_box is the canonical source)
from .rounded_shapes import (
    rounded_box,
    rounded_cylinder, round_cone, capped_torus,
    solid_angle,
)

# Cylinder-family shapes (all exact SDFs, y-axis aligned)
from .cylinders import (
    cylinder, infinite_cylinder, capsule,
    cone, capped_cone, capped_cylinder,
    bounded_infinite_cylinder,
)

# Curve-based shapes
from .curves import (
    line_segment, quadratic_bezier, cubic_bezier, polyline, bezier_spline,
    cubic_bezier_spline_variable_radius,
)

# Specialized shapes
from .specialized import (
    ellipsoid, triangular_prism, box_frame, link, cut_sphere,
    cut_hollow_sphere, death_star, pyramid, rhombus,
)

# ── Define __all__ ────────────────────────────────────────────────────────────
__all__ = [
    # Core 3D primitives
    'sphere',
    'box',
    'rounded_box',
    'torus',
    'plane',
    'half_space',
    'octahedron',
    'hexagonal_prism',
    'gyroid',

    # Cylinder family (exact SDFs, y-axis aligned)
    'cylinder',
    'infinite_cylinder',
    'bounded_infinite_cylinder',
    'capsule',
    'cone',
    'capped_cone',
    'capped_cylinder',

    # Rounded shapes
    'rounded_cylinder',
    'round_cone',
    'capped_torus',
    'solid_angle',

    # 2D shapes (for use with extrude_z)
    'circle',
    'rectangle',
    'ring',
    'polygon',
    'extrude_z',

    # Curves
    'line_segment',
    'quadratic_bezier',
    'cubic_bezier',
    'polyline',
    'bezier_spline',
    'cubic_bezier_spline_variable_radius',

    # Specialized
    'ellipsoid',
    'triangular_prism',
    'box_frame',
    'link',
    'cut_sphere',
    'cut_hollow_sphere',
    'death_star',
    'pyramid',
    'rhombus',
]

# Clean up any modules that might have been imported
import sys as _sys
for _module in ['primitives', 'rounded_shapes', 'cylinders', 'curves', 'specialized',
                'math', 'builtins']:
    if _module in globals():
        globals().pop(_module)
    if f"fidgetpy.shape.{_module}" in _sys.modules:
        _sys.modules.pop(f"fidgetpy.shape.{_module}")

del _module
del _sys
