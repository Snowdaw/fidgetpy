"""
Shape primitives for Fidget.

This module provides a collection of common shape primitives that can be used
to build more complex shapes, including:
- Basic primitives (sphere, box, plane, torus)
- Cylinder-based shapes (cylinder, cone, capsule)
- Rounded shapes (rounded_box, round_cone, capped_torus)
- Curve-based shapes (line_segment, quadratic_bezier, cubic_bezier)
- Specialized shapes (ellipsoid, box_frame, death_star)

All functions return SDF expressions that can be combined using the standard
operators (+, -, *, /, etc.), methods (min, max, etc.), and operations from
the fidgetpy.ops module.
"""

# Import everything from each module
from .primitives import (
    sphere, box_exact, box_mitered, box_mitered_centered,
    rectangle, rectangle_centered_exact, plane, bounded_plane, half_space,
    torus, torus_z, octahedron, hexagonal_prism,
    circle, ring, polygon, half_plane, triangle,
    cylinder_z, cone_z, cone_ang_z, pyramid_z, extrude_z, gyroid
)

from .rounded_shapes import (
    rounded_box, rounded_cylinder, round_cone, capped_torus,
    solid_angle
)

from .cylinders import (
    cylinder, infinite_cylinder, bounded_infinite_cylinder, capsule, vertical_capsule,
    cone, capped_cone, capped_cylinder
)

from .curves import (
    line_segment, quadratic_bezier, cubic_bezier, polyline, bezier_spline,
    cubic_bezier_spline_variable_radius
)

from .specialized import (
    ellipsoid, triangular_prism, box_frame, link, cut_sphere,
    cut_hollow_sphere, death_star, pyramid, rhombus
)

# Define __all__ to control what gets imported with "from fidgetpy.shape import *"
__all__ = [
    # Primitives
    'sphere', 'box_exact', 'box_exact_centered', 'box_mitered', 'box_mitered_centered',
    'rectangle', 'rectangle_centered_exact', 'plane', 'bounded_plane', 'half_space',
    'torus', 'torus_z', 'octahedron', 'hexagonal_prism',
    'circle', 'ring', 'polygon', 'half_plane', 'triangle',
    'cylinder_z', 'cone_z', 'cone_ang_z', 'pyramid_z', 'extrude_z', 'gyroid',
    
    # Rounded Shapes
    'rounded_box', 'rounded_cylinder', 'round_cone', 'capped_torus',
    'solid_angle',
    
    # Cylinders
    'cylinder', 'infinite_cylinder', 'bounded_infinite_cylinder', 'capsule', 'vertical_capsule',
    'cone', 'capped_cone', 'capped_cylinder',
    
    # Curves
    'line_segment', 'quadratic_bezier', 'cubic_bezier', 'polyline', 'bezier_spline',
    'cubic_bezier_spline_variable_radius'
    
    # Specialized
    'ellipsoid', 'triangular_prism', 'box_frame', 'link', 'cut_sphere',
    'cut_hollow_sphere', 'death_star', 'pyramid', 'rhombus'
]

# Clean up any modules that might have been imported
import sys as _sys
for _module in ['primitives', 'rounded_shapes', 'cylinders', 'curves', 'specialized',
                'math', 'builtins']:
    if _module in globals():
        globals().pop(_module)
    if f"fidgetpy.shape.{_module}" in _sys.modules:
        _sys.modules.pop(f"fidgetpy.shape.{_module}")

# Remove the cleanup variables themselves
del _module
del _sys