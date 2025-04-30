# Fidget Python Bindings - Shape Module

This module provides a comprehensive collection of shape primitives for creating signed distance fields (SDFs) in the Fidget Python bindings.

## Documentation Style

All shapes in this module follow a consistent documentation style as outlined in the [DOCUMENTATION_STYLE_GUIDE.md](./DOCUMENTATION_STYLE_GUIDE.md) file. This ensures that all shape functions have:

- Clear, detailed descriptions
- Complete parameter documentation with constraints
- Return value documentation
- Error documentation
- Usage examples

## Module Structure

The shape functions have been organized into logically grouped files:

- **primitives.py**: Basic shape primitives
  - sphere, box, plane, torus, octahedron, hexagonal_prism

- **rounded_shapes.py**: Shapes with rounded features
  - rounded_box, rounded_cylinder, round_cone, capped_torus, solid_angle

- **cylinders.py**: Cylinder-based shapes
  - cylinder, infinite_cylinder, capsule, vertical_capsule, cone, capped_cone, capped_cylinder

- **curves.py**: Curve-based shapes
  - line_segment, quadratic_bezier, cubic_bezier, polyline

- **specialized.py**: More specialized shapes
  - ellipsoid, triangular_prism, box_frame, link, cut_sphere, cut_hollow_sphere, death_star, pyramid, rhombus

## Module Organization

The shape module has been designed to hide implementation details while providing a clean, user-friendly API. All functions are available directly from the `fidgetpy.shape` namespace, while the individual module files are hidden from the user.

Example usage:

```python
import fidgetpy as fp
import fidgetpy.shape as fps

# Basic shapes
sphere = fps.sphere(1.5)
box = fps.box(1.0, 2.0, 1.0)
plane = fps.plane((0, 1, 0), 0)

# Rounded shapes
rounded_box = fps.rounded_box(1.0, 1.0, 1.0, 0.2)
round_cone = fps.round_cone(0.1, 1.0, 2.0)

# Cylinder-based shapes
cylinder = fps.cylinder(1.0, 2.0)
capsule = fps.capsule(2.0, 0.5)

# Curve-based shapes
line = fps.line_segment((-1, 0, 0), (1, 0, 0), 0.1)
curve = fps.quadratic_bezier((-1, 0, 0), (0, 1, 0), (1, 0, 0), 0.1)

# Specialized shapes
ellipsoid = fps.ellipsoid((1.0, 0.5, 0.5))
death_star = fps.death_star(1.0, 0.5, 1.2)
```

## Implementation Notes

All shapes are implemented in Python for consistency and extensibility. They use the basic operations from Fidget expressions to define the shape geometries.

### Creating Complex Shapes

Shapes can be combined using operators and operations from the `fidgetpy.ops` module:

```python
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.ops as fpo

# Create a basic shape
sphere = fps.sphere(1.0)

# Transform it
moved_sphere = sphere.translate((0, 1, 0))

# Combine with another shape
box = fps.box(1.5, 1.5, 1.5)
union = fpo.union(sphere, box)
smooth_union = fpo.smooth_union(sphere, box, 0.3)
```

### SDF Properties

The shapes in this module follow standard SDF conventions:
- Negative values are inside the shape
- Positive values are outside the shape
- The value represents the distance to the surface

Most shapes are "exact" SDFs, meaning they give the true Euclidean distance to the surface. A few shapes (like ellipsoid) are "bound" SDFs, which provide a good approximation but not the exact distance.

### Coordinate System

All shapes are centered at the origin by default. Use transformation operations to position them in space:

```python
# Create a sphere centered at (2, 0, 0)
sphere_at_2x = fps.sphere(1.0).translate((2, 0, 0))

# Or equivalently
sphere_at_2x = fpm.translate(fps.sphere(1.0), (2, 0, 0))
```

## References

Many of the shape implementations are based on the excellent collection of distance functions by Inigo Quilez:
[https://iquilezles.org/articles/distfunctions/](https://iquilezles.org/articles/distfunctions/)