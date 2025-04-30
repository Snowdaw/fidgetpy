# Fidget Operations Module

The `fidgetpy.ops` module provides operations for combining and manipulating SDF (Signed Distance Field) expressions in the Fidget Python bindings. These operations allow you to create complex shapes by combining, blending, and transforming simpler shapes.

## Key Differences from Math Module

Unlike the `fidgetpy.math` module, operations in this module:

1. **Can only be used as direct function calls** - They do not support method call syntax
2. **Focus on shape operations** rather than mathematical functions
3. **Operate primarily on SDF expressions** rather than general numeric values

## Module Organization

The operations are organized into several categories:

### Boolean Operations (`boolean_ops.py`)

Basic and smooth boolean operations for combining shapes:

- `union`, `intersection`, `difference` - Basic boolean operations
- `smooth_union`, `smooth_intersection`, `smooth_difference` - Smooth boolean operations
- `complement`, `boolean_and`, `boolean_or`, `boolean_not`, `boolean_xor` - Additional boolean operations

### Blending Operations (`blending_ops.py`)

Operations for blending between shapes:

- `blend`, `mix`, `lerp` - Linear interpolation between shapes
- `smooth_min`, `smooth_max` - Polynomial smooth min/max
- `exponential_smooth_min`, `exponential_smooth_max` - Exponential smooth min/max
- `power_smooth_min`, `power_smooth_max` - Power-based smooth min/max
- `soft_clamp`, `quad_bezier_blend` - Additional blending utilities

### Domain Operations (`domain_ops.py`)

Operations that modify the domain of SDF expressions:

- `onion`, `shell` - Create shells around shapes
- `elongate` - Stretch shapes along axes
- `twist`, `bend` - Apply deformations to shapes
- `round`, `displace` - Modify shape surfaces
- `mirror_x`, `mirror_y`, `mirror_z` - Mirror shapes across planes

### Miscellaneous Operations (`misc_ops.py`)

Additional utility operations:

- `smooth_step_union`, `chamfer_union`, `chamfer_intersection` - Special union/intersection types
- `engrave` - Create engravings in shapes
- `extrusion`, `revolution` - Create 3D shapes from 2D profiles
- `repeat`, `repeat_limited` - Create repeated patterns of shapes
- `weight_blend` - Blend multiple shapes with weights

## Usage Example

```python
import fidgetpy as fp
import fidgetpy.ops as fpo

# Create basic shapes
sphere = fp.shape.sphere(1.0)
box = fp.shape.box(1.0, 1.0, 1.0)

# Combine shapes with operations
union_shape = fpo.union(sphere, box)
smooth_union_shape = fpo.smooth_union(sphere, box, 0.2)
difference_shape = fpo.difference(box, sphere)

# Apply domain operations
twisted_box = fpo.twist(box, 1.0)
repeated_spheres = fpo.repeat(sphere, (3.0, 3.0, 3.0))

# Create a mesh from the result
mesh = fp.mesh.generate(smooth_union_shape, resolution=64)
```

## Documentation Style

All operations in this module follow a consistent documentation style as outlined in the [DOCUMENTATION_STYLE_GUIDE.md](./DOCUMENTATION_STYLE_GUIDE.md) file.

## Important Note

Remember that operations can only be used as direct function calls. For example:

```python
# Correct usage:
result = fpo.union(shape1, shape2)

# Incorrect usage (will not work):
result = shape1.union(shape2)  # Error! Operations don't support method call syntax