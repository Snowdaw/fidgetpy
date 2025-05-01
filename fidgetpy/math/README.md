# Fidget Python Bindings - Math Module

This module provides mathematical operations for SDF expressions in the Fidget Python bindings.

## Module Structure

The math functions have been organized into logically grouped files:

- **basic_math.py**: Basic mathematical operations
  - min, max, clamp, abs, sign, floor, ceil, round, fract, mod, pow, sqrt, exp, ln

- **trigonometry.py**: Trigonometric functions
  - sin, cos, tan, asin, acos, atan, atan2

- **vector_math.py**: Vector operations
  - length, distance, dot, dot2, ndot, cross, normalize

- **transformations.py**: Transformation operations
  - translate, scale, rotate
  - remap_xyz, remap_affine
  - make_translation_matrix, make_scaling_matrix
  - make_rotation_x_matrix, make_rotation_y_matrix, make_rotation_z_matrix
  - combine_matrices

- **domain_manipulation.py**: Domain manipulation functions
  - repeat, mirror, symmetry

- **interpolation.py**: Interpolation functions
  - mix, lerp
  - smoothstep, smootherstep
  - step

- **logical.py**: Logical operations
  - logical_and, logical_or, logical_not, logical_xor
  - python_and, python_or, logical_if

## Module Organization

The math module has been designed to hide implementation details while providing a clean, user-friendly API. All functions are available directly from the `fidgetpy.math` namespace, while the individual module files are hidden from the user.

Example usage:repeat

```python
import fidgetpy as fp
import fidgetpy.math as fpm

# Basic math operations
result = fpm.min(a, b)
value = fpm.abs(x)

# Vector operations
length_value = fpm.length(vector)
normalized = fpm.normalize(vector)

# Transformations
transformed = fpm.translate(expr, tx, ty, tz)
rotated = fpm.rotate(expr, angle_x, angle_y, angle_z)

# Domain manipulations
repeated = fpm.repeat(expr, rx, ry, rz)
mirrored = fpm.mirror(expr, True, False, False)

# Interpolation
mixed = fpm.mix(a, b, t)
smooth = fpm.smoothstep(edge0, edge1, x)

# Logical operations
intersection = fpm.logical_and(a, b)  # Equivalent to a & b
union = fpm.logical_or(a, b)          # Equivalent to a | b
negation = fpm.logical_not(a)         # Equivalent to ~a
xor_result = fpm.logical_xor(a, b)    # Equivalent to (a | b) & ~(a & b)
conditional = fpm.logical_if(condition, true_value, false_value)  # if-then-else
```

## Implementation Notes

All math functions are implemented in Python for consistency and extensibility. They check for the presence of corresponding methods on SDF expressions and delegate to those when available, otherwise falling back to standard Python implementations.

### Function vs Method Calls

Most functions work both as standalone functions and as extension methods on SDF expressions:

```python
# As standalone functions
result = fpm.min(a, b)
translated = fpm.translate(expr, tx, ty, tz)

# As extension methods on expressions
result = a.min(b)
translated = expr.translate(tx, ty, tz)
```

However, some functions **cannot** be used as methods due to parameter order issues:

- **smoothstep**, **smootherstep**: These functions take parameters `(edge0, edge1, x)` where `x` is the expression, making `x.smoothstep(edge0, edge1)` unintuitive.
- **step**: This function takes parameters `(edge, x)` where `x` is the expression.

For these functions, only the standalone function call style is supported:

```python
# Correct - function style
result = fpm.smoothstep(0.0, 1.0, x)
stepped = fpm.step(0.5, x)

# NOT supported - would be confusing
# x.smoothstep(0.0, 1.0)  # NOT SUPPORTED
# x.step(0.5)  # NOT SUPPORTED
```

### Logical Operations

The module provides two sets of logical operations:

1. **Standard logical operations** (`logical_and`, `logical_or`, `logical_not`, `logical_xor`):
   - These operate directly on SDF values
   - Following SDF conventions: negative inside, positive outside
   - Implement CSG operations (intersection, union, etc.)

2. **Python-style logical operations** (`python_and`, `python_or`):
   - These implement Python's short-circuit logical evaluation
   - Useful when you want the exact behavior of Python's `and` and `or` operators
   - `a python_or b` returns `a` if `a` is truthy, otherwise `b`
   - `a python_and b` returns `a` if `a` is falsy, otherwise `b`

Example:

```python
# Standard logical operations - SDF operations
intersection = fpm.logical_and(sphere, box)  # Both sphere AND box
union = fpm.logical_or(sphere, box)  # Either sphere OR box

# Python-style logical operations - short-circuit evaluation
result1 = fpm.python_or(a, b)  # Returns a if a is truthy, otherwise b
result2 = fpm.python_and(a, b)  # Returns a if a is falsy, otherwise b
```