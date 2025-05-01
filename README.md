# FidgetPy: Python Bindings for Fidget

This project provides Python bindings for the [Fidget](https://github.com/mkeeter/fidget) library, allowing you to define and evaluate complex expressions of implicit surfaces efficiently in Python.

## Features

*   Define expressions using a Pythonic API (method chaining or functional style).
*   Supports standard mathematical operations (+, -, *, /, sqrt, sin, cos, abs, min, max, etc.).
*   Includes operations like `smooth_union` and `translate`.
*   Efficient bulk evaluation at many points simultaneously using NumPy arrays.
*   Automatic selection of JIT (Just-In-Time) compiled backend for performance when available, with fallback to a VM interpreter.
*   Create and use custom variables within expressions.

## Installation

This project uses [`maturin`](https://www.maturin.rs/) to build the Rust extension.

1.  **Set up a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```
2.  **Get dependencies:**
    ```bash
    pip install maturin numpy pytest
    git clone "https://github.com/mkeeter/fidget"
    ```
3.  **Build and install `fidgetpy` in editable mode:**
    ```bash
    # Navigate to the fidgetpy directory
    cd fidgetpy
    maturin develop
    ```

Note that when running the tests with pytest in order for the shape tests to work you need to build Fidget with the fidget-cli demo as it is used for comparing meshes.

## Basic Usage

```python
import fidgetpy as fp

# --- Define Variables (Identity + Name) ---
# Standard axes (var objects with names 'x', 'y', 'z')
x = fp.x()
y = fp.y()
z = fp.z()

# Custom variable (var object with name 'radius')
radius = fp.var(name='radius')

# --- Build Expressions (using var objects directly) ---
# Var objects are implicitly promoted to expressions
sphere_core = x*x + y*y + z*z
sphere = sphere_core.sqrt() - radius # Sphere radius controlled by 'radius' variable

# Create a box using methods directly (optional style)
box_core = fp.x().abs().max(fp.y().abs()).max(fp.z().abs())
box = box_core - 0.5 # Box size 1 (0.5 is implicitly promoted)

# --- Transformations & Combinations ---
# Translate the sphere
translated_sphere = sphere.translate(1.5, 0.0, 0.0)

# Smoothly combine the translated sphere and the box
smoothness = 0.2
combined_shape = fp.ops.smooth_union(translated_sphere, box, smoothness)

# Print the expression structure (uses names from var objects)
print(f"Combined Shape: {combined_shape}")
# Example Output: add(mul(0.500, sub(add(sub(sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z))), radius), sub(max(max(abs(x), abs(y)), abs(z)), 0.500)), sqrt(add(square(sub(sub(sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z))), radius), sub(max(max(abs(x), abs(y)), abs(z)), 0.500))), square(0.200))))), mul(0.500, sub(add(sub(sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z))), radius), sub(max(max(abs(x), abs(y)), abs(z)), 0.500)), sqrt(add(square(sub(sub(sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z))), radius), sub(max(max(abs(x), abs(y)), abs(z)), 0.500))), square(0.200))))))

# --- Evaluation ---
# Define points as a list of lists or NumPy array (N, num_vars)
# Columns must match the order of variables provided later!
points = [
    # x,   y,   z, radius
    [0.0, 0.0, 0.0, 1.0],  # Origin, radius 1 -> dist = -1.0 (inside sphere)
    [1.5, 0.0, 0.0, 1.0],  # Center of translated sphere, radius 1 -> dist = -1.0
    [0.5, 0.0, 0.0, 1.0],  # Near box surface, radius 1
    [3.0, 4.0, 0.0, 1.0],  # Far away, radius 1
]

# Define the list of *original var instances* corresponding to the columns
# The order MUST match the columns in 'points'!
variables_for_eval = [x, y, z, radius] # Use the var objects

# Evaluate the by calling .eval() on the expression
# Backend defaults to 'jit' if available, otherwise 'vm'
distances = fp.eval(combined_shape, points, variables_for_eval)

# Explicitly request VM backend
# distances_vm = combined_shape.eval(points, variables_for_eval, backend='vm')

print("\nEvaluation Points (x, y, z, radius):")
print(points)
print("\nDistances:")
print(distances)

```

## API Consistency & Error Messages

*   **Consistency:** The API offers both method chaining (`expr.sqrt()`) and functional (`fp.sqrt(expr)`) styles for most operations. Naming generally follows standard mathematical conventions.
*   **Error Messages:** Efforts have been made to provide informative Python `ValueError` exceptions for issues like incorrect input shapes, invalid backend names, or invalid context handles. Errors originating from the core Fidget library (like `MismatchedSlices`) are also propagated with their original details.

## Module Organization

The library is organized into several modules for better organization:

* **math:** Mathematical operations for expressions
  * Basic math functions (min, max, clamp, abs, etc.)
  * Trigonometric functions (sin, cos, tan, etc.)
  * Vector math (length, dot, cross, etc.)
  * Transformations (translate, scale, rotate, etc.)
  * Domain manipulation (repeat, mirror, etc.)
  * Interpolation functions (mix, smoothstep, etc.)
  * Logical operations (logical_and, logical_or, etc.)

* **ops:** specific operations
  * Boolean operations (union, intersection, etc.)
  * Blending operations (smooth_union, smooth_intersection, etc.)
  * Domain operations (onion, elongate, etc.)

 * **shape:** basic shapes
   * Primitive shapes (sphere, box, etc.)
   * Cylinder shapes (cylinder, cone, etc.)
   * Curve shapes (polyline, quadratic bezier, etc.)



## Import/Export Functionality

Fidget-Py provides functionality to import and export expressions in both VM format (Fidget's native format) and F-Rep (a human-readable functional representation format).

### VM Format

VM format is a low-level representation of operations used by Fidget internally. It's a simple text format where each line defines a variable, a constant, or an operation.

```python
import fidgetpy as fp

# Create an expression
x, y, z = fp.x(), fp.y(), fp.z()
sphere = ((x**2 + y**2 + z**2)**0.5) - 1.0  # Sphere with radius 1

# Convert to VM format
vm_text = fp.to_vm(sphere)
print(vm_text)
# Output:
# # Fidget VM format export
# # Generated by fidget-py
# _0 var-x
# _1 var-y
# _2 var-z
# _3 mul _0 _0
# _4 mul _1 _1
# _5 mul _2 _2
# _6 add _3 _4
# _7 add _6 _5
# _8 sqrt _7
# _9 const 1
# _a sub _8 _9

# Import from VM format
imported_expr = fp.from_vm(vm_text)
```

### F-Rep Format

F-Rep is a more human-readable functional representation format. It represents the expression as nested function calls.

```python
import fidgetpy as fp

# Create an expression
x, y, z = fp.x(), fp.y(), fp.z()
sphere = ((x**2 + y**2 + z**2)**0.5) - 1.0  # Sphere with radius 1

# Convert to F-Rep format
frep_text = fp.to_frep(sphere)
print(frep_text)
# Output: sub(sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z))), 1.000)

imported_expr = fp.from_frep(frep_text)
```

See the `vm_frep_example.py` file for more detailed examples of using these functions.

## Meshing Functionality

Fidget-Py provides functionality to convert expressions into triangle meshes using the `fp.mesh()` function. This is useful for visualization and exporting to 3D modeling software.

```python
import fidgetpy as fp
import fidgetpy.shape as fps

# Create a sphere with radius 1.0
sphere = fps.sphere(1.0)

# Generate a mesh
mesh = fp.mesh(sphere, bounds_min=[-1.2, -1.2, -1.2], bounds_max=[1.2, 1.2, 1.2])

# Access vertices and triangles
vertices = mesh.vertices
triangles = mesh.triangles

# Save as STL file
fp.save_stl(mesh, "sphere.stl")
```

### Mesh Parameters

The `fp.mesh()` function supports several parameters to control the meshing process:

* **bounds_min**: A list or tuple `[min_x, min_y, min_z]` specifying the minimum corner of the bounding box for meshing. This is the preferred way to define the meshing volume.
* **bounds_max**: A list or tuple `[max_x, max_y, max_z]` specifying the maximum corner of the bounding box for meshing. This is the preferred way to define the meshing volume.
* **depth**: Control the resolution of the mesh. Higher values produce more detailed meshes but take longer to generate. Default is 4.
* **numpy**: If `True`, returns vertices and triangles as NumPy arrays. If `False` (default), returns them as Python lists.
* **threads**: If `True` (default), uses multi-threading for mesh generation. Set to `False` for single-threaded operation.
* **variables** and **variable_values**: Support for custom variables in the expression.
* **center**: (Alternative) Specify the center point for meshing as `[x, y, z]`. Default is `[0, 0, 0]`. Ignored if `bounds_min` and `bounds_max` are provided.
* **scale**: (Alternative) Scale factor for the result. Default is 1.0. Ignored if `bounds_min` and `bounds_max` are provided. Defines the default bounds as `[-scale, -scale, -scale]` to `[scale, scale, scale]` relative to `center`.

### Meshing with Custom Variables

You can mesh expressions that contain custom variables by providing the variables and their values:

```python
import fidgetpy as fp
import fidgetpy.shape as fps

# Create a shape with custom variables
radius_var = fp.var("radius")
height_var = fp.var("height")

# Create a cylinder with variable radius and height
cylinder = fps.cylinder(radius_var, height_var)

# Define the variables and their values
variables = [fp.x(), fp.y(), fp.z(), radius_var, height_var]
variable_values = [0.0, 0.0, 0.0, 1.0, 2.0]  # x=0, y=0, z=0, radius=1.0, height=2.0

# Generate the mesh
mesh = fp.mesh(cylinder,
               bounds_min=[-3,-3,-3],
               bounds_max=[3,3,3],
               depth=5,
               variables=variables,
               variable_values=variable_values)
```

## Exporting Meshes

The `fp.save_stl()` function allows you to export meshes to the STL file format, which is widely supported by 3D printing software and CAD tools.

```python
import fidgetpy as fp
import fidgetpy.shape as fps

# Create a shape
shape = fps.box(1.0, 1.0, 1.0)

# Generate a mesh
mesh = fp.mesh(shape, bounds_min=[-3,-3,-3], bounds_max=[3,3,3], depth=5)

# Save as STL file
fp.save_stl(mesh, "box.stl")
```

The `save_stl()` function works with both Python list and NumPy array mesh representations, so you can use it regardless of the `numpy` parameter you used when generating the mesh.
