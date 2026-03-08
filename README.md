# FidgetPy: Python Bindings for Fidget

This project provides Python bindings for the [Fidget](https://github.com/mkeeter/fidget) library, allowing you to define and evaluate complex Signed Distance Field (SDF) expressions efficiently in Python.

## Features

*   Define SDF expressions using a Pythonic API (method chaining or functional style).
*   Supports standard mathematical operations (+, -, *, /, sqrt, sin, cos, abs, min, max, etc.).
*   Efficient bulk evaluation at many points simultaneously using NumPy arrays.
*   Automatic selection of JIT-compiled backend for performance, with fallback to a VM interpreter.
*   Create and use custom variables within expressions.
*   Named attribute containers (`fp.container`) for bundling geometry with color and other data.
*   Export meshes directly to colored PLY files (per-vertex RGB via SDF expressions).
*   Export to Gaussian Splatting `.ply` files.
*   Import/export in VM format and F-Rep format.

## Installation

This project uses [`maturin`](https://www.maturin.rs/) to build the Rust extension.

1.  **Set up a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    ```
2.  **Get dependencies:**
    ```bash
    pip install maturin numpy pytest
    git clone "https://github.com/mkeeter/fidget"
    ```
3.  **Build and install `fidgetpy` in editable mode:**
    ```bash
    cd fidgetpy
    maturin develop
    ```
    **Outside venv:**
    ```bash
    cd fidgetpy
    maturin build --interpreter /your/python/interpreter/python
    /your/python/interpreter/python -m pip install target/wheels/fidgetpy-*.whl --force-reinstall
    ```

## Basic Usage

```python
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm
import numpy as np

# Standard coordinate variables
x = fp.x()
y = fp.y()
z = fp.z()

# Build an SDF expression
sphere = fps.sphere(1.0)
box    = fps.box_exact(width=1.0, height=1.0, depth=1.0)

# Combine shapes
combined = fp.ops.smooth_union(sphere, box.translate(0.8, 0.0, 0.0), 0.2)

# Evaluate a plain SDF at points
pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
vals = fp.eval(combined, pts)   # x/y/z mapped automatically
print(vals)  # (N,) array of SDF distances

# Evaluate all attributes of a Container at once
fpc = fp.container()
fpc.shape = fps.sphere(1.0)
fpc.r = x * 0.5 + 0.5   # per-point expression
fpc.g = 0.2              # constant
results = fp.eval(fpc, pts)
# {'shape': array([...]), 'r': array([...]), 'g': array([...])}
print(results)
```

## Module Organisation

| Module | Contents |
|--------|----------|
| `fp` (root) | `x()`, `y()`, `z()`, `var()`, `eval()`, `mesh()`, `splat()`, `to_vm()`, `from_vm()`, `to_frep()`, `from_frep()`, `container()` |
| `fp.shape` | Primitive shapes: `sphere`, `box`, `cylinder`, `cone`, `torus`, … |
| `fp.ops` | Combining operations: `union`, `intersection`, `smooth_union`, `onion`, … |
| `fp.math` | Math helpers: `clamp`, `mix`, `atan2`, `hsl`, `translate`, `rotate`, … |

## Containers — Bundling Geometry with Color

A `Container` groups a geometry SDF with any number of named data attributes (colour r/g/b, PBR roughness/metallic, custom fields, etc.).

```python
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm

# fp.container() defaults to shape + r/g/b slots
fpc = fp.container()
fpc.shape = fps.sphere(1.0)
fpc.r = 0.9
fpc.g = 0.1
fpc.b = 0.1

# Custom slots
fpc2 = fp.container("shape", "roughness", "metallic")
fpc2.shape = fps.box_exact(1, 1, 1)
fpc2.roughness = 0.4
fpc2.metallic  = 0.8

# Add / remove attributes after construction
fpc.add("opacity")
fpc.opacity = 0.8
fpc.remove("opacity")
```

### Proximity Paint

Blend attribute values near an SDF region surface:

```python
dot = fps.sphere(0.2).translate(0.5, 0.5, 0.0)
fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.08)   # green dot
```

The blend weight is `clamp(1 - sdf / width, 0, 1)`, so it is 1 inside the region and fades to 0 at `width` units outside.

### Iterate over attributes

```python
for ch in fpc:
    print(ch.name, ch.value)
    vm_str = ch.to_vm()   # VM string for this attribute
```

## Meshing

`fp.mesh()` is the single mesh function:

- **No `output_file`** → returns a `PyMesh` object (`.vertices`, `.triangles`).
- **With `output_file`** → writes a binary PLY file and returns the path. Per-vertex colors are written when the input is a Container with `r`/`g`/`b` attributes.

```python
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm
import math

# Return PyMesh object
m = fp.mesh(fps.sphere(1.0), depth=5)
print(m.vertices.shape, m.triangles.shape)  # requires numpy=True for numpy arrays

# Write plain PLY (no vertex colors)
fp.mesh(fps.sphere(1.0), output_file="sphere.ply", depth=6)

# Write colored PLY from a Container with solid color
fpc = fp.container()
fpc.shape = fps.sphere(1.0)
fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1   # solid red
fp.mesh(fpc, output_file="red_sphere.ply", depth=6)

# Write colored PLY with procedural color (hue from angle around Z)
angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
rgb = fpm.hsl(angle, 1.0, 0.5)
fpc.r = rgb['r']; fpc.g = rgb['g']; fpc.b = rgb['b']
fp.mesh(fpc, output_file="color_wheel.ply", depth=7)

# Container method — same as fp.mesh(fpc, output_file=...)
fpc.mesh(output_file="sphere2.ply", depth=5)
```

### Viewing colored PLY in Blender

After importing the PLY file, vertex colors are stored but not displayed by default:
- In the viewport: *Viewport Shading* dropdown → **Color: Vertex**.
- In a material: add an *Attribute* node with name `Col` → connect to *Base Color*.

### Mesh parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 4 | Octree subdivision depth (higher = more detail, slower) |
| `numpy` | False | Return numpy arrays instead of Python lists |
| `threads` | True | Use multithreading |
| `bounds_min` | — | `[x, y, z]` minimum corner (preferred over center/scale) |
| `bounds_max` | — | `[x, y, z]` maximum corner (preferred over center/scale) |
| `center` | `[0,0,0]` | Center point (used when bounds not given) |
| `scale` | 1.0 | Half-extent (used when bounds not given) |
| `variables` | — | List of variable expressions for custom vars |
| `variable_values` | — | Corresponding values |

## Gaussian Splatting Export

`fp.splat()` works the same way as `fp.mesh()`:

- **No `output_file`** → returns a `Gaussians` object you can inspect and save later.
- **With `output_file`** → writes a 3DGS-compatible `.ply` immediately and returns the path.

```python
import fidgetpy as fp
import fidgetpy.shape as fps

# Return Gaussians object for inspection
g = fp.splat(fps.sphere(1.0))
print(g.count)            # number of Gaussians
print(g.positions.shape)  # (N, 3)
print(g.colors.shape)     # (N, 3), linear RGB in [0, 1]
g.save("sphere.ply")      # write when ready

# Write directly
fp.splat(fps.sphere(1.0), output_file="sphere.ply")

# Container with color
fpc = fp.container()
fpc.shape = fps.sphere(1.0)
fpc.r = 0.2; fpc.g = 0.6; fpc.b = 1.0   # light blue
fp.splat(fpc, output_file="blue_sphere.ply")

# Container method
fpc.splat(output_file="blue_sphere2.ply")
```

Key parameter: `size` controls the sampling grid resolution (`size³` points, default 96).

The output can be loaded in any Gaussian Splatting viewer (e.g. [SuperSplat](https://supersplat.vercel.app/)).

## Import / Export (VM and F-Rep)

`fp.to_vm()` and `fp.to_frep()` follow the same pattern as `fp.mesh()` and `fp.splat()`:

- **No `output_file`** → return text (str for a plain SDF, dict for a Container).
- **With `output_file`** → write file(s) to disk and return the path(s).

For a Container, one file is written per set attribute using `<stem>_<attr><ext>`.

```python
import fidgetpy as fp
import fidgetpy.shape as fps

sphere = fps.sphere(1.0)

# VM format — return string
vm_str = fp.to_vm(sphere)
reimported = fp.from_vm(vm_str)

# VM format — write to file
fp.to_vm(sphere, output_file="sphere.vm")

# F-Rep format — return string
frep_str = fp.to_frep(sphere)
reimported2 = fp.from_frep(frep_str)

# F-Rep format — write to file
fp.to_frep(sphere, output_file="sphere.frep")

# Container — return dicts
fpc = fp.container()
fpc.shape = fps.sphere(1.0)
fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

vms   = fp.to_vm(fpc)    # {'shape': '...', 'r': '...', 'g': '...', 'b': '...'}
freps = fp.to_frep(fpc)  # same structure for F-Rep

# Container — write one file per attribute
fp.to_vm(fpc,   output_file="model.vm")
# writes: model_shape.vm, model_r.vm, model_g.vm, model_b.vm

fp.to_frep(fpc, output_file="model.frep")
# writes: model_shape.frep, model_r.frep, model_g.frep, model_b.frep
```

Both `fp.to_vm()` and `fp.to_frep()` also work on Container objects directly as methods (returning dicts of strings, without file output).

## Custom Variables

```python
import fidgetpy as fp
import fidgetpy.shape as fps

radius_var = fp.var("radius")
cylinder = fps.cylinder(radius_var, 2.0)

# Mesh with variable bound to a value
variables = [fp.x(), fp.y(), fp.z(), radius_var]
variable_values = [0.0, 0.0, 0.0, 1.5]

m = fp.mesh(cylinder,
            bounds_min=[-3, -3, -3],
            bounds_max=[ 3,  3,  3],
            depth=5,
            variables=variables,
            variable_values=variable_values)
```
