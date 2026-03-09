# FidgetPy

Python bindings for [Fidget](https://github.com/mkeeter/fidget) — a library for fast, JIT-compiled Signed Distance Field (SDF) evaluation.

## Installation

```bash
pip install maturin numpy
git clone "https://github.com/mkeeter/fidget"
cd fidgetpy && maturin develop
```

## Quick start

```python
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm

# Build shapes from primitives and combine them
sphere = fps.sphere(1.0)
box    = fps.box(1.0, 1.0, 1.0)
scene  = fp.ops.smooth_union(sphere, box.translate(0.8, 0.0, 0.0), 0.2)

# Mesh it — returns a PyMesh with .vertices and .triangles
m = fp.mesh(scene, depth=6)

# Or write straight to PLY
fp.mesh(scene, output_file="scene.ply", depth=6)
```

## Modules

| Module | What's in it |
|--------|-------------|
| `fp` | `x/y/z()`, `var()`, `eval()`, `mesh()`, `splat()`, `to_vm/frep()`, `from_vm/frep()`, `container()` |
| `fp.shape` (`fps`) | Primitives: `sphere`, `box`, `cylinder`, `torus`, `cone`, `capsule`, … |
| `fp.ops` (`fpo`) | Boolean and blending ops: `union`, `intersection`, `smooth_union`, `onion`, … |
| `fp.math` (`fpm`) | Math helpers: `clamp`, `mix`, `sin`, `atan2`, `hsl`, `translate`, `rotate`, `gradient`, `normal`, `diffuse`, … |

## Containers — expressions as attributes

A `Container` bundles a shape SDF with named per-point data. Useful for adding color or any other extra evaluation data you might need.

```python
fpc = fp.container() # Defaults to have: shape, r, g, and b 
fpc.shape = fps.sphere(1.0)
fpc.r = 0.9          # constant
fpc.g = fp.y() * 0.5 + 0.5  # expression — varies per point
fpc.b = 0.1

# Proximity paint: blend color near a region
dot = fps.sphere(0.2).translate(0.5, 0.5, 0.0)
fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.08)

# Mesh with per-vertex color
fpc.mesh(output_file="colored.ply", depth=6)

# Gaussian splat with color
fpc.splat(output_file="colored.ply")

# Evaluate all attributes at points
import numpy as np
pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
results = fpc.eval(pts)  # {'shape': array([...]), 'r': array([...]), ...}
```

## Meshing and splatting

Both `fp.mesh()` and `fp.splat()` follow the same pattern:

- **No `output_file`** → return an in-memory object (`PyMesh` or `Gaussians`).
- **With `output_file`** → write a `.ply` file and return the path.

```python
# Mesh
m = fp.mesh(fps.sphere(1.0), depth=5, numpy=True)
print(m.vertices.shape, m.triangles.shape)

# Gaussian splat
g = fp.splat(fps.sphere(1.0))
print(g.count, g.positions.shape, g.colors.shape)
g.save("sphere.ply")
```

Key parameters for `fp.mesh()`: `depth` (default 4), `bounds_min`/`bounds_max` or `center`/`scale`, `numpy`, `threads`.
Key parameters for `fp.splat()`: `size` — sampling grid resolution (`size³` points, default 96); `bounds_min`/`bounds_max` — explicit sampling volume (auto-detected if omitted).

## VM and F-Rep serialisation

```python
sphere = fps.sphere(1.0)

vm_str = fp.to_vm(sphere)          # → str
fp.to_vm(sphere, output_file="sphere.vm")

frep_str = fp.to_frep(sphere)      # → str
fp.to_frep(sphere, output_file="sphere.frep")

# Containers write one file per attribute: model_shape.vm, model_r.vm, …
fpc = fp.container(); fpc.shape = sphere; fpc.r = 0.9
fp.to_vm(fpc, output_file="model.vm")

reimported = fp.from_vm(vm_str)
```

## Shading

**`fp.math`** — symbolic shading, composes directly with other SDF expressions:

```python
import fidgetpy.math as fpm

shape = fps.sphere(1.0)

# Surface normal components in [-1, 1] — useful for normal-map visualization
nx, ny, nz = fpm.normal(shape)
fpc.r = nx * 0.5 + 0.5

# Lambertian diffuse — returns an SDF expression, free to compose
light = fpm.diffuse(shape, light_dir=(1, 2, 3))
fpc.r = 0.8 * light
```

**`fp.eval_grad`** — exact surface normals and SDF values at explicit points,
via forward-mode automatic differentiation (one JIT pass):

```python
import numpy as np

m = fp.mesh(shape, depth=6)

# Returns (N, 4): [sdf_value, dx, dy, dz]
grad = fp.eval_grad(shape, m.vertices.astype(np.float32))
normals   = grad[:, 1:]                          # (N, 3) exact normals
brightness = np.clip(normals @ [0.6, 0.8, 0.0], 0, 1)
m.save("diffuse.ply", colors=np.stack([brightness]*3, axis=1))
```

## Custom variables

```python
radius_var = fp.var("radius")
shape = fps.cylinder(radius_var, 2.0)

fp.mesh(shape,
        bounds_min=[-3, -3, -3], bounds_max=[3, 3, 3],
        depth=5,
        variables=[fp.x(), fp.y(), fp.z(), radius_var],
        variable_values=[0.0, 0.0, 0.0, 1.5])
```
