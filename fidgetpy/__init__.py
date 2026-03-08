"""
Fidget Python bindings - A Pythonic interface to the Fidget SDF library.

This module provides a clean, Pythonic API for creating and manipulating
Signed Distance Fields (SDFs) using the Fidget library.

Core functions:
- x(), y(), z(): Coordinate variables
- var(name): Custom variables
- eval(expr, points): Evaluate expressions at points
- mesh(expr, numpy=False): Generate meshes from expressions, returns a Mesh object with attributes:
  * vertices: Points in 3D space (list or numpy array based on 'numpy' parameter)
  * triangles: Triangle indices (list or numpy array based on 'numpy' parameter)
- save_stl(mesh, filepath): Save a Mesh object to an STL file
- vm(expr_or_container): Export SDF or Container to VM string(s)
- splat(expr_or_container, ...): Convert SDF to Gaussian Splatting .ply
- container(*names): Create a Container of named SDF attributes

Submodules:
- shape: Common shape primitives (sphere, box, cylinder, etc.)
- ops: Operations for combining shapes (smooth_union, etc.)
- math: Math operations and transformations
"""

# Import core functions from the Rust module
from fidgetpy.fidgetpy import (
    # Core variables
    x, y, z, var,

    # Evaluation and meshing
    eval as _eval_rust, mesh as _mesh_rust, save_stl,

    # Import/Export
    from_vm, to_vm as _to_vm_rust, from_frep, to_frep,
)

# Import submodules
from . import shape
from . import ops
from . import math
from . import splat as _splat_module
from .expr import Container

# ── Python wrappers that understand Container ──────────────────────────────────

def vm(expr_or_container):
    """
    Export a fidgetpy SDF expression or Container to VM format.

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.

    Returns:
        str:  VM string (plain SDF expression).
        dict: {name: vm_string} for every set attribute (Container).

    Examples:
        # Plain SDF → string
        vm_str = fp.vm(fps.sphere(1.0))

        # Container → dict
        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1
        vms = fp.vm(fpc)        # {'shape': '...', 'r': '...', ...}
    """
    if isinstance(expr_or_container, Container):
        return expr_or_container.vm()
    return _to_vm_rust(expr_or_container)


# Keep to_vm as an alias for backward compatibility (hidden from main API docs)
to_vm = vm


def eval(expr, points, variables=None, **kwargs):
    """
    Evaluate a fidgetpy SDF expression at a set of points.

    Args:
        expr:      A fidgetpy SDF expression or Container (uses 'shape').
        points:    (N, 3) array of xyz coordinates.
        variables: List of variable expressions mapping columns → variables.
                   Can be passed positionally or as a keyword argument.
        **kwargs:  Additional keyword arguments passed to the Rust eval function.

    Returns:
        (N,) array of SDF values.
    """
    if isinstance(expr, Container):
        shape = expr._attrs.get('shape')
        if shape is None:
            raise ValueError(
                "Container has no 'shape' attribute. "
                "Set it with: fpc.shape = some_sdf"
            )
        expr = shape
    if variables is not None:
        return _eval_rust(expr, points, variables=variables, **kwargs)
    return _eval_rust(expr, points, **kwargs)


def mesh(expr, **kwargs):
    """
    Generate a mesh from a fidgetpy SDF expression or Container.

    Args:
        expr:     A fidgetpy SDF expression or Container (uses 'shape' attribute).
        **kwargs: Passed to the Rust mesh function (e.g. depth=6, numpy=True).

    Returns:
        A Mesh object with .vertices and .triangles.
    """
    if isinstance(expr, Container):
        shape = expr._attrs.get('shape')
        if shape is None:
            raise ValueError(
                "Container has no 'shape' attribute. "
                "Set it with: fpc.shape = some_sdf"
            )
        expr = shape
    return _mesh_rust(expr, **kwargs)


def splat(expr_or_container, output_file="gaussians.ply", color=None, **kwargs):
    """
    Convert a fidgetpy SDF expression or Container to a Gaussian Splatting .ply file.

    When a Container is passed, the 'shape' attribute is used for geometry and
    'r', 'g', 'b' attributes (if set) are used for color.  An explicit 'color'
    argument overrides any color stored in the Container.

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       Path to the output .ply file. Default: "gaussians.ply".
        color:             Color override. See fp.splat module for accepted forms.
        **kwargs:          Passed to the underlying splat function.

    Returns:
        str: Path to the written .ply file.

    Examples:
        import fidgetpy as fp
        import fidgetpy.shape as fps

        # Plain SDF, white
        fp.splat(fps.sphere(1.0), output_file="sphere.ply")

        # Container with color attributes
        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.2; fpc.g = 0.6; fpc.b = 1.0
        fp.splat(fpc)
    """
    if isinstance(expr_or_container, Container):
        shape_attr = expr_or_container._attrs.get('shape')
        if shape_attr is None:
            raise ValueError(
                "Container has no 'shape' attribute. "
                "Set it with: fpc.shape = some_sdf"
            )
        if color is None:
            r = expr_or_container._attrs.get('r')
            g = expr_or_container._attrs.get('g')
            b = expr_or_container._attrs.get('b')
            color = (
                1.0 if r is None else r,
                1.0 if g is None else g,
                1.0 if b is None else b,
            )
        expr_or_container = shape_attr

    return _splat_module.splat(
        expr_or_container,
        output_file=output_file,
        color=color,
        **kwargs,
    )


def mesh_ply(expr_or_container, output_file="mesh.ply", **kwargs):
    """
    Mesh a fidgetpy SDF expression or Container and write a colored PLY file.

    When a Container is passed, the 'shape' attribute provides the geometry
    and 'r', 'g', 'b' attributes (floats or fidgetpy expressions) are evaluated
    at every mesh vertex to produce per-vertex colors.  Unset channels default
    to 1.0 (white).

    For a plain SDF expression the mesh is written without vertex colors.

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       Path to the output .ply file.  Default: "mesh.ply".
        **kwargs:          Passed to fp.mesh() (e.g. depth=6).

    Returns:
        str: Path to the written .ply file.

    Examples:
        import fidgetpy as fp
        import fidgetpy.shape as fps

        # Plain SDF — no vertex colors
        fp.mesh_ply(fps.sphere(1.0), "sphere.ply", depth=6)

        # Container — per-vertex colors from r/g/b attributes
        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1   # solid red
        fp.mesh_ply(fpc, "red_sphere.ply", depth=6)

        # Procedural color: hue from angle around Z axis
        import math
        import fidgetpy.math as fpm
        angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
        rgb = fpm.hsl(angle, 1.0, 0.5)
        fpc.r = rgb['r']; fpc.g = rgb['g']; fpc.b = rgb['b']
        fp.mesh_ply(fpc, "color_wheel.ply", depth=7)
    """
    import numpy as np
    from fidgetpy.splat import _eval_sdf_at

    color_channels = None

    if isinstance(expr_or_container, Container):
        shape = expr_or_container._attrs.get('shape')
        if shape is None:
            raise ValueError(
                "Container has no 'shape' attribute. "
                "Set it with: fpc.shape = some_sdf"
            )
        r_attr = expr_or_container._attrs.get('r')
        g_attr = expr_or_container._attrs.get('g')
        b_attr = expr_or_container._attrs.get('b')
        has_color = any(v is not None for v in (r_attr, g_attr, b_attr))
        if has_color:
            color_channels = (
                1.0 if r_attr is None else r_attr,
                1.0 if g_attr is None else g_attr,
                1.0 if b_attr is None else b_attr,
            )
        expr_or_container = shape

    # Always mesh with numpy arrays for easy vertex access
    m = _mesh_rust(expr_or_container, numpy=True, **kwargs)
    verts = np.asarray(m.vertices, dtype=np.float32)   # (N, 3)
    tris  = np.asarray(m.triangles, dtype=np.int32)    # (M, 3)
    N = len(verts)

    # Evaluate per-vertex colors
    colors_u8 = None
    if color_channels is not None:
        colors_f = np.ones((N, 3), dtype=np.float64)
        for i, ch in enumerate(color_channels):
            if isinstance(ch, (int, float)):
                colors_f[:, i] = np.clip(float(ch), 0.0, 1.0)
            else:
                colors_f[:, i] = np.clip(_eval_sdf_at(ch, verts), 0.0, 1.0)
        colors_u8 = (colors_f * 255.0).astype(np.uint8)

    _write_mesh_ply(output_file, verts, tris, colors_u8)
    return output_file


def _write_mesh_ply(path, vertices, triangles, colors_u8=None):
    """
    Write a triangle mesh to a binary PLY file.

    Args:
        path:       Output file path.
        vertices:   (N, 3) float32 array of xyz positions.
        triangles:  (M, 3) int32 array of vertex indices.
        colors_u8:  (N, 3) uint8 array of RGB colors, or None for no color.
    """
    import numpy as np

    N = len(vertices)
    M = len(triangles)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors_u8 is not None:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines += [
        f"element face {M}",
        "property list uchar int vertex_indices",
        "end_header",
        "",  # trailing newline
    ]
    header = "\n".join(header_lines).encode("ascii")

    # Vertex buffer
    if colors_u8 is not None:
        vdtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        vbuf = np.zeros(N, dtype=vdtype)
        vbuf['x'] = vertices[:, 0]
        vbuf['y'] = vertices[:, 1]
        vbuf['z'] = vertices[:, 2]
        vbuf['red']   = colors_u8[:, 0]
        vbuf['green'] = colors_u8[:, 1]
        vbuf['blue']  = colors_u8[:, 2]
    else:
        vdtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        vbuf = np.zeros(N, dtype=vdtype)
        vbuf['x'] = vertices[:, 0]
        vbuf['y'] = vertices[:, 1]
        vbuf['z'] = vertices[:, 2]

    # Face buffer: each face = [count=3, v0, v1, v2]
    fdtype = np.dtype([('count', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
    fbuf = np.zeros(M, dtype=fdtype)
    fbuf['count'] = 3
    fbuf['v0'] = triangles[:, 0]
    fbuf['v1'] = triangles[:, 1]
    fbuf['v2'] = triangles[:, 2]

    with open(path, 'wb') as f:
        f.write(header)
        f.write(vbuf.tobytes())
        f.write(fbuf.tobytes())

    color_note = " with vertex colors" if colors_u8 is not None else ""
    print(f"  Wrote {N:,} vertices, {M:,} triangles{color_note} → {path}")


def container(*names):
    """
    Create a Container with declared (empty) attribute slots.

    Args:
        *names: String names for the attribute slots.  All values start as None;
                assign them with fpc.name = value.

    Returns:
        A Container instance.

    Example:
        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9
        fpc.g = 0.1
        fpc.b = 0.1
    """
    return Container(*names)


# ── Expression extension ───────────────────────────────────────────────────────

import inspect as _inspect


def _extend_expressions():
    """
    Add methods to the SDF expression class.
    Extends the class with functions from the math module, plus .vm().
    """
    expr_class = type(x())

    # Mark expressions so other code can detect them
    setattr(expr_class, '_is_sdf_expr', True)

    # Add math module functions as methods
    math_functions = _inspect.getmembers(math, _inspect.isfunction)
    for name, func in math_functions:
        if name.startswith('_'):
            continue
        if hasattr(expr_class, name):
            continue
        if name in ('step', 'smoothstep', 'smootherstep', 'pulse'):
            continue

        def create_method(math_func):
            def method(self, *args, **kwargs):
                return math_func(self, *args, **kwargs)
            return method

        setattr(expr_class, name, create_method(func))

    # Add .vm() method — export this single expression as a VM string
    def _sdf_vm(self):
        """Return the VM string for this SDF expression."""
        return _to_vm_rust(self)

    setattr(expr_class, 'vm', _sdf_vm)


# Initialize expression extensions when the module is imported
_extend_expressions()
