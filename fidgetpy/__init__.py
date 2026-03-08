"""
Fidget Python bindings - A Pythonic interface to the Fidget SDF library.

This module provides a clean, Pythonic API for creating and manipulating
Signed Distance Fields (SDFs) using the Fidget library.

Core functions:
- x(), y(), z(): Coordinate variables
- var(name): Custom variables
- eval(expr, points): Evaluate expression(s) at points.
  For a Container, evaluates every set attribute and returns a dict.
- mesh(expr, output_file=None, ...): Generate mesh.
  output_file=None → returns PyMesh; set → writes colored PLY, returns path.
- to_vm(expr_or_container, output_file=None): Export to VM format.
  output_file=None → returns str or dict; set → writes file(s), returns path(s).
- from_vm(text): Import SDF from VM string.
- to_frep(expr_or_container, output_file=None): Export to F-Rep format.
  Same output_file semantics as to_vm.
- from_frep(text): Import SDF from F-Rep string.
- splat(expr_or_container, output_file=None, ...): Gaussian Splatting.
  output_file=None → returns Gaussians; set → writes .ply, returns path.
- container(*names): Create a Container of named SDF attributes.

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
    eval as _eval_rust, mesh as _mesh_rust,

    # Import/Export
    from_vm, to_vm as _to_vm_rust, from_frep, to_frep as _to_frep_rust,
)

# Import submodules
from . import shape
from . import ops
from . import math
from . import splat as _splat_module
from ._expr import Container as _Container

# ── Python wrappers that understand Container ──────────────────────────────────

def to_vm(expr_or_container, output_file=None):
    """
    Export a fidgetpy SDF expression or Container to VM format.

    When output_file is None, returns the VM text (str for a plain SDF,
    dict {name: vm_string} for a Container).

    When output_file is provided, writes the text to disk and returns the
    path(s).  For a Container, one file is written per set attribute using
    the pattern ``<stem>_<attr><ext>`` (e.g. ``model_shape.vm``).

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       File path, or None to return text. Default: None.

    Returns:
        str:  VM string or written path (plain SDF).
        dict: {name: vm_string} or {name: written_path} (Container).

    Examples:
        # Return string
        vm_str = fp.to_vm(fps.sphere(1.0))

        # Write to file
        fp.to_vm(fps.sphere(1.0), output_file="sphere.vm")

        # Container → dict of strings
        vms = fp.to_vm(fpc)

        # Container → write one file per attribute
        fp.to_vm(fpc, output_file="model.vm")
        # → model_shape.vm, model_r.vm, model_g.vm, model_b.vm
    """
    if isinstance(expr_or_container, _Container):
        vm_dict = expr_or_container.to_vm()
        if output_file is None:
            return vm_dict
        from pathlib import Path
        p = Path(output_file)
        result = {}
        for name, vm_str in vm_dict.items():
            attr_path = p.parent / f"{p.stem}_{name}{p.suffix}"
            attr_path.write_text(vm_str)
            result[name] = str(attr_path)
        return result
    vm_str = _to_vm_rust(expr_or_container)
    if output_file is None:
        return vm_str
    from pathlib import Path
    Path(output_file).write_text(vm_str)
    return output_file


def to_frep(expr_or_container, output_file=None):
    """
    Export a fidgetpy SDF expression or Container to F-Rep format.

    When output_file is None, returns the F-Rep text (str for a plain SDF,
    dict {name: frep_string} for a Container).

    When output_file is provided, writes the text to disk and returns the
    path(s).  For a Container, one file is written per set attribute using
    the pattern ``<stem>_<attr><ext>`` (e.g. ``model_shape.frep``).

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       File path, or None to return text. Default: None.

    Returns:
        str:  F-Rep string or written path (plain SDF).
        dict: {name: frep_string} or {name: written_path} (Container).

    Examples:
        # Return string
        frep_str = fp.to_frep(fps.sphere(1.0))

        # Write to file
        fp.to_frep(fps.sphere(1.0), output_file="sphere.frep")

        # Container → dict of strings
        freps = fp.to_frep(fpc)

        # Container → write one file per attribute
        fp.to_frep(fpc, output_file="model.frep")
        # → model_shape.frep, model_r.frep, model_g.frep, model_b.frep
    """
    if isinstance(expr_or_container, _Container):
        frep_dict = expr_or_container.to_frep()
        if output_file is None:
            return frep_dict
        from pathlib import Path
        p = Path(output_file)
        result = {}
        for name, frep_str in frep_dict.items():
            attr_path = p.parent / f"{p.stem}_{name}{p.suffix}"
            attr_path.write_text(frep_str)
            result[name] = str(attr_path)
        return result
    frep_str = _to_frep_rust(expr_or_container)
    if output_file is None:
        return frep_str
    from pathlib import Path
    Path(output_file).write_text(frep_str)
    return output_file


def eval(expr, points, variables=None, **kwargs):
    """
    Evaluate a fidgetpy SDF expression or Container at a set of points.

    For a plain SDF expression, returns an (N,) array of SDF values.

    For a Container, evaluates every set attribute and returns a dict
    ``{name: array}``.  Float-valued attributes produce a constant array.
    Expression attributes are evaluated using the x/y/z variable mapping
    (or the explicit ``variables`` list if provided).

    Args:
        expr:      A fidgetpy SDF expression or Container.
        points:    (N, 3) array of xyz coordinates.
        variables: List of variable expressions mapping columns → variables.
                   When None (default), x/y/z are used automatically.
        **kwargs:  Additional keyword arguments passed to the Rust eval function.

    Returns:
        ndarray:          (N,) array of SDF values (plain SDF).
        dict[str, ndarray]: {name: (N,) array} (Container).

    Examples:
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

        # Plain SDF
        vals = fp.eval(fps.sphere(1.0), pts)

        # Container — all attributes at once
        fpc = fp.container()
        fpc.shape = fps.sphere(1.0)
        fpc.r = fp.x() * 0.5 + 0.5
        results = fp.eval(fpc, pts)
        # {'shape': array([...]), 'r': array([...])}
    """
    if isinstance(expr, _Container):
        import numpy as np
        from fidgetpy.splat import _eval_sdf_at
        pts = np.asarray(points, dtype=np.float32)
        result = {}
        for name, val in expr._attrs.items():
            if val is None:
                continue
            if isinstance(val, (int, float)):
                result[name] = np.full(len(pts), float(val))
            elif variables is not None:
                result[name] = np.asarray(
                    _eval_rust(val, points, variables=variables, **kwargs),
                    dtype=np.float64,
                )
            else:
                result[name] = _eval_sdf_at(val, pts)
        return result
    if variables is not None:
        return _eval_rust(expr, points, variables=variables, **kwargs)
    return _eval_rust(expr, points, **kwargs)


def mesh(expr_or_container, output_file=None, verbose=True, **kwargs):
    """
    Generate a mesh from a fidgetpy SDF expression or Container.

    When output_file is None, returns a Mesh object (vertices + triangles).
    When output_file is provided, writes a colored PLY file and returns the path.

    For a Container, the 'shape' attribute provides the geometry and 'r', 'g', 'b'
    attributes (floats or fidgetpy expressions) are evaluated at every mesh vertex
    to produce per-vertex colors (only when writing a PLY file).

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       Path to the output .ply file, or None to return PyMesh.
        verbose:           Print progress when writing a file. Default True.
        **kwargs:          Passed to the Rust mesh function (e.g. depth=6, numpy=True).

    Returns:
        PyMesh: when output_file is None — a Mesh object with .vertices and .triangles.
        str:    when output_file is set — path to the written .ply file.

    Examples:
        # Return mesh object
        m = fp.mesh(fps.sphere(1.0), depth=5)
        print(m.vertices, m.triangles)

        # Write plain PLY (no vertex colors)
        fp.mesh(fps.sphere(1.0), output_file="sphere.ply", depth=5)

        # Write colored PLY from Container
        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1
        fp.mesh(fpc, output_file="red_sphere.ply", depth=6)
    """
    if output_file is None:
        # Return PyMesh (no color evaluation needed)
        if isinstance(expr_or_container, _Container):
            shape = expr_or_container._attrs.get('shape')
            if shape is None:
                raise ValueError(
                    "Container has no 'shape' attribute. "
                    "Set it with: fpc.shape = some_sdf"
                )
            expr_or_container = shape
        return _mesh_rust(expr_or_container, **kwargs)

    # Write colored PLY file
    import numpy as np
    from fidgetpy.splat import _eval_sdf_at

    color_channels = None

    if isinstance(expr_or_container, _Container):
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

    m = _mesh_rust(expr_or_container, numpy=True, **kwargs)
    verts = np.asarray(m.vertices, dtype=np.float32)
    tris  = np.asarray(m.triangles, dtype=np.int32)
    N = len(verts)

    colors_u8 = None
    if color_channels is not None:
        colors_f = np.ones((N, 3), dtype=np.float64)
        for i, ch in enumerate(color_channels):
            if isinstance(ch, (int, float)):
                colors_f[:, i] = np.clip(float(ch), 0.0, 1.0)
            else:
                colors_f[:, i] = np.clip(_eval_sdf_at(ch, verts), 0.0, 1.0)
        colors_u8 = (colors_f * 255.0).astype(np.uint8)

    _write_mesh_ply(output_file, verts, tris, colors_u8, verbose=verbose)
    return output_file


def splat(expr_or_container, output_file=None, color=None, **kwargs):
    """
    Convert a fidgetpy SDF expression or Container to a Gaussian Splatting representation.

    When a Container is passed, the 'shape' attribute is used for geometry and
    'r', 'g', 'b' attributes (if set) are used for color.  An explicit 'color'
    argument overrides any color stored in the Container.

    When output_file is None, returns a Gaussians object you can inspect
    (.positions, .colors, .normals, .scales, .count) and save later.
    When output_file is provided, writes the .ply and returns the path.

    Args:
        expr_or_container: A fidgetpy SDF expression or a Container.
        output_file:       Path to the output .ply file, or None to return a
                           Gaussians object. Default: None.
        color:             Color override. See fp.splat module for accepted forms.
        **kwargs:          Passed to the underlying splat function (size, domain, …).

    Returns:
        Gaussians: when output_file is None.
        str:       path to the written .ply file when output_file is set.

    Examples:
        import fidgetpy as fp
        import fidgetpy.shape as fps

        # Return Gaussians for inspection
        g = fp.splat(fps.sphere(1.0))
        print(g.count, g.positions.shape)
        g.save("sphere.ply")

        # Write directly
        fp.splat(fps.sphere(1.0), output_file="sphere.ply")

        # Container with color attributes
        fpc = fp.container()
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.2; fpc.g = 0.6; fpc.b = 1.0
        fp.splat(fpc, output_file="blue_sphere.ply")
    """
    if isinstance(expr_or_container, _Container):
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



def _write_mesh_ply(path, vertices, triangles, colors_u8=None, verbose=True):
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

    if verbose:
        color_note = " with vertex colors" if colors_u8 is not None else ""
        print(f"  Wrote {N:,} vertices, {M:,} triangles{color_note} → {path}")


def container(*names):
    """
    Create a Container with declared (empty) attribute slots.

    When called with no arguments, the standard slots ``shape``, ``r``, ``g``,
    ``b`` are created automatically.

    Args:
        *names: String names for the attribute slots.  All values start as None;
                assign them with fpc.name = value.  If omitted, defaults to
                ``("shape", "r", "g", "b")``.

    Returns:
        A Container instance.

    Examples:
        # Default: shape + r/g/b slots
        fpc = fp.container()
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

        # Custom slots
        fpc = fp.container("shape", "roughness", "metallic")
    """
    if not names:
        names = ("shape", "r", "g", "b")
    return _Container(*names)


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

    # Add .to_vm() method — export this single expression as a VM string
    def _sdf_to_vm(self):
        """Return the VM string for this SDF expression."""
        return _to_vm_rust(self)

    setattr(expr_class, 'to_vm', _sdf_to_vm)
    # Keep .vm() as a convenience alias on the expression object
    setattr(expr_class, 'vm', _sdf_to_vm)


# Initialize expression extensions when the module is imported
_extend_expressions()
