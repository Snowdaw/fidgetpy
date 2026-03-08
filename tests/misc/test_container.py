"""
Tests for the Container API (fp.container, fp.vm, fp.eval, fp.mesh, fp.splat).

These tests also serve as usage examples, showing how to:
- Declare and fill a Container with named SDF attributes
- Paint proximity-based color onto a Container
- Export individual attributes or the whole Container to VM strings
- Evaluate a color expression at specific points
- Mesh and splat geometry from a Container
- Use HSL colors (floats and per-point fidgetpy expressions)
"""

import math
import pytest
import numpy as np
import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_colored_container():
    """A sphere painted solid red."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9
    fpc.g = 0.1
    fpc.b = 0.1
    return fpc


# ── Construction ──────────────────────────────────────────────────────────────

def test_container_creation_with_names():
    """Declared attribute slots start as None."""
    fpc = fp.container("shape", "r", "g", "b")
    assert "shape" in fpc
    assert "r" in fpc
    assert "g" in fpc
    assert "b" in fpc
    assert fpc.shape is None
    assert fpc.r is None


def test_container_creation_default():
    """fp.container() with no args creates shape/r/g/b slots; assignment works."""
    fpc = fp.container()
    assert len(fpc) == 4
    assert "shape" in fpc and "r" in fpc and "g" in fpc and "b" in fpc
    fpc.shape = fps.sphere(1.0)
    assert fpc.shape is not None


def test_container_string_names_only():
    """Non-string names in fp.container() raise TypeError."""
    with pytest.raises(TypeError):
        fp.container("shape", 42)


def test_container_repr_non_empty(simple_colored_container):
    """repr() shows name: value pairs."""
    r = repr(simple_colored_container)
    assert "shape" in r
    assert "r" in r


def test_container_repr_empty():
    """repr() of a container with no attrs says so."""
    # Use explicit empty name list via Container directly
    from fidgetpy._expr import Container as _C
    assert "empty" in repr(_C()).lower()


# ── Attribute access ──────────────────────────────────────────────────────────

def test_attribute_assignment_float(simple_colored_container):
    """Float values are stored directly."""
    assert simple_colored_container.r == pytest.approx(0.9)
    assert simple_colored_container.g == pytest.approx(0.1)


def test_attribute_assignment_sdf(simple_colored_container):
    """SDF expressions are stored as-is."""
    shape = simple_colored_container.shape
    assert shape is not None
    assert hasattr(shape, '_is_sdf_expr')


def test_attribute_missing_raises(simple_colored_container):
    """Accessing an undeclared attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        _ = simple_colored_container.roughness


def test_attribute_contains(simple_colored_container):
    """'in' operator checks declared attributes."""
    assert "shape" in simple_colored_container
    assert "roughness" not in simple_colored_container


# ── add / update / remove ─────────────────────────────────────────────────────

def test_add_new_attribute():
    """add() declares a new slot with None value."""
    fpc = fp.container("shape")
    fpc.add("roughness")
    assert "roughness" in fpc
    assert fpc.roughness is None


def test_add_existing_is_noop():
    """add() on an existing attribute leaves the value unchanged."""
    fpc = fp.container("shape")
    fpc.shape = fps.sphere(1.0)
    fpc.add("shape")
    assert fpc.shape is not None  # unchanged


def test_add_non_string_raises():
    """add() with a non-string name raises TypeError."""
    fpc = fp.container()
    with pytest.raises(TypeError):
        fpc.add(123)


def test_update_existing():
    """update() modifies an existing attribute value."""
    fpc = fp.container("r")
    fpc.r = 0.5
    fpc.update("r", 0.8)
    assert fpc.r == pytest.approx(0.8)


def test_update_missing_raises():
    """update() on an undeclared attribute raises KeyError."""
    fpc = fp.container()
    with pytest.raises(KeyError):
        fpc.update("opacity", 0.5)  # 'opacity' is not a default slot


def test_remove_attribute():
    """remove() deletes a declared attribute."""
    fpc = fp.container("shape", "r")
    fpc.remove("r")
    assert "r" not in fpc
    assert "shape" in fpc


def test_remove_missing_raises():
    """remove() on an undeclared attribute raises KeyError."""
    fpc = fp.container()
    with pytest.raises(KeyError):
        fpc.remove("roughness")


def test_add_update_remove_sequence():
    """add(), update(), remove() work correctly in sequence."""
    fpc = fp.container()
    fpc.add("r")
    fpc.add("g")
    fpc.update("r", 0.5)
    fpc.remove("g")
    assert "r" in fpc
    assert "g" not in fpc
    assert fpc.r == 0.5


# ── to_vm() export ────────────────────────────────────────────────────────────

def test_vm_plain_sdf_returns_string():
    """fp.to_vm() on a plain SDF expression returns a VM string."""
    sdf = fps.sphere(1.0)
    result = fp.to_vm(sdf)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "var-x" in result


def test_vm_container_returns_dict(simple_colored_container):
    """fp.to_vm() on a Container returns a dict of name → VM string."""
    vms = fp.to_vm(simple_colored_container)
    assert isinstance(vms, dict)
    assert set(vms.keys()) == {"shape", "r", "g", "b"}
    for name, vm_str in vms.items():
        assert isinstance(vm_str, str), f"{name} is not a string"
        assert len(vm_str) > 0


def test_vm_container_float_attribute():
    """Float attributes are serialised as a constant VM expression."""
    fpc = fp.container("r")
    fpc.r = 0.75
    vms = fp.to_vm(fpc)
    # The VM string for a constant must exist and be parseable
    assert "r" in vms
    reimported = fp.from_vm(vms["r"])
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    val = fp.eval(reimported, pts, variables=[fp.x(), fp.y(), fp.z()])
    assert float(val[0]) == pytest.approx(0.75, abs=1e-5)


def test_vm_container_skips_unset_attributes():
    """fp.to_vm() omits attributes that are still None."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    # r, g, b remain None
    vms = fp.to_vm(fpc)
    assert "shape" in vms
    assert "r" not in vms


def test_vm_unset_attribute_raises(simple_colored_container):
    """ChannelEntry.to_vm() raises ValueError for an unset attribute."""
    fpc = fp.container("shape", "opacity")
    fpc.shape = fps.sphere(1.0)
    # opacity is unset
    for ch in fpc:
        if ch.name == "opacity":
            with pytest.raises(ValueError):
                ch.to_vm()


def test_vm_method_on_expression():
    """sdf.to_vm() returns the same string as fp.to_vm(sdf)."""
    sdf = fps.sphere(1.0)
    assert sdf.to_vm() == fp.to_vm(sdf)


def test_to_vm_plain_sdf_writes_file(tmp_path):
    """fp.to_vm(sdf, output_file=...) writes a VM file and returns the path."""
    out = tmp_path / "sphere.vm"
    result = fp.to_vm(fps.sphere(1.0), output_file=str(out))
    assert out.exists()
    assert result == str(out)
    assert "var-x" in out.read_text()


def test_to_vm_container_writes_per_attribute_files(tmp_path):
    """fp.to_vm(container, output_file=...) writes one file per attribute."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

    result = fp.to_vm(fpc, output_file=str(tmp_path / "model.vm"))
    assert isinstance(result, dict)
    assert set(result.keys()) == {"shape", "r", "g", "b"}
    for name, path in result.items():
        assert (tmp_path / f"model_{name}.vm").exists()
        assert path == str(tmp_path / f"model_{name}.vm")


# ── to_frep() export ──────────────────────────────────────────────────────────

def test_to_frep_plain_sdf_returns_string():
    """fp.to_frep() on a plain SDF returns an F-Rep string."""
    sdf = fps.sphere(1.0)
    result = fp.to_frep(sdf)
    assert isinstance(result, str)
    assert "sqrt" in result.lower()


def test_to_frep_container_returns_dict(simple_colored_container):
    """fp.to_frep() on a Container returns a dict of name → F-Rep string."""
    freps = fp.to_frep(simple_colored_container)
    assert isinstance(freps, dict)
    assert set(freps.keys()) == {"shape", "r", "g", "b"}
    for name, frep_str in freps.items():
        assert isinstance(frep_str, str)
        assert len(frep_str) > 0


def test_to_frep_container_float_attribute():
    """Float attributes in a Container produce a valid F-Rep string."""
    fpc = fp.container("r")
    fpc.r = 0.5
    freps = fp.to_frep(fpc)
    assert "r" in freps
    assert isinstance(freps["r"], str)


def test_to_frep_plain_sdf_writes_file(tmp_path):
    """fp.to_frep(sdf, output_file=...) writes a file and returns the path."""
    out = tmp_path / "sphere.frep"
    result = fp.to_frep(fps.sphere(1.0), output_file=str(out))
    assert out.exists()
    assert result == str(out)
    assert len(out.read_text()) > 0


def test_to_frep_container_writes_per_attribute_files(tmp_path):
    """fp.to_frep(container, output_file=...) writes one file per attribute."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

    result = fp.to_frep(fpc, output_file=str(tmp_path / "model.frep"))
    assert isinstance(result, dict)
    assert set(result.keys()) == {"shape", "r", "g", "b"}
    for name, path in result.items():
        assert (tmp_path / f"model_{name}.frep").exists()


def test_channel_entry_to_frep(simple_colored_container):
    """ChannelEntry.to_frep() returns a valid F-Rep string per attribute."""
    for ch in simple_colored_container:
        frep_str = ch.to_frep()
        assert isinstance(frep_str, str)
        assert len(frep_str) > 0


# ── Iteration ─────────────────────────────────────────────────────────────────

def test_iteration_yields_channel_entries(simple_colored_container):
    """Iterating a Container yields ChannelEntry objects with .name and .value."""
    names = []
    for ch in simple_colored_container:
        assert hasattr(ch, 'name')
        assert hasattr(ch, 'value')
        names.append(ch.name)
    assert names == ["shape", "r", "g", "b"]


def test_iteration_to_vm_per_entry(simple_colored_container):
    """Each ChannelEntry.to_vm() returns a valid VM string."""
    for ch in simple_colored_container:
        vm_str = ch.to_vm()
        assert isinstance(vm_str, str)
        assert len(vm_str) > 0


def test_len(simple_colored_container):
    """len() counts declared attributes."""
    assert len(simple_colored_container) == 4


# ── eval() ────────────────────────────────────────────────────────────────────

def test_eval_sdf_at_surface():
    """An SDF expression evaluates near zero at the surface."""
    sdf = fps.sphere(1.0)
    # A point on the surface of a unit sphere
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    vals = fp.eval(sdf, pts, variables=[fp.x(), fp.y(), fp.z()])
    assert float(vals[0]) == pytest.approx(0.0, abs=1e-5)


def test_eval_container_all_attributes():
    """fp.eval() on a Container evaluates every set attribute and returns a dict."""
    fpc = fp.container("shape", "r")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9   # constant float

    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    results = fp.eval(fpc, pts)
    assert isinstance(results, dict)
    assert set(results.keys()) == {"shape", "r"}
    assert float(results["shape"][0]) == pytest.approx(0.0, abs=1e-5)
    assert float(results["r"][0]) == pytest.approx(0.9, abs=1e-6)


def test_eval_container_expression_attribute():
    """fp.eval() evaluates expression attributes per-point."""
    fpc = fp.container("shape", "r")
    fpc.shape = fps.sphere(1.0)
    fpc.r = fp.x() * 0.5 + 0.5   # 0.0 at x=-1, 1.0 at x=1

    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    results = fp.eval(fpc, pts)
    assert float(results["r"][0]) == pytest.approx(0.0, abs=1e-5)
    assert float(results["r"][1]) == pytest.approx(1.0, abs=1e-5)


def test_eval_container_unset_skipped():
    """fp.eval() skips attributes that are None (unset)."""
    fpc = fp.container("shape", "r", "g")
    fpc.shape = fps.sphere(1.0)
    # g is unset
    fpc.r = 0.5

    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    results = fp.eval(fpc, pts)
    assert "shape" in results
    assert "r" in results
    assert "g" not in results


def test_eval_color_expression():
    """A per-point color expression evaluates to expected values.

    Note: fp.eval() is strict — every variable in the mapping must be used by
    the expression, and every column of the points array must be mapped.
    Pass only the variables your expression actually uses.
    """
    # r = x * 0.5 + 0.5  →  at x=1: 1.0, at x=-1: 0.0
    # This expression only uses x, so we pass a 1-column array and [fp.x()].
    r_expr = fp.x() * 0.5 + 0.5
    pts_x = np.array([[1.0], [-1.0]], dtype=np.float32)
    vals = fp.eval(r_expr, pts_x, variables=[fp.x()])
    assert float(vals[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(vals[1]) == pytest.approx(0.0, abs=1e-5)


def test_eval_hsl_color_at_known_point():
    """HSL at h=0 (red hue) returns r≈1, g≈0, b≈0 (before chroma adjustment)."""
    rgb = fpm.hsl(0.0, 1.0, 0.5)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    vars_ = [fp.x(), fp.y(), fp.z()]

    r_val = float(fp.eval(rgb['r'], pts, variables=vars_)[0])
    g_val = float(fp.eval(rgb['g'], pts, variables=vars_)[0])
    b_val = float(fp.eval(rgb['b'], pts, variables=vars_)[0])

    assert r_val == pytest.approx(1.0, abs=1e-5)
    assert g_val == pytest.approx(0.0, abs=1e-5)
    assert b_val == pytest.approx(0.0, abs=1e-5)


def test_eval_hsl_green_hue():
    """HSL at h=1/3 (green hue) with pure saturation: r≈0, g≈1, b≈0."""
    rgb = fpm.hsl(1.0 / 3.0, 1.0, 0.5)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    vars_ = [fp.x(), fp.y(), fp.z()]

    r_val = float(fp.eval(rgb['r'], pts, variables=vars_)[0])
    g_val = float(fp.eval(rgb['g'], pts, variables=vars_)[0])
    b_val = float(fp.eval(rgb['b'], pts, variables=vars_)[0])

    assert r_val == pytest.approx(0.0, abs=1e-5)
    assert g_val == pytest.approx(1.0, abs=1e-5)
    assert b_val == pytest.approx(0.0, abs=1e-5)


# ── paint() ───────────────────────────────────────────────────────────────────

def test_paint_modifies_attribute():
    """paint() blends attribute values near a region SDF."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9
    fpc.g = 0.1
    fpc.b = 0.1

    dot = fps.sphere(0.15).translate(0.5, 0.5, 0.0)
    fpc.paint(dot, r=0.1, g=0.9, b=0.1)

    # After paint, r/g/b should be expressions (not plain floats anymore)
    assert hasattr(fpc.r, '_is_sdf_expr')
    assert hasattr(fpc.g, '_is_sdf_expr')


def test_paint_evaluates_correctly():
    """Inside the painted region the new color dominates; far away the old one does."""
    fpc = fp.container("shape", "r")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.0  # start black

    # Paint a small sphere at the origin white (r=1)
    dot = fps.sphere(0.1)
    fpc.paint(dot, r=1.0, width=0.05)

    vars_ = [fp.x(), fp.y(), fp.z()]

    # Inside dot (origin): r should be close to 1
    inside = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    r_inside = float(fp.eval(fpc.r, inside, variables=vars_)[0])
    assert r_inside == pytest.approx(1.0, abs=0.05)

    # Far away (0.5, 0, 0): r should still be close to 0
    outside = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    r_outside = float(fp.eval(fpc.r, outside, variables=vars_)[0])
    assert r_outside == pytest.approx(0.0, abs=0.05)


def test_paint_returns_self():
    """paint() returns the Container for chaining."""
    fpc = fp.container("shape", "r")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.5
    result = fpc.paint(fps.sphere(0.1), r=1.0)
    assert result is fpc


# ── mesh() ────────────────────────────────────────────────────────────────────

def test_mesh_plain_sdf():
    """fp.mesh() on a plain SDF returns a mesh with vertices and triangles."""
    m = fp.mesh(fps.sphere(1.0), depth=3, numpy=True)
    assert hasattr(m, 'vertices')
    assert hasattr(m, 'triangles')
    assert len(m.vertices) > 0
    assert len(m.triangles) > 0


def test_mesh_container(simple_colored_container):
    """fp.mesh() on a Container meshes the 'shape' attribute."""
    m = fp.mesh(simple_colored_container, depth=3, numpy=True)
    assert len(m.vertices) > 0
    assert len(m.triangles) > 0


def test_mesh_method_on_container(simple_colored_container):
    """Container.mesh() works the same as fp.mesh(container)."""
    m1 = simple_colored_container.mesh(depth=3, numpy=True)
    m2 = fp.mesh(simple_colored_container, depth=3, numpy=True)
    assert m1.vertices.shape == m2.vertices.shape


def test_mesh_container_missing_shape_raises():
    """fp.mesh() raises ValueError when the Container has no 'shape'."""
    fpc = fp.container("r")
    fpc.r = 0.5
    with pytest.raises(ValueError, match="shape"):
        fp.mesh(fpc, depth=3)


# ── splat() ───────────────────────────────────────────────────────────────────

def test_splat_returns_gaussians_object():
    """fp.splat() with no output_file returns a Gaussians object."""
    from fidgetpy.splat import Gaussians
    g = fp.splat(fps.sphere(0.5), size=24, domain=(-0.8, 0.8), verbose=False)
    assert isinstance(g, Gaussians)
    assert g.count > 0
    assert g.positions.shape == (g.count, 3)
    assert g.colors.shape == (g.count, 3)
    assert g.normals.shape == (g.count, 3)
    assert g.scales.shape == (g.count, 3)
    assert g.quaternions.shape == (g.count, 4)
    # Colors should be in [0, 1]
    assert g.colors.min() >= 0.0 and g.colors.max() <= 1.0


def test_splat_gaussians_save(tmp_path):
    """Gaussians.save() writes a valid .ply file."""
    g = fp.splat(fps.sphere(0.5), size=24, domain=(-0.8, 0.8), verbose=False)
    out = tmp_path / "sphere.ply"
    result = g.save(str(out), verbose=False)
    assert out.exists()
    assert out.stat().st_size > 0
    assert result == str(out)


def test_splat_plain_sdf(tmp_path):
    """fp.splat() with output_file writes a valid .ply file and returns the path."""
    out = tmp_path / "sphere.ply"
    result = fp.splat(
        fps.sphere(0.5),
        output_file=str(out),
        size=24,
        domain=(-0.8, 0.8),
        verbose=False,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert result == str(out)


def test_splat_container_with_solid_color(tmp_path):
    """fp.splat() with a solid-color Container produces a .ply file."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)
    fpc.r = 0.2
    fpc.g = 0.6
    fpc.b = 1.0

    out = tmp_path / "blue_sphere.ply"
    fp.splat(fpc, output_file=str(out), size=24, domain=(-0.8, 0.8), verbose=False)
    assert out.exists()
    assert out.stat().st_size > 0


def test_splat_container_with_expression_color(tmp_path):
    """fp.splat() uses per-point color expressions from the Container."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)
    # X-axis gradient: red on +X, blue on -X
    fpc.r = fp.x() * 0.5 + 0.5
    fpc.g = 0.1
    fpc.b = 0.5 - fp.x() * 0.5

    out = tmp_path / "gradient_sphere.ply"
    fp.splat(fpc, output_file=str(out), size=24, domain=(-0.8, 0.8), verbose=False)
    assert out.exists()
    assert out.stat().st_size > 0


def test_splat_container_with_hsl_color(tmp_path):
    """fp.splat() works with an HSL-derived per-point color in a Container."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)

    # Color wheel: hue driven by angle around the Z axis
    angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
    rgb = fpm.hsl(angle, 1.0, 0.5)
    fpc.r = rgb['r']
    fpc.g = rgb['g']
    fpc.b = rgb['b']

    out = tmp_path / "hsl_sphere.ply"
    fp.splat(fpc, output_file=str(out), size=24, domain=(-0.8, 0.8), verbose=False)
    assert out.exists()
    assert out.stat().st_size > 0


def test_splat_zero_color_channels_not_replaced_by_white(tmp_path):
    """Regression: g=0 and b=0 must not be treated as 'unset' and replaced with 1.0.

    A red sphere (r=1, g=0, b=0) was previously rendered white because Python's
    falsy evaluation turned `0 or 1.0` into `1.0` for the g and b channels.
    """
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)
    fpc.r = 1
    fpc.g = 0   # zero — must stay zero
    fpc.b = 0   # zero — must stay zero

    out = tmp_path / "red_sphere.ply"
    fp.splat(fpc, output_file=str(out), size=24, domain=(-0.8, 0.8), verbose=False)

    # Read back the PLY and check that the SH DC values encode red, not white.
    # For degree-0 splats: f_dc_0 >> f_dc_1 and f_dc_0 >> f_dc_2 for red.
    import struct
    data = out.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    body = data[header_end:]

    # Parse the dtype from the header to find f_dc offsets
    header = data[:header_end].decode("ascii")
    props = [l.split()[-1] for l in header.splitlines() if l.startswith("property float")]
    dc0_idx = props.index("f_dc_0")
    dc1_idx = props.index("f_dc_1")
    dc2_idx = props.index("f_dc_2")

    # Read the first Gaussian
    dc0 = struct.unpack_from("<f", body, dc0_idx * 4)[0]
    dc1 = struct.unpack_from("<f", body, dc1_idx * 4)[0]
    dc2 = struct.unpack_from("<f", body, dc2_idx * 4)[0]

    assert dc0 > dc1 + 1.0, f"Red channel DC should dominate green: dc0={dc0:.2f}, dc1={dc1:.2f}"
    assert dc0 > dc2 + 1.0, f"Red channel DC should dominate blue: dc0={dc0:.2f}, dc2={dc2:.2f}"


def test_splat_container_missing_shape_raises():
    """fp.splat() raises ValueError when the Container has no 'shape'."""
    fpc = fp.container("r", "g", "b")
    fpc.r = 1.0; fpc.g = 0.0; fpc.b = 0.0
    with pytest.raises(ValueError, match="shape"):
        fp.splat(fpc)


def test_splat_container_method(tmp_path):
    """Container.splat() writes a .ply file using shape + r/g/b."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)
    fpc.r = 0.9
    fpc.g = 0.1
    fpc.b = 0.1

    out = str(tmp_path / "red_sphere.ply")
    fpc.splat(output_file=out, size=24, domain=(-0.8, 0.8), verbose=False)
    assert (tmp_path / "red_sphere.ply").exists()


def test_splat_container_method_no_output():
    """Container.splat() with no output_file returns a Gaussians object."""
    from fidgetpy.splat import Gaussians
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(0.5)
    fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1
    g = fpc.splat(size=24, domain=(-0.8, 0.8), verbose=False)
    assert isinstance(g, Gaussians)
    assert g.count > 0


# ── Paint + Splat integration ─────────────────────────────────────────────────

def test_paint_then_splat(tmp_path):
    """A Container painted with a proximity dot can be splatted successfully."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.rounded_box(0.8, 0.8, 0.8, 0.02)
    fpc.r = 0.1
    fpc.g = 0.1
    fpc.b = 0.9  # blue base

    # Green dot near a corner
    dot = fps.sphere(0.15).translate(0.3, 0.3, 0.0)
    fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.08)

    out = tmp_path / "painted_box.ply"
    fp.splat(fpc, output_file=str(out), size=24, domain=(-0.8, 0.8), verbose=False)
    assert out.exists()


# ── HSL convenience ───────────────────────────────────────────────────────────

def test_hsl_returns_rgb_dict():
    """fpm.hsl() returns a dict with keys 'r', 'g', 'b'."""
    rgb = fpm.hsl(0.5, 1.0, 0.5)
    assert set(rgb.keys()) == {"r", "g", "b"}


def test_hsl_float_args_gives_floats_or_expr():
    """With all-float args, hsl still returns valid r/g/b (float or expression)."""
    rgb = fpm.hsl(0.0, 1.0, 0.5)
    # We can evaluate them — they should be valid as expressions or plain floats
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    vars_ = [fp.x(), fp.y(), fp.z()]
    for ch in ('r', 'g', 'b'):
        v = rgb[ch]
        if isinstance(v, (int, float)):
            assert 0.0 <= v <= 1.0
        else:
            val = float(fp.eval(v, pts, variables=vars_)[0])
            assert 0.0 <= val <= 1.0


def test_hsl_expression_arg():
    """An expression argument to hsl() produces an expression result."""
    # h driven by x position
    h_expr = fp.x() * 0.5 + 0.5
    rgb = fpm.hsl(h_expr, 1.0, 0.5)
    assert hasattr(rgb['r'], '_is_sdf_expr')
    assert hasattr(rgb['g'], '_is_sdf_expr')
    assert hasattr(rgb['b'], '_is_sdf_expr')


# ── mesh() with output_file ───────────────────────────────────────────────────

def test_mesh_plain_sdf_writes_ply(tmp_path):
    """fp.mesh(output_file=...) on a plain SDF writes a PLY without vertex colors."""
    out = tmp_path / "sphere.ply"
    result = fp.mesh(fps.sphere(1.0), output_file=str(out), depth=4)
    assert out.exists()
    assert result == str(out)
    header = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "element vertex" in header
    assert "element face" in header
    assert "property uchar red" not in header   # no color for plain SDF


def test_mesh_solid_color(tmp_path):
    """fp.mesh(output_file=...) with solid color attributes writes per-vertex RGB."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

    out = tmp_path / "red.ply"
    fp.mesh(fpc, output_file=str(out), depth=4)
    header = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" in header
    assert "property uchar green" in header
    assert "property uchar blue" in header


def test_mesh_expression_color(tmp_path):
    """fp.mesh(output_file=...) evaluates expression color channels at vertex positions."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    # X-axis gradient
    fpc.r = fp.x() * 0.5 + 0.5
    fpc.g = 0.2
    fpc.b = 0.5 - fp.x() * 0.5

    out = tmp_path / "gradient.ply"
    fp.mesh(fpc, output_file=str(out), depth=4)
    assert out.exists()
    assert out.stat().st_size > 0
    header = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" in header


def test_mesh_hsl_color(tmp_path):
    """fp.mesh(output_file=...) works with HSL-derived per-vertex color."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
    rgb = fpm.hsl(angle, 1.0, 0.5)
    fpc.r = rgb['r']; fpc.g = rgb['g']; fpc.b = rgb['b']

    out = tmp_path / "hsl.ply"
    fp.mesh(fpc, output_file=str(out), depth=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_mesh_method_with_output_file(tmp_path):
    """Container.mesh(output_file=...) produces the same result as fp.mesh(container, output_file=...)."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9; fpc.g = 0.1; fpc.b = 0.1

    out1 = str(tmp_path / "a.ply")
    out2 = str(tmp_path / "b.ply")
    fp.mesh(fpc, output_file=out1, depth=4)
    fpc.mesh(output_file=out2, depth=4)

    import os
    assert os.path.getsize(out1) == os.path.getsize(out2)


def test_mesh_partial_color(tmp_path):
    """Unset color channels default to 1.0 (white contribution)."""
    fpc = fp.container("shape", "r")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9   # only r is set; g and b will default to 1.0

    out = tmp_path / "partial.ply"
    fp.mesh(fpc, output_file=str(out), depth=4)
    header = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" in header


def test_mesh_vertex_colors_correct(tmp_path):
    """Verify that solid red (r=1, g=0, b=0) encodes correctly as RGB (255,0,0)."""
    import numpy as np

    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 1.0; fpc.g = 0.0; fpc.b = 0.0

    out = tmp_path / "red.ply"
    fp.mesh(fpc, output_file=str(out), depth=4)

    data = out.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    body = data[header_end:]

    n_verts = int([l for l in header.splitlines() if l.startswith("element vertex")][0].split()[-1])
    assert n_verts > 0

    # Each vertex: 3 floats (xyz) + 3 bytes (rgb) = 15 bytes
    r_byte = body[12]   # offset 12: after x(4) y(4) z(4)
    g_byte = body[13]
    b_byte = body[14]
    assert r_byte == 255, f"Expected r=255, got {r_byte}"
    assert g_byte == 0,   f"Expected g=0, got {g_byte}"
    assert b_byte == 0,   f"Expected b=0, got {b_byte}"


def test_mesh_missing_shape_raises():
    """fp.mesh() raises ValueError when the Container has no 'shape'."""
    fpc = fp.container("r", "g", "b")
    fpc.r = 1.0; fpc.g = 0.0; fpc.b = 0.0
    with pytest.raises(ValueError, match="shape"):
        fp.mesh(fpc, output_file="unused.ply")


def test_mesh_paint_then_mesh(tmp_path):
    """Paint proximity blend → mesh with output_file writes correct colored PLY."""
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.1; fpc.g = 0.1; fpc.b = 0.9   # blue base

    dot = fps.sphere(0.3).translate(0.6, 0.0, 0.0)
    fpc.paint(dot, r=1.0, g=0.2, b=0.0)   # paint orange dot

    out = tmp_path / "painted.ply"
    fp.mesh(fpc, output_file=str(out), depth=5)
    assert out.exists()
    header = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" in header


# ── VM round-trip ─────────────────────────────────────────────────────────────

def test_to_vm_roundtrip_shape():
    """Shape exported to VM and re-imported evaluates identically."""
    sdf = fps.sphere(1.0)
    vm_str = fp.to_vm(sdf)
    reimported = fp.from_vm(vm_str)

    pts = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    vars_ = [fp.x(), fp.y(), fp.z()]
    original_vals = fp.eval(sdf, pts, variables=vars_)
    reimported_vals = fp.eval(reimported, pts, variables=vars_)

    np.testing.assert_allclose(original_vals, reimported_vals, atol=1e-5)


def test_to_vm_container_color_round_trip():
    """Color VM strings round-trip: export, re-import, evaluate match original."""
    # Use a 3D expression so fp.eval works with a standard [x, y, z] mapping.
    fpc = fp.container("r")
    fpc.r = fpm.sqrt(fp.x()**2 + fp.y()**2 + fp.z()**2)  # distance from origin

    vms = fp.to_vm(fpc)
    reimported_r = fp.from_vm(vms["r"])

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    vars_ = [fp.x(), fp.y(), fp.z()]
    original_vals   = fp.eval(fpc.r, pts, variables=vars_)
    reimported_vals = fp.eval(reimported_r, pts, variables=vars_)

    np.testing.assert_allclose(original_vals, reimported_vals, atol=1e-5)
