"""
Tests for symbolic shading functions in fidgetpy.math:
gradient, normal, diffuse.
"""
import numpy as np
import pytest

import fidgetpy as fp
import fidgetpy.shape as fps
import fidgetpy.math as fpm
from fidgetpy.splat import _eval_sdf_at


def _eval(expr, pts):
    return _eval_sdf_at(expr, np.asarray(pts, dtype=np.float32))


# ── gradient ──────────────────────────────────────────────────────────────────

class TestGradient:
    def test_returns_three_expressions(self):
        gx, gy, gz = fpm.gradient(fps.sphere(1.0))
        assert hasattr(gx, '_is_sdf_expr')
        assert hasattr(gy, '_is_sdf_expr')
        assert hasattr(gz, '_is_sdf_expr')

    def test_sphere_gradient_at_surface_x(self):
        gx, gy, gz = fpm.gradient(fps.sphere(1.0))
        pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _eval(gx, pt)[0] == pytest.approx(1.0, abs=0.01)
        assert _eval(gy, pt)[0] == pytest.approx(0.0, abs=0.01)
        assert _eval(gz, pt)[0] == pytest.approx(0.0, abs=0.01)

    def test_sphere_gradient_at_surface_y(self):
        gx, gy, gz = fpm.gradient(fps.sphere(1.0))
        pt = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        assert _eval(gx, pt)[0] == pytest.approx(0.0, abs=0.01)
        assert _eval(gy, pt)[0] == pytest.approx(1.0, abs=0.01)
        assert _eval(gz, pt)[0] == pytest.approx(0.0, abs=0.01)

    def test_gradient_magnitude_near_one(self):
        gx, gy, gz = fpm.gradient(fps.sphere(1.0))
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
                       dtype=np.float32)
        mag = np.sqrt(_eval(gx, pts)**2 + _eval(gy, pts)**2 + _eval(gz, pts)**2)
        np.testing.assert_allclose(mag, 1.0, atol=0.02)


# ── normal ────────────────────────────────────────────────────────────────────

class TestNormal:
    def test_returns_three_expressions(self):
        nx, ny, nz = fpm.normal(fps.sphere(1.0))
        assert hasattr(nx, '_is_sdf_expr')

    def test_unit_length(self):
        nx, ny, nz = fpm.normal(fps.sphere(1.0))
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        mag = np.sqrt(_eval(nx, pts)**2 + _eval(ny, pts)**2 + _eval(nz, pts)**2)
        np.testing.assert_allclose(mag, 1.0, atol=0.01)

    def test_sphere_normal_direction(self):
        nx, ny, nz = fpm.normal(fps.sphere(1.0))
        pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _eval(nx, pt)[0] == pytest.approx(1.0, abs=0.01)
        assert _eval(ny, pt)[0] == pytest.approx(0.0, abs=0.01)
        assert _eval(nz, pt)[0] == pytest.approx(0.0, abs=0.01)

    def test_custom_eps(self):
        nx, ny, nz = fpm.normal(fps.sphere(1.0), eps=1e-3)
        pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _eval(nx, pt)[0] == pytest.approx(1.0, abs=0.02)


# ── diffuse ───────────────────────────────────────────────────────────────────

class TestDiffuse:
    def test_returns_expression(self):
        d = fpm.diffuse(fps.sphere(1.0))
        assert hasattr(d, '_is_sdf_expr')

    def test_range_zero_to_one(self):
        d = fpm.diffuse(fps.sphere(1.0), light_dir=(1, 1, 1))
        pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       dtype=np.float32)
        vals = _eval(d, pts)
        assert np.all(vals >= -0.001)
        assert np.all(vals <= 1.001)

    def test_light_from_x(self):
        d = fpm.diffuse(fps.sphere(1.0), light_dir=(1, 0, 0))
        lit  = _eval(d, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        dark = _eval(d, np.array([[-1.0, 0.0, 0.0]], dtype=np.float32))
        assert lit[0]  == pytest.approx(1.0, abs=0.02)
        assert dark[0] == pytest.approx(0.0, abs=0.02)

    def test_unnormalised_light_dir_same_result(self):
        d1 = fpm.diffuse(fps.sphere(1.0), light_dir=(1, 0, 0))
        d2 = fpm.diffuse(fps.sphere(1.0), light_dir=(5, 0, 0))
        pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _eval(d1, pt)[0] == pytest.approx(_eval(d2, pt)[0], abs=0.01)

    def test_zero_light_dir_raises(self):
        with pytest.raises(ValueError):
            fpm.diffuse(fps.sphere(1.0), light_dir=(0, 0, 0))

    def test_container_integration(self):
        shape = fps.sphere(1.0)
        fpc = fp.container()
        fpc.shape = shape
        fpc.r = fpm.diffuse(shape, light_dir=(1, 0, 0))
        fpc.g = 0.3
        fpc.b = 0.1
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        result = fp.eval(fpc, pts)
        assert result['r'][0] == pytest.approx(1.0, abs=0.05)
        assert result['r'][1] == pytest.approx(0.0, abs=0.05)
