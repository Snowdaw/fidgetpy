"""
Symbolic surface shading — SDF operations for lighting.

All functions take an SDF expression and return SDF expressions (or tuples
of them).  They compose freely with other expressions and are evaluated
lazily inside Fidget's JIT at mesh/splat/eval time.

These are safe for any SDF complexity: they linearise the expression tree
(6 remap_xyz calls for the gradient) but never nest the SDF inside itself.
"""

import math as _py_math
import fidgetpy as fp
from .basic_math import sqrt
from .vector_math import normalize
from .transformations import remap_xyz
from .basic_math import clamp


def gradient(sdf, eps=1e-4):
    """
    Finite-difference gradient of an SDF. Returns (gx, gy, gz) as SDF expressions.

    Uses central differences: ``gx = (sdf(x+e, y, z) − sdf(x−e, y, z)) / 2e``.
    The gradient magnitude is approximately 1 for a well-formed SDF.

    Args:
        sdf: A fidgetpy SDF expression.
        eps: Finite-difference step size. Default 1e-4.

    Returns:
        tuple: (gx, gy, gz) — three SDF expressions.

    Example::

        gx, gy, gz = fpm.gradient(fps.sphere(1.0))
        magnitude = (gx**2 + gy**2 + gz**2).sqrt()
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    inv2e = 1.0 / (2.0 * eps)
    gx = (remap_xyz(sdf, x + eps, y, z) - remap_xyz(sdf, x - eps, y, z)) * inv2e
    gy = (remap_xyz(sdf, x, y + eps, z) - remap_xyz(sdf, x, y - eps, z)) * inv2e
    gz = (remap_xyz(sdf, x, y, z + eps) - remap_xyz(sdf, x, y, z - eps)) * inv2e
    return (gx, gy, gz)


def normal(sdf, eps=1e-4):
    """
    Surface normal as (nx, ny, nz) SDF expressions (unit-length gradient).

    Args:
        sdf: A fidgetpy SDF expression.
        eps: Finite-difference step size. Default 1e-4.

    Returns:
        tuple: (nx, ny, nz) — approximately unit-length SDF expressions.

    Example::

        nx, ny, nz = fpm.normal(fps.sphere(1.0))
        fpc.r = nx * 0.5 + 0.5   # normal-map visualisation
        fpc.g = ny * 0.5 + 0.5
        fpc.b = nz * 0.5 + 0.5
    """
    gx, gy, gz = gradient(sdf, eps)
    return normalize(gx, gy, gz)


def diffuse(sdf, light_dir=(1.0, 1.0, 1.0), eps=1e-4):
    """
    Lambertian (diffuse) lighting as an SDF expression.

    Computes ``clamp(dot(normal, light_dir), 0, 1)`` where the normal is
    derived from the SDF gradient.  Returns a symbolic expression in [0, 1]
    that is 1 where the surface faces the light directly.

    Args:
        sdf:       A fidgetpy SDF expression.
        light_dir: Light direction (need not be unit length). Default (1, 1, 1).
        eps:       Finite-difference step for gradient. Default 1e-4.

    Returns:
        SDF expression in [0, 1].

    Example::

        light = fpm.diffuse(fps.sphere(1.0), light_dir=(1, 2, 3))
        fpc.r = 0.8 * light
        fpc.g = 0.4 * light
        fpc.b = 0.2 * light
    """
    lx, ly, lz = light_dir
    ll = _py_math.sqrt(lx * lx + ly * ly + lz * lz)
    if ll < 1e-12:
        raise ValueError("light_dir must be a non-zero vector")
    lx, ly, lz = lx / ll, ly / ll, lz / ll

    nx, ny, nz = normal(sdf, eps)
    dot = nx * lx + ny * ly + nz * lz
    return clamp(dot, 0.0, 1.0)
