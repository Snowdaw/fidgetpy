"""
Domain manipulation functions for Fidget.

This module provides domain manipulation operations for SDF expressions, including:
- Repetition (repeat)
- Mirroring (mirror)
- Symmetry (symmetry)
"""

import fidgetpy as fp
from .basic_math import floor, abs
from .transformations import remap_xyz

def repeat(expr, px, py, pz):
    """
    Repeats space along each axis with different periods.

    Args:
        expr: The SDF expression to repeat
        px: The repetition period along the x-axis (0 for no repetition)
        py: The repetition period along the y-axis (0 for no repetition)
        pz: The repetition period along the z-axis (0 for no repetition)

    Returns:
        A new SDF expression with repeated space

    Raises:
        TypeError: If expr is not an SDF expression or periods are not numbers
        ValueError: If any period is negative

    Examples:
        # Repeat a sphere every 3 units in x and y, but not in z
        sphere = fp.shape.sphere(1.0)
        repeated_sphere = fpm.repeat(sphere, 3.0, 3.0, 0.0)

    Can be used as either:
    - fpm.repeat(expr, px, py, pz)
    - expr.repeat(px, py, pz) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("First argument must be an SDF expression")

    for period, axis in zip([px, py, pz], ['x', 'y', 'z']):
        if not isinstance(period, (int, float)) and not hasattr(period, '_is_sdf_expr'):
            raise TypeError(f"Period for {axis}-axis must be a number or SDF expression")

        if isinstance(period, (int, float)) and period < 0:
            raise ValueError(f"Period for {axis}-axis must be non-negative")

    x, y, z = fp.x(), fp.y(), fp.z()

    # Handle each axis - if period is 0, use the original coordinate
    mx = x if px == 0 else (x - px * floor(x / px + 0.5))
    my = y if py == 0 else (y - py * floor(y / py + 0.5))
    mz = z if pz == 0 else (z - pz * floor(z / pz + 0.5))

    return remap_xyz(expr, mx, my, mz)

def mirror(expr, mx=False, my=False, mz=False):
    """
    Mirrors space across the specified planes (YZ, XZ, XY).

    This creates reflections of the SDF across the specified planes by taking
    the absolute value of the corresponding coordinates.

    Args:
        expr: The SDF expression to mirror
        mx (bool): Mirror across the YZ plane (x=0). Defaults to False.
        my (bool): Mirror across the XZ plane (y=0). Defaults to False.
        mz (bool): Mirror across the XY plane (z=0). Defaults to False.

    Returns:
        A new SDF expression with mirrored space

    Raises:
        TypeError: If expr is not an SDF expression

    Examples:
        # Mirror a translated sphere across the YZ plane (x=0)
        sphere = fp.shape.sphere(1.0).translate(2, 0, 0)
        mirrored_sphere_x = fpm.mirror(sphere, mx=True)
        # Result will have spheres at both (2,0,0) and (-2,0,0)

        # Mirror across both YZ and XZ planes
        mirrored_sphere_xy = fpm.mirror(sphere, mx=True, my=True)

    Can be used as either:
    - fpm.mirror(expr, mx=False, my=False, mz=False)
    - expr.mirror(mx=False, my=False, mz=False) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")

    x, y, z = fp.x(), fp.y(), fp.z()

    new_x = abs(x) if mx else x
    new_y = abs(y) if my else y
    new_z = abs(z) if mz else z

    return remap_xyz(expr, new_x, new_y, new_z)

def symmetry(expr, sx=False, sy=False, sz=False):
    """
    Creates symmetry around the specified planes (YZ, XZ, XY).

    This is an alternative name for mirror, creating reflections across the
    specified planes.

    Args:
        expr: The SDF expression to make symmetric
        sx (bool): Create symmetry across the YZ plane (x=0). Defaults to False.
        sy (bool): Create symmetry across the XZ plane (y=0). Defaults to False.
        sz (bool): Create symmetry across the XY plane (z=0). Defaults to False.

    Returns:
        A new SDF expression with the specified symmetry

    Raises:
        TypeError: If expr is not an SDF expression

    Examples:
        # Create symmetry for a translated sphere across the YZ plane
        sphere = fp.shape.sphere(1.0).translate(2, 0, 0)
        symmetric_sphere_x = fpm.symmetry(sphere, sx=True)
        # Result will have spheres at both (2,0,0) and (-2,0,0)

        # Create symmetry across all planes (octant symmetry)
        symmetric_sphere_xyz = fpm.symmetry(sphere, sx=True, sy=True, sz=True)

    Can be used as either:
    - fpm.symmetry(expr, sx=False, sy=False, sz=False)
    - expr.symmetry(sx=False, sy=False, sz=False) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")

    return mirror(expr, mx=sx, my=sy, mz=sz)