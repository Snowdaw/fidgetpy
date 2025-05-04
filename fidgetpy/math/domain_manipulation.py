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

    # Determine which variables are needed based on repetition periods
    need_x = px != 0
    need_y = py != 0
    need_z = pz != 0

    # Only create variables that are needed for repetition
    # If a period is 0, we don't need to create that variable
    # since we'll use the identity mapping
    x_var = fp.x() if need_x else None
    y_var = fp.y() if need_y else None
    z_var = fp.z() if need_z else None

    # Apply the repetition formula only to variables we created
    mx = x_var if not need_x else (x_var - px * floor(x_var / px + 0.5))
    my = y_var if not need_y else (y_var - py * floor(y_var / py + 0.5))
    mz = z_var if not need_z else (z_var - pz * floor(z_var / pz + 0.5))

    # For axes that don't need repetition, use the identity map
    # These won't introduce variables
    if not need_x:
        mx = fp.x()
    if not need_y:
        my = fp.y()
    if not need_z:
        mz = fp.z()

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

    # Create only the variables we actually need to mirror
    # For variables we don't need, use a placeholder that
    # passes through the original coordinate without causing
    # variable creation
    
    # Determine which variables are needed based on mirror flags
    need_x = mx
    need_y = my
    need_z = mz
    
    # Create original variables only if needed for mirroring
    x_var = fp.x() if need_x else None
    y_var = fp.y() if need_y else None
    z_var = fp.z() if need_z else None
    
    # Apply transformations only on the variables we actually created
    new_x = abs(x_var) if mx else None
    new_y = abs(y_var) if my else None
    new_z = abs(z_var) if mz else None
    
    # Build an identity mapping for coordinates we don't need to mirror
    # These identities won't introduce new variables
    if not need_x:
        new_x = fp.x()  # Identity map for x
    if not need_y:
        new_y = fp.y()  # Identity map for y
    if not need_z:
        new_z = fp.z()  # Identity map for z
        
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