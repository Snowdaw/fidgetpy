"""
Domain manipulation functions for Fidget.

This module provides domain manipulation operations for SDF expressions, including:
- Repetition (repeat, repeat_xyz)
- Mirroring (mirror_x, mirror_y, mirror_z)
- Symmetry (symmetry_x, symmetry_y, symmetry_z)
"""

import fidgetpy as fp
from .basic_math import floor, abs
from .transformations import remap_xyz

def repeat(expr, period):
    """
    Repeats space along all axes with the same period.
    
    Args:
        expr: The SDF expression to repeat
        period: The repetition period for all axes (must be > 0 for repetition to occur)
    
    Returns:
        A new SDF expression with repeated space
        
    Raises:
        TypeError: If expr is not an SDF expression or period is not a number
        ValueError: If period is negative
        
    Examples:
        # Repeat a sphere every 3 units in all directions
        sphere = fp.shape.sphere(1.0)
        repeated_sphere = fpm.repeat(sphere, 3.0)
        
    Can be used as either:
    - fpm.repeat(expr, period)
    - expr.repeat(period) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("First argument must be an SDF expression")
    
    if not isinstance(period, (int, float)) and not hasattr(period, '_is_sdf_expr'):
        raise TypeError("Period must be a number or SDF expression")
        
    if isinstance(period, (int, float)) and period < 0:
        raise ValueError("Period must be non-negative")
        
    return repeat_xyz(expr, period, period, period)

def repeat_xyz(expr, px, py, pz):
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
        repeated_sphere = fpm.repeat_xyz(sphere, 3.0, 3.0, 0.0)
        
    Can be used as either:
    - fpm.repeat_xyz(expr, px, py, pz)
    - expr.repeat_xyz(px, py, pz) (via extension)
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

def repeat_x(expr, period):
    """
    Repeats space along x-axis with the given period.
    
    Args:
        expr: The SDF expression to repeat
        period: The repetition period along the x-axis (0 for no repetition)
    
    Returns:
        A new SDF expression with repeated space along the x-axis
        
    Raises:
        TypeError: If expr is not an SDF expression or period is not a number
        ValueError: If period is negative
        
    Examples:
        # Repeat a sphere every 3 units along the x-axis
        sphere = fp.shape.sphere(1.0)
        repeated_sphere = fpm.repeat_x(sphere, 3.0)
        
    Can be used as either:
    - fpm.repeat_x(expr, period)
    - expr.repeat_x(period) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("First argument must be an SDF expression")
    
    if not isinstance(period, (int, float)) and not hasattr(period, '_is_sdf_expr'):
        raise TypeError("Period must be a number or SDF expression")
        
    if isinstance(period, (int, float)) and period < 0:
        raise ValueError("Period must be non-negative")
        
    return repeat_xyz(expr, period, 0, 0)

def repeat_y(expr, period):
    """
    Repeats space along y-axis with the given period.
    
    Args:
        expr: The SDF expression to repeat
        period: The repetition period along the y-axis (0 for no repetition)
    
    Returns:
        A new SDF expression with repeated space along the y-axis
        
    Raises:
        TypeError: If expr is not an SDF expression or period is not a number
        ValueError: If period is negative
        
    Examples:
        # Repeat a sphere every 3 units along the y-axis
        sphere = fp.shape.sphere(1.0)
        repeated_sphere = fpm.repeat_y(sphere, 3.0)
        
    Can be used as either:
    - fpm.repeat_y(expr, period)
    - expr.repeat_y(period) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("First argument must be an SDF expression")
    
    if not isinstance(period, (int, float)) and not hasattr(period, '_is_sdf_expr'):
        raise TypeError("Period must be a number or SDF expression")
        
    if isinstance(period, (int, float)) and period < 0:
        raise ValueError("Period must be non-negative")
        
    return repeat_xyz(expr, 0, period, 0)

def repeat_z(expr, period):
    """
    Repeats space along z-axis with the given period.
    
    Args:
        expr: The SDF expression to repeat
        period: The repetition period along the z-axis (0 for no repetition)
    
    Returns:
        A new SDF expression with repeated space along the z-axis
        
    Raises:
        TypeError: If expr is not an SDF expression or period is not a number
        ValueError: If period is negative
        
    Examples:
        # Repeat a sphere every 3 units along the z-axis
        sphere = fp.shape.sphere(1.0)
        repeated_sphere = fpm.repeat_z(sphere, 3.0)
        
    Can be used as either:
    - fpm.repeat_z(expr, period)
    - expr.repeat_z(period) (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("First argument must be an SDF expression")
    
    if not isinstance(period, (int, float)) and not hasattr(period, '_is_sdf_expr'):
        raise TypeError("Period must be a number or SDF expression")
        
    if isinstance(period, (int, float)) and period < 0:
        raise ValueError("Period must be non-negative")
        
    return repeat_xyz(expr, 0, 0, period)

def mirror_x(expr):
    """
    Mirrors space across the YZ plane (x=0).
    
    This creates a reflection of the SDF across the YZ plane, making
    the SDF symmetric with respect to the x-coordinate.
    
    Args:
        expr: The SDF expression to mirror
    
    Returns:
        A new SDF expression with mirrored space
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Mirror a translated sphere to create symmetry across the YZ plane
        sphere = fp.shape.sphere(1.0).translate(2, 0, 0)
        mirrored_sphere = fpm.mirror_x(sphere)
        # Result will have spheres at both (2,0,0) and (-2,0,0)
        
    Can be used as either:
    - fpm.mirror_x(expr)
    - expr.mirror_x() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, abs(x), y, z)

def mirror_y(expr):
    """
    Mirrors space across the XZ plane (y=0).
    
    This creates a reflection of the SDF across the XZ plane, making
    the SDF symmetric with respect to the y-coordinate.
    
    Args:
        expr: The SDF expression to mirror
    
    Returns:
        A new SDF expression with mirrored space
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Mirror a translated sphere to create symmetry across the XZ plane
        sphere = fp.shape.sphere(1.0).translate(0, 2, 0)
        mirrored_sphere = fpm.mirror_y(sphere)
        # Result will have spheres at both (0,2,0) and (0,-2,0)
        
    Can be used as either:
    - fpm.mirror_y(expr)
    - expr.mirror_y() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x, abs(y), z)

def mirror_z(expr):
    """
    Mirrors space across the XY plane (z=0).
    
    This creates a reflection of the SDF across the XY plane, making
    the SDF symmetric with respect to the z-coordinate.
    
    Args:
        expr: The SDF expression to mirror
    
    Returns:
        A new SDF expression with mirrored space
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Mirror a translated sphere to create symmetry across the XY plane
        sphere = fp.shape.sphere(1.0).translate(0, 0, 2)
        mirrored_sphere = fpm.mirror_z(sphere)
        # Result will have spheres at both (0,0,2) and (0,0,-2)
        
    Can be used as either:
    - fpm.mirror_z(expr)
    - expr.mirror_z() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x, y, abs(z))

def symmetry_x(expr):
    """
    Creates symmetry around the YZ plane (x=0).
    
    This is an alternative name for mirror_x.
    
    Args:
        expr: The SDF expression to make symmetric
    
    Returns:
        A new SDF expression with symmetry across the YZ plane
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Create symmetry for a translated sphere across the YZ plane
        sphere = fp.shape.sphere(1.0).translate(2, 0, 0)
        symmetric_sphere = fpm.symmetry_x(sphere)
        # Result will have spheres at both (2,0,0) and (-2,0,0)
        
    Can be used as either:
    - fpm.symmetry_x(expr)
    - expr.symmetry_x() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    return mirror_x(expr)

def symmetry_y(expr):
    """
    Creates symmetry around the XZ plane (y=0).
    
    This is an alternative name for mirror_y.
    
    Args:
        expr: The SDF expression to make symmetric
    
    Returns:
        A new SDF expression with symmetry across the XZ plane
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Create symmetry for a translated sphere across the XZ plane
        sphere = fp.shape.sphere(1.0).translate(0, 2, 0)
        symmetric_sphere = fpm.symmetry_y(sphere)
        # Result will have spheres at both (0,2,0) and (0,-2,0)
        
    Can be used as either:
    - fpm.symmetry_y(expr)
    - expr.symmetry_y() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    return mirror_y(expr)

def symmetry_z(expr):
    """
    Creates symmetry around the XY plane (z=0).
    
    This is an alternative name for mirror_z.
    
    Args:
        expr: The SDF expression to make symmetric
    
    Returns:
        A new SDF expression with symmetry across the XY plane
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Create symmetry for a translated sphere across the XY plane
        sphere = fp.shape.sphere(1.0).translate(0, 0, 2)
        symmetric_sphere = fpm.symmetry_z(sphere)
        # Result will have spheres at both (0,0,2) and (0,0,-2)
        
    Can be used as either:
    - fpm.symmetry_z(expr)
    - expr.symmetry_z() (via extension)
    """
    # Validate inputs
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("Argument must be an SDF expression")
        
    return mirror_z(expr)