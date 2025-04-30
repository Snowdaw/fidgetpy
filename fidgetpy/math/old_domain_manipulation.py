"""
Domain manipulation functions for Fidget.

This module provides domain manipulation operations for SDF expressions, including:
- Repetition (repeat, repeat_xyz)
- Mirroring (mirror_x, mirror_y, mirror_z)
"""

import fidgetpy as fp
from .basic_math import floor, abs
from .transformations import remap_xyz

def repeat(expr, period):
    """
    Repeats space along all axes with the same period.
    
    Can be used as either:
    - fpm.repeat(expr, period)
    - expr.repeat(period) (via extension)
    """
    return repeat_xyz(expr, period, period, period)

def repeat_xyz(expr, px, py, pz):
    """
    Repeats space along each axis with different periods.
    A period of 0 indicates no repetition along that axis.
    
    Can be used as either:
    - fpm.repeat_xyz(expr, px, py, pz)
    - expr.repeat_xyz(px, py, pz) (via extension)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Handle each axis - if period is 0, use the original coordinate
    mx = x if px == 0 else (x - px * floor(x / px + 0.5))
    my = y if py == 0 else (y - py * floor(y / py + 0.5))
    mz = z if pz == 0 else (z - pz * floor(z / pz + 0.5))
    
    return remap_xyz(expr, mx, my, mz)

def repeat_x(expr, period):
    """
    Repeats space along x-axis with the given period.
    Can be used as either:
    - fpm.repeat_x(expr, period)
    - expr.repeat_x(period) (via extension)
    """
    return repeat_xyz(expr, period, 0, 0)

def repeat_y(expr, period):
    """
    Repeats space along y-axis with the given period.
    Can be used as either:
    - fpm.repeat_y(expr, period)
    - expr.repeat_y(period) (via extension)
    """
    return repeat_xyz(expr, 0, period, 0)

def repeat_z(expr, period):
    """
    Repeats space along z-axis with the given period.
    Can be used as either:
    - fpm.repeat_z(expr, period)
    - expr.repeat_z(period) (via extension)
    """
    return repeat_xyz(expr, 0, 0, period)

def mirror_x(expr):
    """
    Mirrors space across the YZ plane.
    
    Can be used as either:
    - fpm.mirror_x(expr)
    - expr.mirror_x() (via extension)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, abs(x), y, z)

def mirror_y(expr):
    """
    Mirrors space across the XZ plane.
    
    Can be used as either:
    - fpm.mirror_y(expr)
    - expr.mirror_y() (via extension)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x, abs(y), z)

def mirror_z(expr):
    """
    Mirrors space across the XY plane.
    
    Can be used as either:
    - fpm.mirror_z(expr)
    - expr.mirror_z() (via extension)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x, y, abs(z))

def symmetry_x(expr):
    """
    Creates symmetry around the YZ plane.
    This is an alternative name for mirror_x.
    
    Can be used as either:
    - fpm.symmetry_x(expr)
    - expr.symmetry_x() (via extension)
    """
    return mirror_x(expr)

def symmetry_y(expr):
    """
    Creates symmetry around the XZ plane.
    This is an alternative name for mirror_y.
    
    Can be used as either:
    - fpm.symmetry_y(expr)
    - expr.symmetry_y() (via extension)
    """
    return mirror_y(expr)

def symmetry_z(expr):
    """
    Creates symmetry around the XY plane.
    This is an alternative name for mirror_z.
    
    Can be used as either:
    - fpm.symmetry_z(expr)
    - expr.symmetry_z() (via extension)
    """
    return mirror_z(expr)