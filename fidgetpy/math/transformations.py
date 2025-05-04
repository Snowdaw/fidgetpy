"""
Transformation functions for Fidget.

This module provides transformation operations for SDF expressions, including:
- Translation (translate)
- Scaling (scale)
- Rotation (rotate)
- Affine transformations (remap_xyz, remap_affine)
- Matrix helpers for creating transformation matrices
"""

import math as py_math
import fidgetpy as fp
from .trigonometry import sin, cos

# ===== Transformations =====

def translate(expr, tx, ty, tz):
    """
    Translates an expression by the specified amounts.
    
    This function shifts the coordinate system, effectively moving the shape
    in the opposite direction.
    
    Args:
        expr: The SDF expression to translate
        tx: Translation amount along X axis
        ty: Translation amount along Y axis
        tz: Translation amount along Z axis
        
    Returns:
        A new SDF expression representing the translated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Translate a sphere 2 units along the x-axis
        sphere = fp.shape.sphere(1.0)
        translated_sphere = fpm.translate(sphere, 2.0, 0.0, 0.0)
        
        # Translate using method call syntax
        translated_sphere = sphere.translate(2.0, 0.0, 0.0)
        
    Can be used as either:
    - fpm.translate(expr, tx, ty, tz)
    - expr.translate(tx, ty, tz) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("translate requires an SDF expression")
    
    # Determine which variables are needed based on translation amounts
    need_x = tx != 0
    need_y = ty != 0
    need_z = tz != 0

    # Only create variables that are needed for translation
    x_var = fp.x() if need_x else None
    y_var = fp.y() if need_y else None
    z_var = fp.z() if need_z else None

    # Apply the translation only to variables we created
    new_x = x_var - tx if need_x else None
    new_y = y_var - ty if need_y else None
    new_z = z_var - tz if need_z else None

    # For axes that don't need translation, use identity map
    if not need_x:
        new_x = fp.x()
    if not need_y:
        new_y = fp.y()
    if not need_z:
        new_z = fp.z()

    return remap_xyz(expr, new_x, new_y, new_z)

def scale(expr, sx, sy, sz):
    """
    Scales an expression non-uniformly along each axis.

    This function scales the coordinate system, effectively scaling the shape
    by the reciprocal of the scale factors. Positive scale factors will preserve
    the shape's orientation, while negative scale factors will mirror the shape
    along that axis. A scale factor of 1 leaves the axis unchanged.

    Args:
        expr: The SDF expression to scale
        sx: Scale factor along X axis (non-zero)
        sy: Scale factor along Y axis (non-zero)
        sz: Scale factor along Z axis (non-zero)

    Returns:
        A new SDF expression representing the scaled shape

    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If any scale factor is zero

    Examples:
        # Scale a sphere to create an ellipsoid
        sphere = fp.shape.sphere(1.0)
        ellipsoid = fpm.scale(sphere, 2.0, 1.0, 0.5)

        # Scale using method call syntax
        ellipsoid = sphere.scale(2.0, 1.0, 0.5)

    Can be used as either:
    - fpm.scale(expr, sx, sy, sz)
    - expr.scale(sx, sy, sz) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("scale requires an SDF expression")

    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError("Scale factors cannot be zero")

    # Determine which variables are needed based on scale factors
    # Only when scale is 1.0 (identity) can we skip creating that variable
    need_x = sx != 1.0
    need_y = sy != 1.0
    need_z = sz != 1.0

    # Only create variables that are needed for scaling
    x_var = fp.x() if need_x else None
    y_var = fp.y() if need_y else None
    z_var = fp.z() if need_z else None

    # Apply the scaling only to variables we created
    new_x = x_var / sx if need_x else None
    new_y = y_var / sy if need_y else None
    new_z = z_var / sz if need_z else None

    # For axes that don't need scaling, use identity map
    if not need_x:
        new_x = fp.x()
    if not need_y:
        new_y = fp.y()
    if not need_z:
        new_z = fp.z()

    return remap_xyz(expr, new_x, new_y, new_z)

def rotate(expr, rx, ry, rz):
    """
    Rotates an expression around the Z, Y, and then X axes (intrinsic Tait-Bryan angles).

    This function applies rotations sequentially: first around the Z axis by rz,
    then around the new Y axis by ry, and finally around the newest X axis by rx.
    The coordinate system is rotated, effectively rotating the shape in the
    opposite direction around the axes.

    Args:
        expr: The SDF expression to rotate
        rx: Rotation angle around the final X axis in radians
        ry: Rotation angle around the intermediate Y axis in radians
        rz: Rotation angle around the initial Z axis in radians

    Returns:
        A new SDF expression representing the rotated shape

    Raises:
        TypeError: If expr is not an SDF expression

    Examples:
        # Rotate a box 45 degrees around X and 30 degrees around Z
        box = fp.shape.box(1.0, 2.0, 0.5)
        rotated_box = fpm.rotate(box, py_math.pi / 4, 0, py_math.pi / 6)

        # Rotate using method call syntax
        rotated_box = box.rotate(py_math.pi / 4, 0, py_math.pi / 6)

    Can be used as either:
    - fpm.rotate(expr, rx, ry, rz)
    - expr.rotate(rx, ry, rz) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate requires an SDF expression")
        
    # Special case: if all rotation angles are zero, we can use identity mapping
    if rx == 0 and ry == 0 and rz == 0:
        return expr  # No rotation needed, return original expression
        
    # For rotation, we generally need all three variables due to the nature of rotation matrices
    # except in the special case of zero rotation
    x, y, z = fp.x(), fp.y(), fp.z()

    # Apply rotations in reverse order to the coordinate system (Z -> Y -> X)
    # This corresponds to rotating the object X -> Y -> Z

    # Determine which rotations are actually needed
    need_z_rotation = rz != 0
    need_y_rotation = ry != 0
    need_x_rotation = rx != 0
    
    # Initialize with original coordinates
    x1, y1, z1 = x, y, z
    
    # Rotate around Z axis if needed
    if need_z_rotation:
        cz = cos(rz)
        sz = sin(rz)
        x1 = x * cz + y * sz
        y1 = -x * sz + y * cz
        z1 = z
    
    # Initialize second step with first step results
    x2, y2, z2 = x1, y1, z1
    
    # Rotate around Y axis if needed (using the Z-rotated coordinates)
    if need_y_rotation:
        cy = cos(ry)
        sy = sin(ry)
        x2 = x1 * cy - z1 * sy
        y2 = y1
        z2 = x1 * sy + z1 * cy
    
    # Initialize final results with second step results
    x_final, y_final, z_final = x2, y2, z2
    
    # Rotate around X axis if needed (using the Y-Z-rotated coordinates)
    if need_x_rotation:
        cx = cos(rx)
        sx = sin(rx)
        x_final = x2
        y_final = y2 * cx + z2 * sx
        z_final = -y2 * sx + z2 * cx

    return remap_xyz(expr, x_final, y_final, z_final)

def remap_xyz(expr, x_expr, y_expr, z_expr):
    """
    Remaps the coordinate system for an expression.
    
    This is a low-level function that allows arbitrary remapping of the
    coordinate system. It's used internally by other transformation functions
    but can also be used directly for custom transformations.
    
    Args:
        expr: The SDF expression to transform
        x_expr: Expression for the new x coordinate
        y_expr: Expression for the new y coordinate
        z_expr: Expression for the new z coordinate
        
    Returns:
        A new SDF expression with remapped coordinates
        
    Raises:
        TypeError: If expr is not an SDF expression or doesn't support remapping
        
    Examples:
        # Create a twisted shape by remapping coordinates
        box = fp.shape.box(1.0, 1.0, 1.0)
        twist_amount = 1.0
        twisted_box = fpm.remap_xyz(
            box,
            fp.x() * cos(twist_amount * fp.y()) - fp.z() * sin(twist_amount * fp.y()),
            fp.y(),
            fp.x() * sin(twist_amount * fp.y()) + fp.z() * cos(twist_amount * fp.y())
        )
        
    Can be used as either:
    - fpm.remap_xyz(expr, x_expr, y_expr, z_expr)
    - expr.remap_xyz(x_expr, y_expr, z_expr) (via extension)
    """
    if hasattr(expr, '_is_sdf_expr') and hasattr(expr, 'remap_xyz'):
        return expr.remap_xyz(x_expr, y_expr, z_expr)
    else:
        raise TypeError("Expression does not support remap_xyz")

def remap_affine(expr, matrix):
    """
    Applies an affine transformation to an expression.
    
    Can be used as either:
    - fpm.remap_affine(expr, matrix)
    - expr.remap_affine(matrix) (via extension)
    
    The matrix should be a flat array of 12 elements representing a 3x4 matrix:
    [m00, m01, m02, m10, m11, m12, m20, m21, m22, tx, ty, tz]
    
    Where:
    - m00-m22 is the 3x3 rotation/scaling matrix
    - tx, ty, tz is the translation vector
    
    Args:
        expr: The expression to transform
        matrix: A flat array of 12 elements representing a 3x4 affine matrix
        
    Returns:
        Transformed expression
   """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("remap_affine requires an SDF expression")
        
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 12:
        raise ValueError("Matrix must be a flat array of 12 elements")
        
    if hasattr(expr, 'remap_affine'):
        return expr.remap_affine(matrix)
    else:
        raise TypeError("Expression does not support remap_affine")

