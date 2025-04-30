"""
Transformation functions for Fidget.

This module provides transformation operations for SDF expressions, including:
- Translation (translate, translate_x, translate_y, translate_z)
- Scaling (scale, scale_xyz)
- Rotation (rotate_x, rotate_y, rotate_z)
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
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x - tx, y - ty, z - tz)

def translate_x(expr, tx):
    """
    Translates an expression along the X axis.
    
    This function shifts the coordinate system along the X axis, effectively
    moving the shape in the opposite direction.
    
    Args:
        expr: The SDF expression to translate
        tx: Translation amount along X axis
        
    Returns:
        A new SDF expression representing the translated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Translate a sphere 2 units along the x-axis
        sphere = fp.shape.sphere(1.0)
        translated_sphere = fpm.translate_x(sphere, 2.0)
        
        # Translate using method call syntax
        translated_sphere = sphere.translate_x(2.0)
        
    Can be used as either:
    - fpm.translate_x(expr, tx)
    - expr.translate_x(tx) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("translate_x requires an SDF expression")
        
    return translate(expr, tx, 0, 0)

def translate_y(expr, ty):
    """
    Translates an expression along the Y axis.
    
    Can be used as either:
    - fpm.translate_y(expr, ty)
    - expr.translate_y(ty) (via extension)
    
    Args:
        expr: The expression to translate
        ty: Translation amount along Y axis
        
    Returns:
        Translated expression
    """
    return translate(expr, 0, ty, 0)

def translate_z(expr, tz):
    """
    Translates an expression along the Z axis.
    
    Can be used as either:
    - fpm.translate_z(expr, tz)
    - expr.translate_z(tz) (via extension)
    
    Args:
        expr: The expression to translate
        tz: Translation amount along Z axis
        
    Returns:
        Translated expression
    """
    return translate(expr, 0, 0, tz)

def scale_xyz(expr, sx, sy, sz):
    """
    Scales an expression non-uniformly along each axis.
    
    This function scales the coordinate system, effectively scaling the shape
    by the reciprocal of the scale factors. Positive scale factors will preserve
    the shape's orientation, while negative scale factors will mirror the shape
    along that axis.
    
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
        ellipsoid = fpm.scale_xyz(sphere, 2.0, 1.0, 0.5)
        
        # Scale using method call syntax
        ellipsoid = sphere.scale_xyz(2.0, 1.0, 0.5)
        
    Can be used as either:
    - fpm.scale_xyz(expr, sx, sy, sz)
    - expr.scale_xyz(sx, sy, sz) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("scale_xyz requires an SDF expression")
        
    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError("Scale factors cannot be zero")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return remap_xyz(expr, x / sx, y / sy, z / sz)

def scale(expr, s):
    """
    Scales an expression uniformly.
    
    Can be used as either:
    - fpm.scale(expr, s)
    - expr.scale(s) (via extension)
    
    Args:
        expr: The expression to scale
        s: Uniform scale factor
        
    Returns:
        Scaled expression
    """
    if s == 0:
        raise ValueError("Scale factor cannot be zero")
        
    return scale_xyz(expr, s, s, s)

def rotate_x(expr, angle):
    """
    Rotates an expression around the X axis.
    
    Can be used as either:
    - fpm.rotate_x(expr, angle)
    - expr.rotate_x(angle) (via extension)
    
    Args:
        expr: The expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        Rotated expression
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate_x requires an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    c = cos(angle)
    s = sin(angle)
    return remap_xyz(expr, x, y * c - z * s, y * s + z * c)

def rotate_y(expr, angle):
    """
    Rotates an expression around the Y axis.
    
    Can be used as either:
    - fpm.rotate_y(expr, angle)
    - expr.rotate_y(angle) (via extension)
    
    Args:
        expr: The expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        Rotated expression
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate_y requires an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    c = cos(angle)
    s = sin(angle)
    return remap_xyz(expr, x * c + z * s, y, -x * s + z * c)

def rotate_z(expr, angle):
    """
    Rotates an expression around the Z axis.
    
    Can be used as either:
    - fpm.rotate_z(expr, angle)
    - expr.rotate_z(angle) (via extension)
    
    Args:
        expr: The expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        Rotated expression
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate_z requires an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    c = cos(angle)
    s = sin(angle)
    return remap_xyz(expr, x * c - y * s, x * s + y * c, z)

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

# Helper functions for creating common affine transformations

def make_translation_matrix(tx, ty, tz):
    """
    Creates a translation matrix.
    
    This function creates a 3x4 affine transformation matrix that represents
    a translation in 3D space.
    
    Args:
        tx: Translation along X axis
        ty: Translation along Y axis
        tz: Translation along Z axis
        
    Returns:
        A flat array of 12 elements representing a translation matrix
        
    Examples:
        # Create a translation matrix and apply it to a shape
        matrix = fpm.make_translation_matrix(1.0, 2.0, 3.0)
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    # Use regular Python math here since we're creating constants,
    # not SDF expressions
    return [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        float(tx), float(ty), float(tz)
    ]

def make_scaling_matrix(sx, sy, sz):
    """
    Creates a scaling matrix.
    
    Args:
        sx: Scale factor along X axis
        sy: Scale factor along Y axis
        sz: Scale factor along Z axis
        
    Returns:
        A flat array of 12 elements representing a scaling matrix
    """
    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError("Scale factors cannot be zero")
        
    return [
        float(sx), 0.0, 0.0,
        0.0, float(sy), 0.0,
        0.0, 0.0, float(sz),
        0.0, 0.0, 0.0
    ]

def make_rotation_x_matrix(angle):
    """
    Creates a rotation matrix around the X axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A flat array of 12 elements representing a rotation matrix
    """
    # Use py_math here since we're precomputing the matrix,
    # not creating SDF expressions
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        1.0, 0.0, 0.0,
        0.0, float(c), float(-s),
        0.0, float(s), float(c),
        0.0, 0.0, 0.0
    ]

def make_rotation_y_matrix(angle):
    """
    Creates a rotation matrix around the Y axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A flat array of 12 elements representing a rotation matrix
    """
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        float(c), 0.0, float(s),
        0.0, 1.0, 0.0,
        float(-s), 0.0, float(c),
        0.0, 0.0, 0.0
    ]

def make_rotation_z_matrix(angle):
    """
    Creates a rotation matrix around the Z axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A flat array of 12 elements representing a rotation matrix
    """
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        float(c), float(-s), 0.0,
        float(s), float(c), 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0
    ]

def combine_matrices(m1, m2):
    """
    Combines (multiplies) two transformation matrices.
    
    The result is equivalent to applying m2 first, then m1.
    This function allows you to chain multiple transformations together.
    
    Args:
        m1: First transformation matrix (applied second)
        m2: Second transformation matrix (applied first)
        
    Returns:
        Combined transformation matrix
        
    Raises:
        ValueError: If either input is not a flat array of 12 elements
        
    Examples:
        # Create a transformation that rotates then translates
        rotation = fpm.make_rotation_y_matrix(py_math.pi/4)
        translation = fpm.make_translation_matrix(1.0, 0.0, 0.0)
        
        # Combine them (translation applied after rotation)
        combined = fpm.combine_matrices(translation, rotation)
        
        # Apply to a shape
        transformed_shape = fpm.remap_affine(shape, combined)
    """
    if not isinstance(m1, (list, tuple)) or len(m1) != 12:
        raise ValueError("First matrix must be a flat array of 12 elements")
    if not isinstance(m2, (list, tuple)) or len(m2) != 12:
        raise ValueError("Second matrix must be a flat array of 12 elements")
    
    # Extract components from first matrix
    m1_00, m1_01, m1_02 = m1[0], m1[1], m1[2]
    m1_10, m1_11, m1_12 = m1[3], m1[4], m1[5]
    m1_20, m1_21, m1_22 = m1[6], m1[7], m1[8]
    m1_tx, m1_ty, m1_tz = m1[9], m1[10], m1[11]
    
    # Extract components from second matrix
    m2_00, m2_01, m2_02 = m2[0], m2[1], m2[2]
    m2_10, m2_11, m2_12 = m2[3], m2[4], m2[5]
    m2_20, m2_21, m2_22 = m2[6], m2[7], m2[8]
    m2_tx, m2_ty, m2_tz = m2[9], m2[10], m2[11]
    
    # Compute the 3x3 matrix multiplication
    r00 = m1_00*m2_00 + m1_01*m2_10 + m1_02*m2_20
    r01 = m1_00*m2_01 + m1_01*m2_11 + m1_02*m2_21
    r02 = m1_00*m2_02 + m1_01*m2_12 + m1_02*m2_22
    
    r10 = m1_10*m2_00 + m1_11*m2_10 + m1_12*m2_20
    r11 = m1_10*m2_01 + m1_11*m2_11 + m1_12*m2_21
    r12 = m1_10*m2_02 + m1_11*m2_12 + m1_12*m2_22
    
    r20 = m1_20*m2_00 + m1_21*m2_10 + m1_22*m2_20
    r21 = m1_20*m2_01 + m1_21*m2_11 + m1_22*m2_21
    r22 = m1_20*m2_02 + m1_21*m2_12 + m1_22*m2_22
    
    # Compute the translation part
    rtx = m1_00*m2_tx + m1_01*m2_ty + m1_02*m2_tz + m1_tx
    rty = m1_10*m2_tx + m1_11*m2_ty + m1_12*m2_tz + m1_ty
    rtz = m1_20*m2_tx + m1_21*m2_ty + m1_22*m2_tz + m1_tz
    
    return [
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
        rtx, rty, rtz
    ]