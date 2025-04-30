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
    
    This function shifts the coordinate system along the Y axis, effectively 
    moving the shape in the opposite direction.
    
    Args:
        expr: The SDF expression to translate
        ty: Translation amount along Y axis
        
    Returns:
        A new SDF expression representing the translated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Translate a sphere 2 units along the y-axis
        sphere = fp.shape.sphere(1.0)
        translated_sphere = fpm.translate_y(sphere, 2.0)
        
        # Translate using method call syntax
        translated_sphere = sphere.translate_y(2.0)
        
    Can be used as either:
    - fpm.translate_y(expr, ty)
    - expr.translate_y(ty) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("translate_y requires an SDF expression")
        
    return translate(expr, 0, ty, 0)

def translate_z(expr, tz):
    """
    Translates an expression along the Z axis.
    
    This function shifts the coordinate system along the Z axis, effectively 
    moving the shape in the opposite direction.
    
    Args:
        expr: The SDF expression to translate
        tz: Translation amount along Z axis
        
    Returns:
        A new SDF expression representing the translated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Translate a sphere 2 units along the z-axis
        sphere = fp.shape.sphere(1.0)
        translated_sphere = fpm.translate_z(sphere, 2.0)
        
        # Translate using method call syntax
        translated_sphere = sphere.translate_z(2.0)
        
    Can be used as either:
    - fpm.translate_z(expr, tz)
    - expr.translate_z(tz) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("translate_z requires an SDF expression")
        
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
    Scales an expression uniformly along all axes.
    
    This function scales the coordinate system uniformly, effectively scaling
    the shape by the reciprocal of the scale factor. A positive scale factor
    will preserve the shape's orientation, while a negative scale factor will
    invert the shape.
    
    Args:
        expr: The SDF expression to scale
        s: Uniform scale factor (non-zero)
        
    Returns:
        A new SDF expression representing the scaled shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If the scale factor is zero
        
    Examples:
        # Scale a sphere to double its size
        sphere = fp.shape.sphere(1.0)
        larger_sphere = fpm.scale(sphere, 2.0)
        
        # Scale using method call syntax
        larger_sphere = sphere.scale(2.0)
        
    Can be used as either:
    - fpm.scale(expr, s)
    - expr.scale(s) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("scale requires an SDF expression")
        
    if s == 0:
        raise ValueError("Scale factor cannot be zero")
        
    return scale_xyz(expr, s, s, s)

def rotate_x(expr, angle):
    """
    Rotates an expression around the X axis.
    
    This function rotates the coordinate system around the X axis, effectively
    rotating the shape in the opposite direction. The rotation follows the
    right-hand rule.
    
    Args:
        expr: The SDF expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        A new SDF expression representing the rotated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Rotate a cylinder 45 degrees around the x-axis
        cylinder = fp.shape.cylinder(1.0, 2.0)
        rotated_cylinder = fpm.rotate_x(cylinder, py_math.pi/4)
        
        # Rotate using method call syntax
        rotated_cylinder = cylinder.rotate_x(py_math.pi/4)
        
    Can be used as either:
    - fpm.rotate_x(expr, angle)
    - expr.rotate_x(angle) (via extension)
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
    
    This function rotates the coordinate system around the Y axis, effectively
    rotating the shape in the opposite direction. The rotation follows the
    right-hand rule.
    
    Args:
        expr: The SDF expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        A new SDF expression representing the rotated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Rotate a cylinder 45 degrees around the y-axis
        cylinder = fp.shape.cylinder(1.0, 2.0)
        rotated_cylinder = fpm.rotate_y(cylinder, py_math.pi/4)
        
        # Rotate using method call syntax
        rotated_cylinder = cylinder.rotate_y(py_math.pi/4)
        
    Can be used as either:
    - fpm.rotate_y(expr, angle)
    - expr.rotate_y(angle) (via extension)
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
    
    This function rotates the coordinate system around the Z axis, effectively
    rotating the shape in the opposite direction. The rotation follows the
    right-hand rule.
    
    Args:
        expr: The SDF expression to rotate
        angle: Rotation angle in radians
        
    Returns:
        A new SDF expression representing the rotated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Rotate a box 45 degrees around the z-axis
        box = fp.shape.box(1.0, 2.0, 0.5)
        rotated_box = fpm.rotate_z(box, py_math.pi/4)
        
        # Rotate using method call syntax
        rotated_box = box.rotate_z(py_math.pi/4)
        
    Can be used as either:
    - fpm.rotate_z(expr, angle)
    - expr.rotate_z(angle) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate_z requires an SDF expression")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    c = cos(angle)
    s = sin(angle)
    return remap_xyz(expr, x * c - y * s, x * s + y * c, z)

def remap_xyz(expr, new_x, new_y, new_z):
    """
    Remaps the coordinate system for an expression.
    
    This is a low-level function that allows arbitrary remapping of the
    coordinate system. It's used internally by other transformation functions
    but can also be used directly for custom transformations.
    
    Args:
        expr: The SDF expression to transform
        new_x: Expression for the new x coordinate
        new_y: Expression for the new y coordinate
        new_z: Expression for the new z coordinate
        
    Returns:
        A new SDF expression with remapped coordinates
        
    Raises:
        TypeError: If expr is not an SDF expression
        
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
    - fpm.remap_xyz(expr, new_x, new_y, new_z)
    - expr.remap_xyz(new_x, new_y, new_z) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("remap_xyz requires an SDF expression")
        
    if hasattr(expr, 'remap_xyz'):
        return expr.remap_xyz(new_x, new_y, new_z)
    else:
        raise TypeError("Expression does not support coordinate remapping")

def remap_affine(expr, matrix):
    """
    Applies an affine transformation to an expression using a 4x4 matrix.
    
    This function applies a general affine transformation defined by a 4x4 matrix
    to the coordinate system. The matrix should be in row-major order.
    
    Args:
        expr: The SDF expression to transform
        matrix: A 4x4 transformation matrix (list of 16 values in row-major order)
        
    Returns:
        A new SDF expression with the transformation applied
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If matrix is not a list of 16 values
        
    Examples:
        # Create a transformation matrix and apply it
        # This example creates a matrix that rotates around Y and translates
        matrix = fpm.make_rotation_y_matrix(py_math.pi/4)
        matrix = fpm.combine_matrices(
            matrix, 
            fpm.make_translation_matrix(1.0, 0.0, 0.0)
        )
        
        sphere = fp.shape.sphere(1.0)
        transformed_sphere = fpm.remap_affine(sphere, matrix)
        
    Can be used as either:
    - fpm.remap_affine(expr, matrix)
    - expr.remap_affine(matrix) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("remap_affine requires an SDF expression")
        
    if not isinstance(matrix, list) or len(matrix) != 16:
        raise ValueError("Matrix must be a list of 16 values (4x4 matrix in row-major order)")
        
    if hasattr(expr, 'remap_affine'):
        return expr.remap_affine(matrix)
    else:
        # Fall back to manual implementation using remap_xyz
        x, y, z = fp.x(), fp.y(), fp.z()
        
        # Extract matrix components (row-major order)
        m00, m01, m02, m03 = matrix[0], matrix[1], matrix[2], matrix[3]
        m10, m11, m12, m13 = matrix[4], matrix[5], matrix[6], matrix[7]
        m20, m21, m22, m23 = matrix[8], matrix[9], matrix[10], matrix[11]
        
        # Apply the transformation (ignoring the homogeneous component for simplicity)
        new_x = m00 * x + m01 * y + m02 * z + m03
        new_y = m10 * x + m11 * y + m12 * z + m13
        new_z = m20 * x + m21 * y + m22 * z + m23
        
        return remap_xyz(expr, new_x, new_y, new_z)

# ===== Matrix Helpers =====

def make_translation_matrix(tx, ty, tz):
    """
    Creates a 4x4 translation matrix.
    
    Args:
        tx: Translation along X axis
        ty: Translation along Y axis
        tz: Translation along Z axis
        
    Returns:
        A 4x4 translation matrix as a list of 16 values in row-major order
        
    Examples:
        # Create a translation matrix
        matrix = fpm.make_translation_matrix(1.0, 2.0, 3.0)
        
        # Apply it to a shape
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    return [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0
    ]

def make_scaling_matrix(sx, sy, sz):
    """
    Creates a 4x4 scaling matrix.
    
    Args:
        sx: Scale factor along X axis (non-zero)
        sy: Scale factor along Y axis (non-zero)
        sz: Scale factor along Z axis (non-zero)
        
    Returns:
        A 4x4 scaling matrix as a list of 16 values in row-major order
        
    Raises:
        ValueError: If any scale factor is zero
        
    Examples:
        # Create a scaling matrix
        matrix = fpm.make_scaling_matrix(2.0, 1.0, 0.5)
        
        # Apply it to a shape
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError("Scale factors cannot be zero")
        
    return [
        sx,  0.0, 0.0, 0.0,
        0.0, sy,  0.0, 0.0,
        0.0, 0.0, sz,  0.0,
        0.0, 0.0, 0.0, 1.0
    ]

def make_rotation_x_matrix(angle):
    """
    Creates a 4x4 rotation matrix for rotation around the X axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A 4x4 rotation matrix as a list of 16 values in row-major order
        
    Examples:
        # Create a rotation matrix for 45 degrees around X
        matrix = fpm.make_rotation_x_matrix(py_math.pi/4)
        
        # Apply it to a shape
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, c,   -s,  0.0,
        0.0, s,   c,   0.0,
        0.0, 0.0, 0.0, 1.0
    ]

def make_rotation_y_matrix(angle):
    """
    Creates a 4x4 rotation matrix for rotation around the Y axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A 4x4 rotation matrix as a list of 16 values in row-major order
        
    Examples:
        # Create a rotation matrix for 45 degrees around Y
        matrix = fpm.make_rotation_y_matrix(py_math.pi/4)
        
        # Apply it to a shape
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        c,   0.0, s,   0.0,
        0.0, 1.0, 0.0, 0.0,
        -s,  0.0, c,   0.0,
        0.0, 0.0, 0.0, 1.0
    ]

def make_rotation_z_matrix(angle):
    """
    Creates a 4x4 rotation matrix for rotation around the Z axis.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        A 4x4 rotation matrix as a list of 16 values in row-major order
        
    Examples:
        # Create a rotation matrix for 45 degrees around Z
        matrix = fpm.make_rotation_z_matrix(py_math.pi/4)
        
        # Apply it to a shape
        transformed_shape = fpm.remap_affine(shape, matrix)
    """
    c = py_math.cos(angle)
    s = py_math.sin(angle)
    return [
        c,   -s,  0.0, 0.0,
        s,   c,   0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]

def combine_matrices(a, b):
    """
    Combines two 4x4 transformation matrices by multiplication.
    
    The resulting matrix represents applying transformation b followed by
    transformation a.
    
    Args:
        a: First 4x4 matrix (applied second)
        b: Second 4x4 matrix (applied first)
        
    Returns:
        A new 4x4 matrix representing the combined transformation
        
    Raises:
        ValueError: If either input is not a list of 16 values
        
    Examples:
        # Create a transformation that rotates then translates
        rotation = fpm.make_rotation_y_matrix(py_math.pi/4)
        translation = fpm.make_translation_matrix(1.0, 0.0, 0.0)
        
        # Combine them (translation applied after rotation)
        combined = fpm.combine_matrices(translation, rotation)
        
        # Apply to a shape
        transformed_shape = fpm.remap_affine(shape, combined)
    """
    if not isinstance(a, list) or len(a) != 16:
        raise ValueError("First matrix must be a list of 16 values (4x4 matrix)")
    if not isinstance(b, list) or len(b) != 16:
        raise ValueError("Second matrix must be a list of 16 values (4x4 matrix)")
        
    result = [0.0] * 16
    
    # Matrix multiplication (row-major order)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i*4 + j] += a[i*4 + k] * b[k*4 + j]
                
    return result