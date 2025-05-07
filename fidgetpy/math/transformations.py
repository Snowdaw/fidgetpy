"""
Transformation functions for Fidget.

This module provides transformation operations for SDF expressions, including:
- Translation (translate)
- Scaling (scale)
- Rotation (rotate)
- Reflection (reflect, reflect_axis, reflect_plane)
- Symmetry (symmetric)
- Deformations (twist, taper, shear, revolve)
- Attraction/Repulsion (attract, repel)
- Morphing (morph, extrude_z)
- Affine transformations (remap_xyz, remap_affine)
- Matrix helpers for creating transformation matrices
"""

import math as py_math
import fidgetpy as fp
from .trigonometry import sin, cos
from .basic_math import sqrt, pow

# ===== Transformations =====

def translate(expr, tx, ty, tz, pivot_x=None, pivot_y=None, pivot_z=None):
    """
    Translates an expression by the specified amounts, optionally from a pivot point.
    
    This function shifts the coordinate system, effectively moving the shape
    in the opposite direction.
    
    Args:
        expr: The SDF expression to translate
        tx: Translation amount along X axis
        ty: Translation amount along Y axis
        tz: Translation amount along Z axis
        pivot_x: Optional X coordinate of pivot point. If provided with pivot_y and pivot_z, translates relative to this point.
        pivot_y: Optional Y coordinate of pivot point. If provided with pivot_x and pivot_z, translates relative to this point.
        pivot_z: Optional Z coordinate of pivot point. If provided with pivot_x and pivot_y, translates relative to this point.
        
    Returns:
        A new SDF expression representing the translated shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Translate a sphere 2 units along the x-axis
        sphere = fp.shape.sphere(1.0)
        translated_sphere = fpm.translate(sphere, 2.0, 0.0, 0.0)
        
        # Translate using method call syntax with a pivot point
        # Moving a shape 2 units in the +X direction from pivot point (1,1,1)
        # The pivot parameters indicate the center point around which to translate
        pivot_translated = sphere.translate(2.0, 0.0, 0.0, pivot_x=1.0, pivot_y=1.0, pivot_z=1.0)
        
    Can be used as either:
    - fpm.translate(expr, tx, ty, tz, pivot_x=None, pivot_y=None, pivot_z=None)
    - expr.translate(tx, ty, tz, pivot_x=None, pivot_y=None, pivot_z=None) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("translate requires an SDF expression")
    
    # Check if pivot point is provided (all three coordinates must be provided)
    has_pivot = pivot_x is not None and pivot_y is not None and pivot_z is not None
    
    if has_pivot:
        # For pivot-based translation, we need to:
        # 1. Apply the translation vector to the pivot point
        # 2. Calculate the difference between the new pivot position and original pivot position
        # This gives us the effective translation for each point in the space
        
        # New position of pivot after translation
        new_pivot_x = pivot_x + tx
        new_pivot_y = pivot_y + ty
        new_pivot_z = pivot_z + tz
        
        # Calculate the translation vector for each point in space
        # For this, we find where each point would end up relative to the new pivot
        x = fp.x()
        y = fp.y()
        z = fp.z()
        
        # Translate coordinates to the new positions relative to the moved pivot
        new_x = x - (new_pivot_x - pivot_x)
        new_y = y - (new_pivot_y - pivot_y)
        new_z = z - (new_pivot_z - pivot_z)
        
        return remap_xyz(expr, new_x, new_y, new_z)
    
    # Standard translation without pivot
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

def scale(expr, sx, sy, sz, pivot_x=None, pivot_y=None, pivot_z=None):
    """
    Scales an expression non-uniformly along each axis from a pivot point.

    This function scales the coordinate system, effectively scaling the shape
    by the reciprocal of the scale factors. Positive scale factors will preserve
    the shape's orientation, while negative scale factors will mirror the shape
    along that axis. A scale factor of 1 leaves the axis unchanged.

    Args:
        expr: The SDF expression to scale
        sx: Scale factor along X axis (non-zero)
        sy: Scale factor along Y axis (non-zero)
        sz: Scale factor along Z axis (non-zero)
        pivot_x: Optional X coordinate of pivot point. If provided with pivot_y and pivot_z, scales around this point.
        pivot_y: Optional Y coordinate of pivot point. If provided with pivot_x and pivot_z, scales around this point.
        pivot_z: Optional Z coordinate of pivot point. If provided with pivot_x and pivot_y, scales around this point.

    Returns:
        A new SDF expression representing the scaled shape

    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If any scale factor is zero

    Examples:
        # Scale a sphere to create an ellipsoid
        sphere = fp.shape.sphere(1.0)
        ellipsoid = fpm.scale(sphere, 2.0, 1.0, 0.5)

        # Scale using method call syntax with a pivot point
        translated_sphere = sphere.translate(2.0, 1.0, 0.5)
        # Scale around the translated sphere's center (2,1,0.5)
        # Simply provide the actual pivot point - no need for negation
        scaled_sphere = translated_sphere.scale(2.0, 1.0, 0.5, pivot_x=2.0, pivot_y=1.0, pivot_z=0.5)

    Can be used as either:
    - fpm.scale(expr, sx, sy, sz, pivot_x=None, pivot_y=None, pivot_z=None)
    - expr.scale(sx, sy, sz, pivot_x=None, pivot_y=None, pivot_z=None) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("scale requires an SDF expression")

    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError("Scale factors cannot be zero")
        
    # Check if pivot point is provided (all three coordinates must be provided)
    has_pivot = pivot_x is not None and pivot_y is not None and pivot_z is not None
    
    # Handle scaling around a pivot point
    if has_pivot:
        # When user provides a pivot point, they should expect the transformation to occur
        # around that exact point, without needing to understand SDF coordinate transformations.
        # We handle the necessary negations internally to make the API more intuitive.
        
        # 1. Translate pivot point to origin (SDF transforms coordinate space in opposite direction)
        translated = translate(expr, -pivot_x, -pivot_y, -pivot_z)
        
        # 2. Apply scaling (using the simpler non-pivot version)
        scaled = scale(translated, sx, sy, sz)
        
        # 3. Translate back to original position
        return translate(scaled, pivot_x, pivot_y, pivot_z)
    
    # Original scaling logic (around origin)
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

def rotate(expr, rx, ry, rz, pivot_x=None, pivot_y=None, pivot_z=None):
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
        pivot_x: Optional X coordinate of pivot point. If provided with pivot_y and pivot_z, rotates around this point.
        pivot_y: Optional Y coordinate of pivot point. If provided with pivot_x and pivot_z, rotates around this point.
        pivot_z: Optional Z coordinate of pivot point. If provided with pivot_x and pivot_y, rotates around this point.

    Returns:
        A new SDF expression representing the rotated shape

    Raises:
        TypeError: If expr is not an SDF expression

    Examples:
        # Rotate a box 45 degrees around X and 30 degrees around Z
        box = fp.shape.box(1.0, 2.0, 0.5)
        rotated_box = fpm.rotate(box, py_math.pi / 4, 0, py_math.pi / 6)

        # Rotate using method call syntax with pivot point
        translated_box = box.translate(2.0, 1.0, 0.5)
        # Rotate around the box's center, not the origin
        # Simply provide the actual pivot point - no need to understand SDF internals
        rotated_box = translated_box.rotate(py_math.pi/4, 0, 0, pivot_x=2.0, pivot_y=1.0, pivot_z=0.5)

    Can be used as either:
    - fpm.rotate(expr, rx, ry, rz, pivot_x=None, pivot_y=None, pivot_z=None)
    - expr.rotate(rx, ry, rz, pivot_x=None, pivot_y=None, pivot_z=None) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("rotate requires an SDF expression")
        
    # Special case: if all rotation angles are zero, we can use identity mapping
    if rx == 0 and ry == 0 and rz == 0:
        return expr  # No rotation needed, return original expression
        
    # Check if pivot point is provided (all three coordinates must be provided)
    has_pivot = pivot_x is not None and pivot_y is not None and pivot_z is not None
        
    # Handle rotation around a pivot point
    if has_pivot:
        # When user provides a pivot point, they should expect the rotation to occur
        # around that exact point, without needing to understand SDF coordinate transformations.
        # We handle the necessary negations internally to make the API more intuitive.
        
        # 1. Translate pivot point to origin (SDF transforms coordinate space in opposite direction)
        translated = translate(expr, -pivot_x, -pivot_y, -pivot_z)
        
        # 2. Apply rotation (using the simpler non-pivot version)
        rotated = rotate(translated, rx, ry, rz)
        
        # 3. Translate back to original position
        return translate(rotated, pivot_x, pivot_y, pivot_z)
        
    # Original rotation logic (around origin)
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

def reflect(expr, nx, ny, nz, d=0):
    """
    Reflects an expression across a plane defined by normal (nx, ny, nz) and distance d.
    
    This function reflects the coordinate system, effectively reflecting the shape
    across the plane defined by the equation nx*x + ny*y + nz*z = d.
    
    Args:
        expr: The SDF expression to reflect
        nx: X component of the plane normal vector
        ny: Y component of the plane normal vector
        nz: Z component of the plane normal vector
        d: Distance of the plane from the origin (default: 0)
        
    Returns:
        A new SDF expression representing the reflected shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If the normal vector is zero (nx=ny=nz=0)
        
    Examples:
        # Reflect a sphere across the XY plane (z=0)
        sphere = fp.shape.sphere(1.0).translate(0, 0, 2.0)
        reflected_sphere = fpm.reflect(sphere, 0, 0, 1, 0)
        
        # Reflect using method call syntax
        reflected_sphere = sphere.reflect(0, 0, 1, 0)
        
    Can be used as either:
    - fpm.reflect(expr, nx, ny, nz, d)
    - expr.reflect(nx, ny, nz, d) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("reflect requires an SDF expression")
        
    if nx == 0 and ny == 0 and nz == 0:
        raise ValueError("Normal vector cannot be zero")
        
    # Normalization factor
    norm_sq = nx*nx + ny*ny + nz*nz
    
    # Create coordinate variables
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Distance from point to plane
    dist = (nx*x + ny*y + nz*z - d) / norm_sq
    
    # Reflection formula: p' = p - 2(n·(p-p0))n
    # Where p is the point, n is the normal, and p0 is a point on the plane
    new_x = x - 2*nx*dist
    new_y = y - 2*ny*dist
    new_z = z - 2*nz*dist
    
    return remap_xyz(expr, new_x, new_y, new_z)

def reflect_axis(expr, axis='x', offset=0):
    """
    Reflects an expression across a coordinate plane.
    
    This function reflects the coordinate system across one of the coordinate
    planes (YZ, XZ, or XY), effectively reflecting the shape.
    
    Args:
        expr: The SDF expression to reflect
        axis: The axis perpendicular to the reflection plane ('x', 'y', or 'z')
        offset: Distance of the reflection plane from the origin (default: 0)
        
    Returns:
        A new SDF expression representing the reflected shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Examples:
        # Reflect a shape across the YZ plane (x=0)
        box = fp.shape.box(1.0, 2.0, 0.5).translate(2.0, 0, 0)
        reflected_box = fpm.reflect_axis(box, 'x')
        
        # Reflect across a custom plane (y=3)
        reflected_box = fpm.reflect_axis(box, 'y', 3.0)
        
    Can be used as either:
    - fpm.reflect_axis(expr, axis, offset)
    - expr.reflect_axis(axis, offset) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("reflect_axis requires an SDF expression")
    
    if axis.lower() == 'x':
        return reflect(expr, 1, 0, 0, offset)
    elif axis.lower() == 'y':
        return reflect(expr, 0, 1, 0, offset)
    elif axis.lower() == 'z':
        return reflect(expr, 0, 0, 1, offset)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def reflect_plane(expr, plane='xy'):
    """
    Reflects an expression by swapping two coordinates.
    
    This function reflects the shape across one of the diagonal planes
    (x=y, y=z, or x=z) by swapping the appropriate coordinates.
    
    Args:
        expr: The SDF expression to reflect
        plane: The reflection plane ('xy', 'yz', or 'xz')
        
    Returns:
        A new SDF expression representing the reflected shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If plane is not 'xy', 'yz', or 'xz'
        
    Examples:
        # Reflect a shape across the x=y plane
        box = fp.shape.box(1.0, 2.0, 0.5)
        reflected_box = fpm.reflect_plane(box, 'xy')
        
    Can be used as either:
    - fpm.reflect_plane(expr, plane)
    - expr.reflect_plane(plane) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("reflect_plane requires an SDF expression")
    
    plane = plane.lower()
    if plane == 'xy':
        return remap_xyz(expr, fp.y(), fp.x(), fp.z())
    elif plane == 'yz':
        return remap_xyz(expr, fp.x(), fp.z(), fp.y())
    elif plane == 'xz':
        return remap_xyz(expr, fp.z(), fp.y(), fp.x())
    else:
        raise ValueError("Plane must be 'xy', 'yz', or 'xz'")

def symmetric(expr, axis='x'):
    """
    Creates a symmetric shape by clipping at the origin and reflecting across a plane.
    
    This function first clips the given shape at the specified axis's origin,
    then duplicates and reflects the remaining shape across that plane,
    creating perfect symmetry along the specified axis.
    
    Args:
        expr: The SDF expression to make symmetric
        axis: The axis along which to create symmetry ('x', 'y', or 'z')
        
    Returns:
        A new SDF expression representing the symmetrized shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Examples:
        # Create a symmetric shape from a sphere positioned at x=2
        sphere = fp.shape.sphere(1.0).translate(2.0, 0.0, 0.0)
        symmetric_sphere = fpm.symmetric(sphere, 'x')
        
        # Create a symmetric shape along the z axis
        cone = fp.shape.cone(1.0, 2.0).translate(0.0, 0.0, 1.0)
        symmetric_cone = fpm.symmetric(cone, 'z')
        
    Can be used as either:
    - fpm.symmetric(expr, axis)
    - expr.symmetric(axis) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("symmetric requires an SDF expression")
    
    # Validate axis
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Create a mapping for the clipping operation
    new_coords = {'x': x, 'y': y, 'z': z}  # Start with identity mapping
    
    # Replace the specified axis with max(axis, 0) to clip at the origin
    new_coords[axis] = coords[axis].max(0)
    
    # Clip the shape at the origin
    clipped = expr.remap_xyz(new_coords['x'], new_coords['y'], new_coords['z'])
    
    # Reflect the clipped part across the appropriate plane
    reflected = reflect_axis(clipped, axis)
    
    # Combine original clipped and reflected parts
    return clipped.min(reflected)

def twist(expr, amount):
    """
    Applies a twist deformation around the Z axis.
    
    This function twists the coordinate system around the Z axis, with the amount
    of twist proportional to the Z coordinate. The twist angle at a given Z
    coordinate is Z * amount.
    
    Args:
        expr: The SDF expression to twist
        amount: The amount of twist per unit Z (in radians per unit)
        
    Returns:
        A new SDF expression representing the twisted shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        
    Examples:
        # Create a twisted box
        box = fp.shape.box(1.0, 1.0, 2.0)
        twisted_box = fpm.twist(box, 0.5)  # Twist 0.5 radians per unit Z
        
    Can be used as either:
    - fpm.twist(expr, amount)
    - expr.twist(amount) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("twist requires an SDF expression")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate rotation angle based on z coordinate
    angle = z * amount
    
    # Calculate sines and cosines
    c = cos(angle)
    s = sin(angle)
    
    # Apply twist transformation
    new_x = x * c - y * s
    new_y = x * s + y * c
    new_z = z
    
    return remap_xyz(expr, new_x, new_y, new_z)

def taper(expr, axis='z', plane_axes=None, base=0.0, height=1.0, scale=0.5, base_scale=1.0):
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("symmetric requires an SDF expression")
    
    # Validate axis
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Create a mapping for the clipping operation
    new_coords = {'x': x, 'y': y, 'z': z}  # Start with identity mapping
    
    # Replace the specified axis with max(axis, 0) to clip at the origin
    new_coords[axis] = coords[axis].max(0)
    
    # Clip the shape at the origin
    clipped = expr.remap_xyz(new_coords['x'], new_coords['y'], new_coords['z'])
    
    # Reflect the clipped part across the appropriate plane
    reflected = reflect_axis(clipped, axis)
    
    # Combine original clipped and reflected parts
    return clipped.min(reflected)
    Tapers a shape along one axis by scaling coordinates in the perpendicular plane.
    
    This function scales coordinates in a plane perpendicular to the specified axis,
    based on the position along that axis. The scaling factor varies linearly from
    base_scale at position=base to scale at position=base+height.
    
    Args:
        expr: The SDF expression to taper
        axis: The axis along which to taper ('x', 'y', or 'z') (default: 'z')
        plane_axes: Tuple of axes to scale (if None, automatically selects the two perpendicular axes)
        base: The position along the taper axis where tapering begins
        height: The distance over which tapering occurs
        scale: The scale factor at base+height
        base_scale: The scale factor at base (default: 1.0)
        
    Returns:
        A new SDF expression representing the tapered shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If height is zero or axis is invalid
        
    Examples:
        # Create a tapered cylinder (traditional taper_xy_z equivalent)
        cylinder = fp.shape.cylinder(1.0, 2.0)
        tapered = fpm.taper(cylinder, axis='z', base=0.0, height=2.0, scale=0.5)
        
        # Create a tapered box along X axis
        box = fp.shape.box(2.0, 1.0, 1.0)
        tapered = fpm.taper(box, axis='x', base=-1.0, height=2.0, scale=0.5)
        
    Can be used as either:
    - fpm.taper(expr, axis, plane_axes, base, height, scale, base_scale)
    - expr.taper(axis, plane_axes, base, height, scale, base_scale) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("taper requires an SDF expression")
        
    if height == 0:
        raise ValueError("Taper height cannot be zero")
    
    # Validate axis
    
    if axis not in ['x', 'y', 'z', 'X', 'Y', 'Z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
        
    axis = axis.lower()
    # Automatically determine plane axes if not specified
    if plane_axes is None:
        all_axes = {'x', 'y', 'z'}
        plane_axes = list(all_axes - {axis})
    else:
        # Validate plane_axes
        if len(plane_axes) != 2:
            raise ValueError("plane_axes must contain exactly two axis names")
            
        for a in plane_axes:
            if a.lower() not in ['x', 'y', 'z']:
                raise ValueError("plane_axes can only contain 'x', 'y', or 'z'")
                
        if axis in [a.lower() for a in plane_axes]:
            raise ValueError("taper axis must be perpendicular to plane_axes")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Get the taper axis coordinate
    taper_axis_coord = coords[axis]
    
    # Get the plane coordinates to scale
    plane_coords = [coords[a.lower()] for a in plane_axes]
    
    # Calculate scale factor based on position along taper axis
    factor = (taper_axis_coord - base) / height
    # Clamp factor between 0 and 1
    factor = factor.min(1.0).max(0.0)
    # Interpolate between base_scale and scale
    current_scale = base_scale + (scale - base_scale) * factor
    
    # Scale the plane coordinates
    scaled_coords = [coord / current_scale for coord in plane_coords]
    
    # Create new coordinate mapping
    new_coords = {'x': x, 'y': y, 'z': z}  # Start with identity mapping
    
    # Update with scaled coordinates
    for i, axis_name in enumerate([a.lower() for a in plane_axes]):
        new_coords[axis_name] = scaled_coords[i]
    
    # Apply the remapping
    return remap_xyz(expr, new_coords['x'], new_coords['y'], new_coords['z'])

# Legacy taper_xy_z function removed in favor of the more flexible taper function

def shear(expr, shear_axis='x', control_axis='y', base=0.0, height=1.0, offset=1.0, base_offset=0.0):
    """
    Shears a shape along one axis as a function of another.
    
    This function applies a shear transformation that displaces points along
    the shear_axis based on their position along the control_axis. The offset
    varies linearly from base_offset at control_axis=base to offset at control_axis=base+height.
    
    Args:
        expr: The SDF expression to shear
        shear_axis: The axis to shear along ('x', 'y', or 'z')
        control_axis: The axis that controls the shear amount ('x', 'y', or 'z')
        base: The position along the control_axis where shearing begins
        height: The distance over which shearing occurs
        offset: The displacement along shear_axis at control_axis=base+height
        base_offset: The displacement along shear_axis at control_axis=base (default: 0.0)
        
    Returns:
        A new SDF expression representing the sheared shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If height is zero or axes are invalid
        
    Examples:
        # Create a sheared box (traditional shear_x_y equivalent)
        box = fp.shape.box(1.0, 2.0, 0.5)
        sheared = fpm.shear(box, shear_axis='x', control_axis='y', base=0.0, height=2.0, offset=1.0)
        
        # Create a box sheared along Z as a function of X
        box = fp.shape.box(2.0, 1.0, 1.0)
        sheared = fpm.shear(box, shear_axis='z', control_axis='x', base=-1.0, height=2.0, offset=0.5)
        
    Can be used as either:
    - fpm.shear(expr, shear_axis, control_axis, base, height, offset, base_offset)
    - expr.shear(shear_axis, control_axis, base, height, offset, base_offset) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("shear requires an SDF expression")
        
    if height == 0:
        raise ValueError("Shear height cannot be zero")
    
    # Validate axes
    shear_axis = shear_axis.lower()
    control_axis = control_axis.lower()
    
    if shear_axis not in ['x', 'y', 'z']:
        raise ValueError("Shear axis must be 'x', 'y', or 'z'")
        
    if control_axis not in ['x', 'y', 'z']:
        raise ValueError("Control axis must be 'x', 'y', or 'z'")
        
    if shear_axis == control_axis:
        raise ValueError("Shear and control axes must be different")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Get the control axis coordinate
    control_coord = coords[control_axis]
    
    # Calculate offset factor based on control coordinate
    factor = (control_coord - base) / height
    # Clamp factor between 0 and 1
    factor = factor.min(1.0).max(0.0)
    # Interpolate between base_offset and offset
    current_offset = base_offset + (offset - base_offset) * factor
    
    # Create new coordinate mapping
    new_coords = {'x': x, 'y': y, 'z': z}  # Start with identity mapping
    
    # Apply shear to the appropriate axis
    new_coords[shear_axis] = coords[shear_axis] - current_offset
    
    # Apply the remapping
    return remap_xyz(expr, new_coords['x'], new_coords['y'], new_coords['z'])

# Legacy shear_x_y function removed in favor of the more flexible shear function

def revolve(expr, axis='y', offset=0.0):
    """
    Revolves a 2D shape about an axis.
    
    This function revolves a 2D shape around a specified axis, creating a 3D solid
    of revolution. The axis of revolution is parallel to the specified axis and
    offset from the origin by the given amount.
    
    Args:
        expr: The SDF expression to revolve (should be a 2D shape)
        axis: The axis of revolution ('x', 'y', or 'z') (default: 'y')
        offset: Offset of the revolution axis from the origin (default: 0.0)
        
    Returns:
        A new SDF expression representing the revolved shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Examples:
        # Create a torus by revolving a circle around the Y axis
        circle = fp.shape.circle(0.5).translate(2.0, 0.0, 0.0)
        torus = fpm.revolve(circle, axis='y')
        
        # Create a vase by revolving a profile around the Z axis
        profile = fp.shape.box(0.5, 2.0, 0.1).translate(1.0, 0.0, 0.0)
        vase = fpm.revolve(profile, axis='z')
        
    Can be used as either:
    - fpm.revolve(expr, axis, offset)
    - expr.revolve(axis, offset) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("revolve requires an SDF expression")
    
    # Validate axis
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Determine the plane perpendicular to the revolution axis
    if axis == 'x':
        # Revolve around X axis, use YZ plane
        r = fpm.sqrt(fpm.pow(y, 2) + fpm.pow(z, 2))
        # Remap to use Y as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, x, 0)
    elif axis == 'y':
        # Revolve around Y axis, use XZ plane
        r = fpm.sqrt(fpm.pow(x, 2) + fpm.pow(z, 2))
        # Remap to use Y as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, y, 0)
    else:  # axis == 'z'
        # Revolve around Z axis, use XY plane
        r = fpm.sqrt(fpm.pow(x, 2) + fpm.pow(y, 2))
        # Remap to use Z as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, z, 0)

def revolve(expr, axis='y', offset=0.0):
    """
    Revolves a 2D shape about an axis.
    
    This function revolves a 2D shape around a specified axis, creating a 3D solid
    of revolution. The axis of revolution is parallel to the specified axis and
    offset from the origin by the given amount.
    
    Args:
        expr: The SDF expression to revolve (should be a 2D shape)
        axis: The axis of revolution ('x', 'y', or 'z') (default: 'y')
        offset: Offset of the revolution axis from the origin (default: 0.0)
        
    Returns:
        A new SDF expression representing the revolved shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Examples:
        # Create a torus by revolving a circle around the Y axis
        circle = fp.shape.circle(0.5).translate(2.0, 0.0, 0.0)
        torus = fpm.revolve(circle, axis='y')
        
        # Create a vase by revolving a profile around the Z axis
        profile = fp.shape.box(0.5, 2.0, 0.1).translate(1.0, 0.0, 0.0)
        vase = fpm.revolve(profile, axis='z')
        
    Can be used as either:
    - fpm.revolve(expr, axis, offset)
    - expr.revolve(axis, offset) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("revolve requires an SDF expression")
    
    # Validate axis
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Determine the plane perpendicular to the revolution axis
    if axis == 'x':
        # Revolve around X axis, use YZ plane
        r = sqrt(pow(y, 2) + pow(z, 2))
        # Remap to use X as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, x, 0)
    elif axis == 'y':
        # Revolve around Y axis, use XZ plane
        r = sqrt(pow(x, 2) + pow(z, 2))
        # Remap to use Y as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, y, 0)
    else:  # axis == 'z'
        # Revolve around Z axis, use XY plane
        r = sqrt(pow(x, 2) + pow(y, 2))
        # Remap to use Z as the profile axis and R as the distance from axis
        return remap_xyz(expr, r - offset, z, 0)

def attract(expr, locus, radius, exaggerate=1.0):
    """
    Attracts a shape toward a point within a given radius.
    
    This function creates a smooth attraction effect that pulls points toward
    the locus point, with the effect diminishing with distance and limited by
    the radius parameter.
    
    Args:
        expr: The SDF expression to attract
        locus: The [x, y, z] coordinates of the attraction point
        radius: The radius of influence for the attraction
        exaggerate: A multiplier to control the strength of the effect (default: 1.0)
        
    Returns:
        A new SDF expression representing the attracted shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If radius is negative or zero
        
    Examples:
        # Attract a sphere towards the point [2, 0, 0]
        sphere = fp.shape.sphere(1.0)
        attracted = fpm.attract(sphere, [2, 0, 0], 3.0)
        
    Can be used as either:
    - fpm.attract(expr, locus, radius, exaggerate)
    - expr.attract(locus, radius, exaggerate) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("attract requires an SDF expression")
        
    if radius <= 0:
        raise ValueError("Attraction radius must be positive")
    
    # Extract locus components
    if not isinstance(locus, (list, tuple)) or len(locus) != 3:
        raise ValueError("Locus must be a list or tuple of 3 coordinates [x, y, z]")
    
    lx, ly, lz = locus
    
    # Current position
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Vector from point to locus
    dx = lx - x
    dy = ly - y
    dz = lz - z
    
    # Distance to locus
    dist = length([dx, dy, dz])
    
    # Normalized direction to locus (avoiding division by zero)
    epsilon = 1e-10
    normalized_dist = max(dist, epsilon)
    nx = dx / normalized_dist
    ny = dy / normalized_dist
    nz = dz / normalized_dist
    
    # Calculate falloff based on distance
    # 1 at the locus, 0 at radius and beyond
    falloff = max(1.0 - dist / radius, 0.0)
    
    # Smooth falloff curve
    smooth_falloff = falloff**2 * (3 - 2 * falloff)
    
    # Calculate offset based on falloff and exaggeration
    offset = smooth_falloff * exaggerate * radius
    
    # Apply offset in the direction of the locus
    new_x = x + nx * offset
    new_y = y + ny * offset
    new_z = z + nz * offset
    
    return remap_xyz(expr, new_x, new_y, new_z)

def repel(expr, locus, radius, exaggerate=1.0):
    """
    Repels a shape away from a point within a given radius.
    
    This function creates a smooth repulsion effect that pushes points away from
    the locus point, with the effect diminishing with distance and limited by
    the radius parameter.
    
    Args:
        expr: The SDF expression to repel
        locus: The [x, y, z] coordinates of the repulsion point
        radius: The radius of influence for the repulsion
        exaggerate: A multiplier to control the strength of the effect (default: 1.0)
        
    Returns:
        A new SDF expression representing the repelled shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If radius is negative or zero
        
    Examples:
        # Repel a sphere away from the point [2, 0, 0]
        sphere = fp.shape.sphere(1.0)
        repelled = fpm.repel(sphere, [2, 0, 0], 3.0)
        
    Can be used as either:
    - fpm.repel(expr, locus, radius, exaggerate)
    - expr.repel(locus, radius, exaggerate) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("repel requires an SDF expression")
        
    if radius <= 0:
        raise ValueError("Repulsion radius must be positive")
    
    # Extract locus components
    if not isinstance(locus, (list, tuple)) or len(locus) != 3:
        raise ValueError("Locus must be a list or tuple of 3 coordinates [x, y, z]")
    
    lx, ly, lz = locus
    
    # Current position
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Vector from point to locus
    dx = lx - x
    dy = ly - y
    dz = lz - z
    
    # Distance to locus
    dist = length([dx, dy, dz])
    
    # Normalized direction to locus (avoiding division by zero)
    epsilon = 1e-10
    normalized_dist = max(dist, epsilon)
    nx = dx / normalized_dist
    ny = dy / normalized_dist
    nz = dz / normalized_dist
    
    # Calculate falloff based on distance
    # 1 at the locus, 0 at radius and beyond
    falloff = max(1.0 - dist / radius, 0.0)
    
    # Smooth falloff curve
    smooth_falloff = falloff**2 * (3 - 2 * falloff)
    
    # Calculate offset based on falloff and exaggeration
    offset = smooth_falloff * exaggerate * radius
    
    # Apply offset in the direction away from the locus
    new_x = x - nx * offset
    new_y = y - ny * offset
    new_z = z - nz * offset
    
    return remap_xyz(expr, new_x, new_y, new_z)

def morph(expr1, expr2, factor):
    """
    Morphs between two shapes based on a factor between 0 and 1.
    
    This function performs a weighted linear combination of the two
    input shapes, creating a smooth transition from expr1 to expr2.
    
    Args:
        expr1: The first SDF expression
        expr2: The second SDF expression
        factor: A value between 0 and 1 controlling the morph
               (0 = 100% expr1, 1 = 100% expr2)
        
    Returns:
        A new SDF expression representing the morphed shape
        
    Raises:
        TypeError: If expr1 or expr2 is not an SDF expression
        ValueError: If factor is outside the range [0, 1]
        
    Examples:
        # Morph between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        half_morph = fpm.morph(sphere, box, 0.5)  # 50% sphere, 50% box
        
    Can be used as either:
    - fpm.morph(expr1, expr2, factor)
    - expr1.morph(expr2, factor) (via extension)
    """
    if not hasattr(expr1, '_is_sdf_expr') or not hasattr(expr2, '_is_sdf_expr'):
        raise TypeError("morph requires SDF expressions")
        
    # Calculate linear interpolation between the two shapes
    # expr1 * (1 - factor) + expr2 * factor
    return expr1 * (1 - factor) + expr2 * factor

def extrude(expr, axis='z', min_val=-1.0, max_val=1.0):
    """
    Extrudes a 2D shape along a specified axis.
    
    This function takes a 2D shape and extends it between
    the specified coordinates to create a 3D shape.
    
    Args:
        expr: The 2D SDF expression to extrude
        axis: The axis along which to extrude ('x', 'y', or 'z') (default: 'z')
        min_val: The lower coordinate along the extrusion axis
        max_val: The upper coordinate along the extrusion axis
        
    Returns:
        A new SDF expression representing the extruded 3D shape
        
    Raises:
        TypeError: If expr is not an SDF expression
        ValueError: If min_val is greater than or equal to max_val
        ValueError: If axis is not 'x', 'y', or 'z'
        
    Examples:
        # Create a cylinder by extruding a circle along the Z axis
        circle = fp.shape.circle(1.0)
        cylinder = fpm.extrude(circle, axis='z', min_val=-1.0, max_val=1.0)
        
        # Create a prism by extruding a triangle along the Y axis
        triangle = fp.shape.polygon(3, 1.0)
        prism = fpm.extrude(triangle, axis='y', min_val=0.0, max_val=2.0)
        
    Can be used as either:
    - fpm.extrude(expr, axis, min_val, max_val)
    - expr.extrude(axis, min_val, max_val) (via extension)
    """
    if not hasattr(expr, '_is_sdf_expr'):
        raise TypeError("extrude requires an SDF expression")
        
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    # Validate axis
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Get coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Map axis names to actual coordinate variables
    coords = {'x': x, 'y': y, 'z': z}
    
    # Get the axis coordinate
    axis_coord = coords[axis]
    
    # Calculate the SDF by combining the 2D shape's SDF with a box in the extrusion axis
    axis_dist = max(axis_coord - max_val, min_val - axis_coord)
    
    # The final SDF is the maximum of the 2D SDF and the axis distance
    return max(expr, axis_dist)
