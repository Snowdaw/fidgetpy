"""
Domain operations for Fidget.

This module provides domain manipulation operations for SDF expressions, including:
- Domain distortion (twist, bend)
- Domain modification (elongate, onion, shell)
- Domain mirroring (mirror_x, mirror_y, mirror_z)
- Surface operations (round, displace)

All operations in this module are for direct function calls only.
"""

import fidgetpy as fp
import fidgetpy.math as fpm

def onion(sdf, thickness):
    """
    Creates concentric shells of an SDF with a given thickness.
    
    This operation creates multiple concentric shells around the original shape,
    each with the specified thickness. It's useful for creating layered or
    repeated shell structures.
    
    Args:
        sdf: The SDF expression
        thickness: The thickness of each shell (must be positive)
        
    Returns:
        An SDF expression representing concentric shells of the original SDF
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If thickness is negative
        
    Examples:
        # Create concentric shells around a sphere
        sphere = fp.shape.sphere(1.0)
        shells = fpo.onion(sphere, 0.2)  # Creates shells at distances 0.2, 0.4, 0.6, etc.
        
    IMPORTANT: Only for direct function calls: fpo.onion(sdf, thickness)
    Method calls are not supported for operations.
    """
    return fpm.abs(sdf) - thickness

def elongate(sdf, amount):
    """
    Elongates an SDF along specified axes.
    
    This function stretches the space in which the SDF lives, effectively
    elongating the shape along the desired axes. The elongation creates a
    region of constant distance along each axis.
    
    Args:
        sdf: The SDF expression to elongate
        amount: A tuple/list/vector of (x, y, z) elongation factors
                Use 0 for no elongation on that axis
                All values must be non-negative
                
    Returns:
        An SDF expression representing the elongated shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If any elongation amount is negative
        
    Examples:
        # Elongate a sphere along the x-axis
        sphere = fp.shape.sphere(1.0)
        elongated = fpo.elongate(sphere, (0.5, 0.0, 0.0))  # Stretches the sphere by 0.5 units along x
        
        # Elongate a box along multiple axes
        box = fp.shape.box(0.5, 0.5, 0.5)
        pill = fpo.elongate(box, (0.5, 0.2, 0.0))  # Creates a pill-like shape
        
    IMPORTANT: Only for direct function calls: fpo.elongate(sdf, amount)
    Method calls are not supported for operations.
    """
    # Get the x, y, z coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Create clamped coordinates that implement the elongation
    x_clamp = fpm.clamp(x, -amount[0], amount[0])
    y_clamp = fpm.clamp(y, -amount[1], amount[1])
    z_clamp = fpm.clamp(z, -amount[2], amount[2])
    
    # Create a point with the clamped coordinates removed
    p_mod = fpm.remap_xyz(sdf, x - x_clamp, y - y_clamp, z - z_clamp)
    
    # Return the elongated SDF
    return p_mod

def twist(sdf, amount):
    """
    Applies a twist deformation along the Y axis.
    
    This operation twists the space around the Y axis, with the amount of twist
    proportional to the Y coordinate. The twist is applied as a rotation in the XZ plane.
    
    Args:
        sdf: The SDF expression to twist
        amount: The amount of twist (radians per unit of Y)
        
    Returns:
        An SDF expression representing the twisted shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Twist a cylinder around the Y axis
        cylinder = fp.shape.cylinder(0.5, 2.0)
        twisted = fpo.twist(cylinder, 1.0)  # Twists by 1 radian per unit of Y
        
        # Create a twisted column
        box = fp.shape.box(0.5, 2.0, 0.5)
        twisted_column = fpo.twist(box, 0.5)  # Gentle twist for a decorative column
        
    IMPORTANT: Only for direct function calls: fpo.twist(sdf, amount)
    Method calls are not supported for operations.
    """
    # Get the x, y, z coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the twist angle based on y position
    angle = y * amount
    
    # Calculate the sine and cosine for the rotation
    c = fpm.cos(angle)
    s = fpm.sin(angle)
    
    # Create twisted coordinates
    x_twist = c * x - s * z
    z_twist = s * x + c * z
    
    # Remap the coordinates with the twist transformation
    return fpm.remap_xyz(sdf, x_twist, y, z_twist)

def bend(sdf, amount):
    """
    Applies a bend deformation along the X axis around the Y axis.
    
    This operation bends the space along the X axis, with the amount of bend
    proportional to the X coordinate. The bend is applied as a rotation in the XY plane.
    
    Args:
        sdf: The SDF expression to bend
        amount: The amount of bending (radians per unit of X)
        
    Returns:
        An SDF expression representing the bent shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Bend a box to create an arch
        box = fp.shape.box(2.0, 0.5, 0.5)
        arch = fpo.bend(box, 0.5)  # Bends the box into an arch shape
        
        # Create a curved path
        path = fp.shape.box(3.0, 0.2, 0.2)
        curved_path = fpo.bend(path, 0.3)  # Gentle curve for a path
        
    IMPORTANT: Only for direct function calls: fpo.bend(sdf, amount)
    Method calls are not supported for operations.
    """
    # Get the x, y, z coordinates
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the bend angle based on x position
    angle = x * amount
    
    # Calculate the sine and cosine for the rotation
    c = fpm.cos(angle)
    s = fpm.sin(angle)
    
    # Create bent coordinates
    x_bent = c * x - s * y
    y_bent = s * x + c * y
    
    # Remap the coordinates with the bend transformation
    return fpm.remap_xyz(sdf, x_bent, y_bent, z)

def round(sdf, radius):
    """
    Rounds the edges of an SDF by a given radius.
    
    This operation uniformly rounds all edges and corners of a shape by
    effectively growing the shape outward by the specified radius.
    
    Args:
        sdf: The SDF expression to round
        radius: The rounding radius (must be positive)
        
    Returns:
        An SDF expression representing the rounded shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If radius is negative
        
    Examples:
        # Round the edges of a box
        box = fp.shape.box(1.0, 1.0, 1.0)
        rounded_box = fpo.round(box, 0.2)  # Creates a box with 0.2 radius rounded edges
        
        # Round a complex shape
        shape = fpo.union(fp.shape.box(1.0, 0.5, 0.5), fp.shape.sphere(0.7))
        rounded = fpo.round(shape, 0.1)  # Rounds all edges of the combined shape
        
    IMPORTANT: Only for direct function calls: fpo.round(sdf, radius)
    Method calls are not supported for operations.
    """
    return sdf - radius

def shell(sdf, thickness):
    """
    Creates a shell of an SDF with a given thickness.
    
    This operation creates a hollow shell around the original shape with the
    specified thickness. Unlike onion, this creates only a single shell.
    
    Args:
        sdf: The SDF expression
        thickness: The thickness of the shell (must be positive)
        
    Returns:
        An SDF expression representing a shell of the original SDF
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If thickness is negative
        
    Examples:
        # Create a hollow sphere
        sphere = fp.shape.sphere(1.0)
        hollow_sphere = fpo.shell(sphere, 0.1)  # Creates a spherical shell of thickness 0.1
        
        # Create a hollow box
        box = fp.shape.box(1.0, 1.0, 1.0)
        hollow_box = fpo.shell(box, 0.05)  # Creates a box with 0.05 thick walls
        
    IMPORTANT: Only for direct function calls: fpo.shell(sdf, thickness)
    Method calls are not supported for operations.
    """
    return fpm.abs(sdf) - thickness

def displace(sdf, displacement_func):
    """
    Displaces the SDF by a displacement function.
    
    This operation modifies the surface of a shape by adding a displacement
    function to the distance field. Positive displacement moves the surface outward,
    while negative displacement moves it inward.
    
    Args:
        sdf: The SDF expression
        displacement_func: An SDF expression representing the displacement amount
        
    Returns:
        An SDF expression representing the displaced shape
        
    Raises:
        TypeError: If inputs are not SDF expressions
        
    Examples:
        # Create a wavy sphere using sine displacement
        sphere = fp.shape.sphere(1.0)
        x, y, z = fp.x(), fp.y(), fp.z()
        wave = 0.1 * fpm.sin(10 * x) * fpm.sin(10 * y) * fpm.sin(10 * z)
        wavy_sphere = fpo.displace(sphere, wave)  # Creates a sphere with wavy surface
        
        # Add noise to a shape
        box = fp.shape.box(1.0, 1.0, 1.0)
        noise = 0.05 * fpm.sin(20 * x) * fpm.cos(20 * y) * fpm.sin(15 * z)
        noisy_box = fpo.displace(box, noise)  # Creates a box with noisy surface
        
    IMPORTANT: Only for direct function calls: fpo.displace(sdf, displacement_func)
    Method calls are not supported for operations.
    """
    return sdf + displacement_func

def mirror_x(sdf):
    """
    Mirrors an SDF across the YZ plane (x=0).
    
    This operation creates a mirror reflection of the shape across the YZ plane,
    effectively making the shape symmetric along the X axis.
    
    Args:
        sdf: The SDF expression
        
    Returns:
        An SDF expression representing the mirrored shape
        
    Raises:
        TypeError: If input is not an SDF expression
        
    Examples:
        # Mirror a shape that's in the positive X space
        box = fp.shape.box(1.0, 1.0, 1.0).translate(2.0, 0.0, 0.0)  # Box at x=2
        mirrored = fpo.mirror_x(box)  # Creates boxes at both x=2 and x=-2
        
        # Create a symmetric shape from an asymmetric one
        shape = fp.shape.sphere(1.0).translate(0.5, 0.0, 0.0)
        symmetric = fpo.mirror_x(shape)  # Creates a symmetric shape across the YZ plane
        
    IMPORTANT: Only for direct function calls: fpo.mirror_x(sdf)
    Method calls are not supported for operations.
    """
    x = fp.x()
    return fpm.remap_xyz(sdf, fpm.abs(x), fp.y(), fp.z())

def mirror_y(sdf):
    """
    Mirrors an SDF across the XZ plane (y=0).
    
    This operation creates a mirror reflection of the shape across the XZ plane,
    effectively making the shape symmetric along the Y axis.
    
    Args:
        sdf: The SDF expression
        
    Returns:
        An SDF expression representing the mirrored shape
        
    Raises:
        TypeError: If input is not an SDF expression
        
    Examples:
        # Mirror a shape that's in the positive Y space
        box = fp.shape.box(1.0, 1.0, 1.0).translate(0.0, 2.0, 0.0)  # Box at y=2
        mirrored = fpo.mirror_y(box)  # Creates boxes at both y=2 and y=-2
        
        # Create a symmetric shape from an asymmetric one
        shape = fp.shape.sphere(1.0).translate(0.0, 0.5, 0.0)
        symmetric = fpo.mirror_y(shape)  # Creates a symmetric shape across the XZ plane
        
    IMPORTANT: Only for direct function calls: fpo.mirror_y(sdf)
    Method calls are not supported for operations.
    """
    y = fp.y()
    return fpm.remap_xyz(sdf, fp.x(), fpm.abs(y), fp.z())

def mirror_z(sdf):
    """
    Mirrors an SDF across the XY plane (z=0).
    
    This operation creates a mirror reflection of the shape across the XY plane,
    effectively making the shape symmetric along the Z axis.
    
    Args:
        sdf: The SDF expression
        
    Returns:
        An SDF expression representing the mirrored shape
        
    Raises:
        TypeError: If input is not an SDF expression
        
    Examples:
        # Mirror a shape that's in the positive Z space
        box = fp.shape.box(1.0, 1.0, 1.0).translate(0.0, 0.0, 2.0)  # Box at z=2
        mirrored = fpo.mirror_z(box)  # Creates boxes at both z=2 and z=-2
        
        # Create a symmetric shape from an asymmetric one
        shape = fp.shape.sphere(1.0).translate(0.0, 0.0, 0.5)
        symmetric = fpo.mirror_z(shape)  # Creates a symmetric shape across the XY plane
        
    IMPORTANT: Only for direct function calls: fpo.mirror_z(sdf)
    Method calls are not supported for operations.
    """
    z = fp.z()
    return fpm.remap_xyz(sdf, fp.x(), fp.y(), fpm.abs(z))