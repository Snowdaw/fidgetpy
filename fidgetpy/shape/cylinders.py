"""
Cylinder-based shape primitives for Fidget.

This module provides various cylinder-based shape primitives for SDF expressions, including:
- Basic cylinders (cylinder, infinite_cylinder, capped_cylinder)
- Capsules (capsule, vertical_capsule)
- Cones (cone, capped_cone)

All shapes are centered at the origin by default and can be transformed using
the standard transformation methods.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def cylinder(radius=1.0, height=2.0):
    """
    Create a cylinder centered at the origin and aligned with the y-axis.
    
    This function creates an exact signed distance field for a cylinder.
    The cylinder is centered at the origin with its axis aligned with the y-axis.
    The distance field is negative inside the cylinder and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        radius: The radius of the cylinder (must be positive)
        height: The height of the cylinder (must be positive)
        
    Returns:
        An SDF expression representing a cylinder centered at the origin
        
    Raises:
        ValueError: If radius or height is negative or zero
        
    Examples:
        # Create a standard cylinder
        cyl = fps.cylinder(1.0, 2.0)  # Creates a cylinder with radius 1.0 and height 2.0
        
        # Create a thin, tall cylinder
        tall_cyl = fps.cylinder(0.5, 5.0)  # Creates a cylinder with radius 0.5 and height 5.0
        
        # Create a wide, short cylinder (like a disk)
        disk = fps.cylinder(3.0, 0.2)  # Creates a disk-like cylinder
    """
    # Only check concrete values, not symbolic expressions
    if isinstance(radius, (int, float)) and radius <= 0:
        raise ValueError("Cylinder radius must be positive")
    if isinstance(height, (int, float)) and height <= 0:
        raise ValueError("Cylinder height must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    d = (x**2 + z**2).sqrt() - radius
    h = y.abs() - height / 2.0
    
    outside_d = d.max(0.0)
    outside_h = h.max(0.0)
    
    inside = d.max(h).min(0.0)
    outside = (outside_d**2 + outside_h**2).sqrt()
    
    return inside + outside

def infinite_cylinder(radius=1.0, axis=(0, 1, 0)):
    """
    Create an infinite cylinder centered at the origin along a specified axis.
    
    This function creates an exact signed distance field for an infinite cylinder.
    The cylinder extends infinitely along the specified axis and has a constant radius.
    The distance field is negative inside the cylinder and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        radius: The radius of the cylinder (must be positive)
        axis: The axis of the cylinder as a vector (x, y, z) (will be normalized)
        
    Returns:
        An SDF expression representing an infinite cylinder
        
    Raises:
        ValueError: If radius is negative or zero, or if axis is (0,0,0)
        
    Examples:
        # Create a vertical infinite cylinder
        vert_cyl = fps.infinite_cylinder(1.0, (0, 1, 0))  # Along y-axis
        
        # Create a horizontal infinite cylinder
        horiz_cyl = fps.infinite_cylinder(0.5, (1, 0, 0))  # Along x-axis
        
        # Create an angled infinite cylinder
        angled_cyl = fps.infinite_cylinder(0.75, (1, 1, 1))  # Along the (1,1,1) direction
    """
    if radius <= 0:
        raise ValueError("Cylinder radius must be positive")
        
    ax, ay, az = axis
    length = math.sqrt(ax*ax + ay*ay + az*az)
    
    if length == 0:
        raise ValueError("Axis vector cannot be (0,0,0)")
        
    ax, ay, az = ax/length, ay/length, az/length
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Project point onto axis
    t = x*ax + y*ay + z*az
    px = t*ax
    py = t*ay
    pz = t*az
    
    # Calculate distance to axis
    dx = x - px
    dy = y - py
    dz = z - pz
    
    return (dx**2 + dy**2 + dz**2).sqrt() - radius

def capsule(height=2.0, radius=0.5):
    """
    Create a capsule centered at the origin and aligned with the y-axis.
    
    A capsule is a cylinder with hemispherical caps at both ends.
    This function creates an exact signed distance field for a capsule.
    The capsule is centered at the origin with its axis aligned with the y-axis.
    
    Args:
        height: The height of the capsule's cylindrical section (not including the rounded ends)
               Must be positive
        radius: The radius of the capsule (must be positive)
        
    Returns:
        An SDF expression representing a capsule centered at the origin
        
    Raises:
        ValueError: If height or radius is negative or zero
        
    Examples:
        # Create a standard capsule
        cap = fps.capsule(2.0, 0.5)  # Creates a capsule with height 2.0 and radius 0.5
        
        # Create a pill-shaped capsule
        pill = fps.capsule(1.0, 0.5)  # Creates a shorter, pill-shaped capsule
        
        # Create a long, thin capsule
        thin_cap = fps.capsule(5.0, 0.3)  # Creates a long, thin capsule
    """
    if height <= 0 or radius <= 0:
        raise ValueError("Capsule height and radius must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Project point onto line segment
    half_height = height / 2.0
    p_y = y.min(half_height).max(-half_height)
    
    # Distance from point to line segment
    d = ((x**2 + (y - p_y)**2 + z**2).sqrt() - radius)
    
    return d

def vertical_capsule(height=2.0, radius=0.5):
    """
    Create a vertical capsule with its base at the origin, extending upward.
    
    This is an optimized version of capsule() when the capsule is aligned
    with the y-axis. Unlike the regular capsule, this one is not centered
    at the origin but has its base at the origin and extends upward.
    
    Args:
        height: The height of the capsule's cylindrical section (not including the rounded ends)
               Must be positive
        radius: The radius of the capsule (must be positive)
        
    Returns:
        An SDF expression representing a vertical capsule with its base at the origin
        
    Raises:
        ValueError: If height or radius is negative or zero
        
    Examples:
        # Create a standard vertical capsule
        vcap = fps.vertical_capsule(2.0, 0.5)  # Creates a vertical capsule with height 2.0 and radius 0.5
        
        # Create a short, wide vertical capsule
        wide_vcap = fps.vertical_capsule(1.0, 1.0)  # Creates a short, wide vertical capsule
        
        # Create a tall, thin vertical capsule
        tall_vcap = fps.vertical_capsule(5.0, 0.3)  # Creates a tall, thin vertical capsule
    """
    if height <= 0 or radius <= 0:
        raise ValueError("Vertical capsule height and radius must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Create a copy of y clamped to [0, height]
    p_y = y - fpm.clamp(y, 0.0, height)
    
    # Return distance to this point minus the radius
    return fpm.length([x, p_y, z]) - radius

def cone(angle=30.0, height=0.5):
    """
    Create a cone centered at the origin, aligned with the y-axis.
    
    This function creates a signed distance field for a cone using the robust
    method described by Inigo Quilez. The cone is centered at the origin,
    with its base at y = -height/2 and its tip at y = height/2.
    
    Args:
        angle: The angle of the cone in degrees (range: 0-90)
               This is the angle between the cone's side and its axis
        height: The total height of the cone (must be positive)
        
    Returns:
        An SDF expression representing a cone centered at the origin
        
    Raises:
        ValueError: If height is negative or zero, or if angle is not strictly between 0 and 90 degrees
        
    Examples:
        # Create a standard cone
        cone = fps.cone(30.0, 2.0)  # Creates a cone with 30° angle and height 2.0
        
        # Create a narrow, tall cone
        narrow_cone = fps.cone(15.0, 3.0)  # Creates a narrow cone with 15° angle
        
        # Create a wide, short cone
        wide_cone = fps.cone(60.0, 1.0)  # Creates a wide cone with 60° angle
    """
    if height <= 0:
        raise ValueError("Cone height must be positive")
    # The reference GLSL method relies on tan(angle), which is undefined at 90 degrees,
    # and involves division by tan(angle), problematic at 0 degrees.
    if angle <= 0 or angle >= 90:
        raise ValueError("Cone angle must be strictly between 0 and 90 degrees")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # GLSL parameters
    rad_angle = math.radians(angle)
    s = math.sin(rad_angle)
    c = math.cos(rad_angle)
    
    # Point in GLSL coordinate system (tip at origin y=0, base at y=-height)
    # Map world point P=(x,y,z) [centered cone] to GLSL point p_glsl=(x, y_glsl, z)
    # y_glsl = y - tip_height = y - height / 2.0
    p_prime_y = y - height / 2.0
    p_prime_xz_len = fpm.sqrt(x**2 + z**2) # length(p_glsl.xz)
    
    # 2D vectors in (radial_distance, vertical_distance) space
    w = [p_prime_xz_len, p_prime_y] # vec2( length(p'.xz), p'.y )
    
    # q = h * vec2(tan(angle), -1.0)
    tan_angle = s / c # Safe due to angle check above
    q = [height * tan_angle, -height] # vec2
    
    # Dot products needed
    dot_wq = w[0] * q[0] + w[1] * q[1]
    dot_qq = q[0] * q[0] + q[1] * q[1]
    
    # Clamp factors
    # Add epsilon to dot_qq to prevent potential division by zero if height or angle is such that dot_qq is exactly zero (highly unlikely)
    dot_qq_safe = dot_qq + 1e-9
    clamp_a = fpm.clamp(dot_wq / dot_qq_safe, 0.0, 1.0)
    
    # Add epsilon to q[0] to prevent potential division by zero if angle is very close to 0
    q0_safe = q[0] + fpm.sign(q[0]) * 1e-9 + 1e-9 # Handle q[0] == 0 case
    clamp_b_x = fpm.clamp(w[0] / q0_safe, 0.0, 1.0)
    
    # Calculate vectors a and b
    # a = w - q * clamp_a
    a = [w[0] - q[0] * clamp_a, w[1] - q[1] * clamp_a]
    # b = w - q * vec2(clamp_b_x, 1.0)
    b = [w[0] - q[0] * clamp_b_x, w[1] - q[1] * 1.0]
    
    # Squared distances
    dot_aa = a[0] * a[0] + a[1] * a[1]
    dot_bb = b[0] * b[0] + b[1] * b[1]
    
    # Minimum squared distance
    d_sq = fpm.min(dot_aa, dot_bb)
    
    # Sign calculation
    k = -1.0 # sign(q.y) is always -1 since q.y = -height
    term1 = k * (w[0] * q[1] - w[1] * q[0])
    term2 = k * (w[1] - q[1])
    s_sign = fpm.max(term1, term2)
    
    # Final result: sqrt(d_sq) * sign(s_sign)
    # Ensure d_sq is non-negative before sqrt to handle potential floating-point inaccuracies near zero
    return fpm.sqrt(d_sq.max(0.0)) * fpm.sign(s_sign)

def capped_cone(radius1=1.0, radius2=0.5, height=2.0):
    """
    Create a capped cone centered at the origin, aligned with the y-axis.
    
    This function creates a signed distance field for a cone with circular caps
    at both ends. The cone is centered at the origin with its axis aligned with
    the y-axis. The bottom cap has radius1 and the top cap has radius2.
    
    Args:
        height: The height of the cone (must be positive)
        radius1: The radius at the bottom of the cone (must be positive)
        radius2: The radius at the top of the cone (must be positive)
        
    Returns:
        An SDF expression representing a capped cone centered at the origin
        
    Raises:
        ValueError: If height, radius1, or radius2 is negative or zero
        
    Examples:
        # Create a standard capped cone
        ccone = fps.capped_cone(2.0, 1.0, 0.5)  # Creates a cone with height 2.0, bottom radius 1.0, top radius 0.5
        
        # Create a cylinder (when both radii are equal)
        cyl = fps.capped_cone(3.0, 0.8, 0.8)  # Creates a cylinder with height 3.0 and radius 0.8
        
        # Create a truncated cone
        tcone = fps.capped_cone(1.5, 1.2, 0.7)  # Creates a truncated cone
    """
    if height <= 0 or radius1 <= 0 or radius2 <= 0:
        raise ValueError("Capped cone height and radii must be positive")

    x, y, z = fp.x(), fp.y(), fp.z()
    ra = radius1 # Use ra, rb for clarity with formula
    rb = radius2
    h = height

    # Component-wise implementation of https://www.shadertoy.com/view/tsSXzK
    # Adapted for Fidget's centered coordinate system (formula assumes base at origin)

    q_x = fpm.sqrt(x**2 + z**2) # Radial distance
    y_shifted = y + h / 2.0 # Shift y to match formula's origin assumption

    rba = rb - ra
    baba = h * h # dot(b-a, b-a) where a=(0,-h/2,0), b=(0,h/2,0) -> b-a=(0,h,0)

    # Normalized projection onto axis segment [0, 1]
    # paba = dot(p-a, b-a) / baba = (y_shifted * h) / (h*h)
    paba = y_shifted / h

    # Distance to cap regions
    # Use logical_if for conditional radius selection based on paba
    cax = fpm.max(0.0, q_x - fpm.logical_if(paba < 0.5, ra, rb))
    # cay is normalized distance along axis outside caps [-0.5, 0] if inside caps
    cay = fpm.abs(paba - 0.5) - 0.5

    # Distance to slanted cone region
    k = rba*rba + baba
    # Add epsilon to k to prevent potential division by zero if k is exactly zero
    k_safe = k + 1e-9
    # f is projection factor onto cone slant line segment [0, 1]
    f = fpm.clamp( (rba*(q_x-ra) + paba*baba) / k_safe, 0.0, 1.0 )
    cbx = q_x - ra - f*rba
    # cby is normalized distance from projection point f along axis
    cby = paba - f

    # Determine sign based on inside/outside slant (cbx) and caps (cay)
    s = fpm.logical_if((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)

    # Calculate squared distances and return final distance
    # Note: cay and cby are normalized, multiply by h for actual distance
    dist_sq_a = cax*cax + (cay*h)*(cay*h) # Cap distance squared
    dist_sq_b = cbx*cbx + (cby*h)*(cby*h) # Slant distance squared

    return s * fpm.sqrt(fpm.min(dist_sq_a, dist_sq_b))

def capped_cylinder(radius=1.0, height=2.0, cap_height=0.0):
    """
    Create a capped cylinder centered at the origin, aligned with the y-axis.
    
    This function creates an exact signed distance field for a cylinder with caps.
    The cylinder is centered at the origin with its axis aligned with the y-axis.
    If cap_height is provided, the caps will be extended by that amount.
    
    Args:
        radius: The radius of the cylinder (must be positive)
        height: The height of the cylinder (must be positive)
        cap_height: Optional additional height for the caps (default: 0.0)
        
    Returns:
        An SDF expression representing a capped cylinder centered at the origin
        
    Raises:
        ValueError: If radius or height is negative or zero, or if cap_height is negative
        
    Examples:
        # Create a standard capped cylinder
        ccyl = fps.capped_cylinder(1.0, 2.0)  # Creates a cylinder with radius 1.0 and height 2.0
        
        # Create a cylinder with extended caps
        ext_ccyl = fps.capped_cylinder(1.0, 2.0, 0.3)  # Creates a cylinder with extended caps
        
        # Create a wide, short cylinder (like a disk)
        disk = fps.capped_cylinder(3.0, 0.2)  # Creates a disk-like cylinder
    """
    # Only check concrete values, not symbolic expressions
    if isinstance(radius, (int, float)) and radius <= 0:
        raise ValueError("Capped cylinder radius must be positive")
    if isinstance(height, (int, float)) and height <= 0:
        raise ValueError("Capped cylinder height must be positive")
    if isinstance(cap_height, (int, float)) and cap_height < 0:
        raise ValueError("Cap height cannot be negative")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate distance in radial and axial directions
    rxy = fpm.length([x, z])
    dy = fpm.abs(y) - height/2.0
    
    # Calculate distance
    d = [rxy - radius, dy]
    
    # Get the inside/outside distance
    # Handle each component separately to avoid array issues
    inside = fpm.min(fpm.max(d[0], d[1]), 0.0)
    
    # For the outside distance, handle each component separately
    outside_x = fpm.max(d[0], 0.0)
    outside_y = fpm.max(d[1], 0.0)
    outside = fpm.length([outside_x, outside_y])
    
    return inside + outside