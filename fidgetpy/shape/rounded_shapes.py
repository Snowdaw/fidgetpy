"""
Rounded shape primitives for Fidget.

This module provides shapes with rounded features, such as rounded boxes,
rounded cylinders, and other primitives with smooth edges or corners.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def rounded_box(width=1.0, height=1.0, depth=1.0, radius=0.1):
    """
    Create a rounded box SDF.
    
    Args:
        width: The width of the box (x dimension)
        height: The height of the box (y dimension)
        depth: The depth of the box (z dimension)
        radius: The radius of the rounded corners
        
    Returns:
        An SDF expression representing a rounded box
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    half_width = width / 2.0
    half_height = height / 2.0
    half_depth = depth / 2.0
    
    dx = x.abs() - half_width + radius
    dy = y.abs() - half_height + radius
    dz = z.abs() - half_depth + radius
    
    inside = dx.max(dy).max(dz).min(0.0)
    outside = (dx.max(0.0)**2 + dy.max(0.0)**2 + dz.max(0.0)**2).sqrt()
    
    return inside + outside - radius

def rounded_cylinder(radius=1.0, height=2.0, rounding=0.1):
    """
    Create a rounded cylinder SDF.
    
    Args:
        radius: The radius of the cylinder without rounding
        height: The height of the cylinder without rounding
        rounding: The radius of the edge rounding
        
    Returns:
        An SDF expression representing a rounded cylinder
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the distance to the cylinder without rounding
    d_xz = (x**2 + z**2).sqrt() - (radius - rounding)
    d_y = y.abs() - (height / 2.0 - rounding)
    
    inside = d_xz.max(d_y).min(0.0)
    outside = (fpm.max(d_xz, 0.0)**2 + fpm.max(d_y, 0.0)**2).sqrt()
    
    return inside + outside - rounding

def round_cone(top_radius=0.1, bottom_radius=1.0, height=2.0):
    """
    Create a round cone SDF (Revised based on standard GLSL implementation).

    Args:
        top_radius: The radius at the top of the cone (r2 in GLSL)
        bottom_radius: The radius at the bottom of the cone (r1 in GLSL)
        height: The height of the cone

    Returns:
        An SDF expression representing a round cone
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    r1 = bottom_radius # GLSL r1
    r2 = top_radius    # GLSL r2
    h = height

    # Avoid division by zero or instability if height is very small
    h = fpm.max(h, 1e-6)

    # Calculate q = (length(xz), y)
    q_x = fpm.length([x, z])
    q_y = y

    # Calculate cone slope parameters
    b = (r1 - r2) / h
    # Clamp the argument to sqrt to avoid potential domain errors due to precision
    a_arg = fpm.max(0.0, 1.0 - b*b)
    a = fpm.sqrt(a_arg)

    # Calculate k = dot(q, vec2(-b, a))
    k = q_x * (-b) + q_y * a

    # Conditions using fpm.step (evaluates to 0 or 1)
    # Condition 1: k < 0.0 (Below bottom cap)
    cond1 = fpm.step(k, -1e-6) # Use small epsilon for safer comparison k < 0
    
    # Condition 2: k > a*h (Above top cap)
    # Use step(a*h, k) which is 1 if k >= a*h, 0 otherwise
    cond2 = fpm.step(a*h + 1e-6, k) # Use small epsilon for safer comparison k > a*h

    # Distance calculations
    # dist1: Below bottom cap: length(q) - r1
    dist1 = fpm.length([q_x, q_y]) - r1
    
    # dist2: Above top cap: length(q - vec2(0.0, h)) - r2
    dist2 = fpm.length([q_x, q_y - h]) - r2
    
    # dist3: Slanted side: dot(q, vec2(a, b)) - r1
    dist3 = q_x * a + q_y * b - r1

    # Select using nested logical_if (mix)
    # if cond1: return dist1
    # else if cond2: return dist2
    # else: return dist3
    # logical_if(condition, true_val, false_val) uses mix(false_val, true_val, condition)
    
    # Inner if: logical_if(cond2, dist2, dist3) -> mix(dist3, dist2, cond2)
    inner_result = fpm.logical_if(cond2, dist2, dist3)
    
    # Outer if: logical_if(cond1, dist1, inner_result) -> mix(inner_result, dist1, cond1)
    final_result = fpm.logical_if(cond1, dist1, inner_result)

    return final_result


def capped_torus(angle=0.5, major_radius=1.0, minor_radius=0.25):
    """
    Create a capped torus SDF (a segment of a torus with caps).
    
    Args:
        angle: Half-angle of the cap in radians (0.5 = quarter, PI = full torus)
        major_radius: The major radius of the torus (distance from center to center of tube)
        minor_radius: The minor radius of the torus (radius of the tube)
        
    Returns:
        An SDF expression representing a capped torus
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Take the absolute value of x to create symmetry
    p_x = x.abs()
    
    # Calculate sine and cosine of the cap angle
    sc_x = math.cos(angle)
    sc_y = math.sin(angle)
    
    # Determine whether we're in the circular region or the capped region
    condition = sc_y * p_x > sc_x * y
    
    # Use logical_if instead of if_then_else
    k = fpm.logical_if(
        condition,
        fpm.dot([p_x, y], [sc_x, sc_y]),
        fpm.length([p_x, y])
    )
    
    # Calculate distance to the capped torus
    return fpm.sqrt(fpm.dot([p_x, y, z], [p_x, y, z]) + major_radius**2 - 2.0*major_radius*k) - minor_radius

def solid_angle(angle=30.0, radius=1.0):
    """
    Create a solid angle SDF (a cone with a spherical cap).
    
    This function creates a signed distance field for a solid angle, which is
    a cone with a spherical cap. The angle parameter controls the aperture of
    the cone, and the radius parameter controls the size of the sphere.
    
    Args:
        angle: The angle of the cone in degrees (must be between 0 and 90)
        radius: The radius of the sphere from which the solid angle is cut (must be positive)
        
    Returns:
        An SDF expression representing a solid angle
        
    Raises:
        ValueError: If angle is not between 0 and 90, or if radius is negative or zero
        
    Examples:
        # Create a standard solid angle
        solid = fps.solid_angle(30.0, 1.0)  # Creates a solid angle with 30° aperture
        
        # Create a wide solid angle
        wide_solid = fps.solid_angle(60.0, 1.0)  # Creates a solid angle with 60° aperture
        
        # Create a larger solid angle
        large_solid = fps.solid_angle(45.0, 2.0)  # Creates a larger solid angle
    """
    if angle <= 0 or angle >= 90:
        raise ValueError("Solid angle aperture must be between 0 and 90 degrees")
    if radius <= 0:
        raise ValueError("Solid angle radius must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate sine and cosine of the angle
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    
    # Get position in cylindrical coordinates
    q = [fpm.length([x, z]), y]
    
    # Calculate distance to the solid angle
    l = fpm.length(q) - radius
    
    # Calculate dot product and clamp it
    dot_product = fpm.dot(q, [c, s])
    clamped_dot = fpm.clamp(dot_product, 0.0, radius)
    
    # Calculate vector for distance computation
    qx_minus_c_dot = q[0] - c * clamped_dot
    qy_minus_s_dot = q[1] - s * clamped_dot
    
    # Calculate distance m
    m = fpm.length([qx_minus_c_dot, qy_minus_s_dot])
    
    # Use logical_if instead of if_then_else for the sign factor
    sign_factor = fpm.logical_if(s*q[0] - c*q[1] < 0.0, -1.0, 1.0)

    return fpm.max(l, m * sign_factor)