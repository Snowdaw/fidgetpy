"""
Specialized shape primitives for Fidget.

This module provides more specialized shape primitives that don't fit into other 
categories, such as death star, ellipsoid, revolved vesica, rhombus, etc.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def ellipsoid(rx=1.0, ry=0.5, rz=0.5):
    """
    Create an ellipsoid SDF.
    
    Args:
        radii: The radii of the ellipsoid along the x, y, and z axes
        
    Returns:
        An SDF expression representing an ellipsoid
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Scale the space
    scaled_x = x / rx
    scaled_y = y / ry
    scaled_z = z / rz
    
    # Use a sphere in the scaled space
    scaled_dist = (scaled_x**2 + scaled_y**2 + scaled_z**2).sqrt() - 1.0
    
    # Approximate the true distance
    # This is a simpler approximation that works better with the mesher
    return scaled_dist * min(rx, ry, rz)

def triangular_prism(size_xy=1.0, height=2.0):
    """
    Create a triangular prism SDF.
    
    Args:
        size_xy: The size of the triangular base
        height: The height of the prism
        
    Returns:
        An SDF expression representing a triangular prism
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Take absolute values for symmetry
    q = [x.abs(), y, z.abs()]
    
    # Calculate distance to triangular prism
    # This is based on the formula from distfunctions.html
    # We're using a bound version rather than exact SDF for simplicity
    d_xy = q[0] * 0.866025 + q[1] * 0.5  # 0.866025 = sin(60°), 0.5 = cos(60°)
    d_xy_neg = -q[1]
    d_xy_max = fpm.max(d_xy, d_xy_neg) - size_xy * 0.5
    
    d_z = q[2] - height / 2.0
    
    return fpm.max(d_xy_max, d_z)

def box_frame(width=1.0, height=1.0, depth=1.0, thickness=0.1):
    """
    Create a hollow box frame SDF (like a wireframe box).
    
    This implementation uses the difference between two boxes to create
    a frame with clean edges. The outer box uses the exact Euclidean distance
    field, while the inner box is slightly smaller by twice the thickness.
    
    Args:
        width: The outer width of the box frame
        height: The outer height of the box frame
        depth: The outer depth of the box frame
        thickness: The thickness of the frame edges
        
    Returns:
        An SDF expression representing a box frame
        
    Raises:
        ValueError: If any dimension is negative or zero, or if thickness is too large
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Box frame dimensions must be positive")
    
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    
    if thickness * 2 >= min(width, height, depth):
        raise ValueError("Thickness is too large for the given dimensions")
    
    # Create the outer box
    outer_box = fp.shape.box_exact(width, height, depth)
    
    # Create the inner box (smaller by twice the thickness)
    inner_width = width - 2 * thickness
    inner_height = height - 2 * thickness
    inner_depth = depth - 2 * thickness
    inner_box = fp.shape.box_exact(inner_width, inner_height, inner_depth)
    
    # Subtract the inner box from the outer box
    return fpm.max(outer_box, -inner_box)

def link(length=1.0, width=0.5, height=0.25, thickness=0.1):
    """
    Create a chain link SDF.
    
    This function creates a signed distance field for a chain link, which is
    essentially a torus with straight sections. The link is centered at the origin
    and aligned with the y-axis.
    
    Args:
        length: The length of the straight sections (must be positive)
        width: The width of the link (outer radius in the x direction) (must be positive)
        height: The height of the link (outer radius in the z direction) (must be positive)
        thickness: The thickness of the link (tube radius) (must be positive)
        
    Returns:
        An SDF expression representing a chain link
        
    Raises:
        ValueError: If any parameter is negative or zero
        
    Examples:
        # Create a standard chain link
        link = fps.link(1.0, 0.5, 0.25, 0.1)  # Creates a chain link with specified dimensions
        
        # Create a thicker chain link
        thick_link = fps.link(1.0, 0.5, 0.25, 0.2)  # Creates a chain link with thicker tubes
        
        # Create a longer chain link
        long_link = fps.link(2.0, 0.5, 0.25, 0.1)  # Creates a longer chain link
    """
    if length <= 0 or width <= 0 or height <= 0 or thickness <= 0:
        raise ValueError("Link dimensions must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Transform y coordinate to handle the straight sections
    q = [x, fpm.max(y.abs() - length, 0.0), z]
    
    # Calculate the distance to a torus centered at the origin
    # Use width as the outer radius in x direction and height as the outer radius in z direction
    # For simplicity, we'll use the average of width and height as the outer radius
    outer_radius = (width + height) / 2.0
    
    # Calculate the distance to a torus centered at the origin
    return fpm.length([fpm.length([q[0], q[1]]) - outer_radius, q[2]]) - thickness

def cut_sphere(radius=1.0, height=0.5):
    """
    Create a cut sphere SDF.
    
    Args:
        radius: The radius of the sphere
        height: The height of the cut (from center)
        
    Returns:
        An SDF expression representing a cut sphere
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Sampling independent computations (from distfunctions.html)
    w = math.sqrt(radius*radius - height*height)
    
    # Get position in cylindrical coordinates
    q = [fpm.length([x, z]), y]
    
    # Implementation from distfunctions.html
    s = (height - radius) * q[0]**2 + w**2 * (height + radius - 2.0*q[1])
    
    # Create the SDF based on the region
    inside = (s < 0.0)
    in_cylinder = (q[0] < w)
    
    d1 = fpm.length(q) - radius
    d2 = height - q[1]
    d3 = fpm.length([q[0] - w, q[1] - height])
    
    # Return appropriate distance based on region using logical_if
    return fpm.logical_if(
        inside,
        d1,
        fpm.logical_if(in_cylinder, d2, d3)
    )

def cut_hollow_sphere(radius=1.0, height=0.5, thickness=0.1):
    """
    Create a cut hollow sphere SDF.
    
    Args:
        radius: The radius of the sphere
        height: The height of the cut (from center)
        thickness: The thickness of the sphere shell
        
    Returns:
        An SDF expression representing a cut hollow sphere
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Sampling independent computations (from distfunctions.html)
    # Use symbolic sqrt if radius or height are symbolic expressions
    if isinstance(radius, (int, float)) and isinstance(height, (int, float)):
        w = math.sqrt(radius*radius - height*height)
    else:
        # Use symbolic operations for expressions
        w = (radius*radius - height*height).sqrt()
    
    # Get position in cylindrical coordinates
    q = [fpm.length([x, z]), y]
    
    # Implementation from distfunctions.html
    condition = (height * q[0] < w * q[1])
    
    d1 = fpm.length([q[0] - w, q[1] - height])
    d2 = fpm.abs(fpm.length(q) - radius)
    
    return fpm.logical_if(condition, d1, d2) - thickness

def death_star(radius1=1.0, radius2=0.5, distance=1.2):
    """
    Create a death star SDF (a sphere with a spherical chunk taken out of it).
    
    Args:
        radius1: The radius of the main sphere
        radius2: The radius of the sphere to subtract
        distance: The distance of the center of the second sphere from the first
        
    Returns:
        An SDF expression representing a death star
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Sampling independent computations (from distfunctions.html)
    a = (radius1*radius1 - radius2*radius2 + distance*distance) / (2.0 * distance)
    b = math.sqrt(max(radius1*radius1 - a*a, 0.0))
    
    # Get position in cylindrical coordinates (x is the axis of the centers)
    q = [x, fpm.length([y, z])]
    
    # Implementation from distfunctions.html
    condition = (q[0]*b - q[1]*a > distance * fpm.max(b - q[1], 0.0))
    
    # Use component-wise operations instead of direct list operations
    q_minus_a_b = [q[0] - a, q[1] - b]
    d1 = fpm.length(q_minus_a_b)
    
    q_minus_dist = [q[0] - distance, q[1] - 0.0]
    d2 = fpm.max(fpm.length(q) - radius1, -fpm.length(q_minus_dist) + radius2)
    
    # Return distance to death star using logical_if
    return fpm.logical_if(condition, d1, d2)

def pyramid(width=1.0, height=1.0, length=None, base_z=0.0):
    """
    Create a pyramid SDF with a rectangular base centered at the origin.
    
    This implementation uses plane cutting and reflection operations to create
    a pyramid. The base is centered in the XY plane at z=base_z,
    with the apex at (0, 0, base_z+height).
    
    Args:
        width: Width of the base along the X axis (must be positive)
        height: Height of the pyramid from base to apex (must be positive)
        length: Length of the base along the Y axis (if None, equals width for a square base)
        base_z: Z coordinate of the base (default: 0.0)
        
    Returns:
        An SDF expression representing a pyramid
        
    Raises:
        ValueError: If width, length, or height is negative or zero
        
    Examples:
        # Create a square-based pyramid
        pyramid = fps.pyramid(1.0, 1.0)
        
        # Create a tall, rectangular-based pyramid
        tall_pyramid = fps.pyramid(1.0, 2.0, 1.5)
        
        # Create a pyramid with base at z=1.0
        raised_pyramid = fps.pyramid(1.0, 1.0, base_z=1.0)
    """
    # If length is not specified, make a square-based pyramid
    if length is None:
        length = width
        
    if width <= 0 or length <= 0 or height <= 0:
        raise ValueError("Pyramid dimensions must be positive")
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate half-dimensions
    dx = width / 2
    dy = length / 2
    
    # Calculate hypotenuses (for triangles from center to edges)
    hy_x = fpm.sqrt(fpm.pow(dx, 2) + fpm.pow(height, 2))
    hy_y = fpm.sqrt(fpm.pow(dy, 2) + fpm.pow(height, 2))
    
    # Calculate parameters for the x-plane
    short_x = fpm.min(dx, height)
    mid_x = fpm.max(dx, height)
    area_x = fpm.sqrt((hy_x + (mid_x + short_x)) *
                      (short_x - (hy_x - mid_x)) *
                      (short_x + (hy_x - mid_x)) *
                      (hy_x + (mid_x - short_x))) / 4
    x_offset = 2 * area_x / hy_x
    
    # Create the x-plane
    x_plane = -(x * height / hy_x - z * dx / hy_x + x_offset)
    
    # Reflect the x-plane across the YZ plane (x=0)
    x_plane_reflected = fpm.reflect_axis(x_plane, 'x')
    x_planes = fpm.max(x_plane, x_plane_reflected)
    
    # Calculate parameters for the y-plane
    short_y = fpm.min(dy, height)
    mid_y = fpm.max(dy, height)
    area_y = fpm.sqrt((hy_y + (mid_y + short_y)) *
                      (short_y - (hy_y - mid_y)) *
                      (short_y + (hy_y - mid_y)) *
                      (hy_y + (mid_y - short_y))) / 4
    y_offset = 2 * area_y / hy_y
    
    # Create the y-plane
    y_plane = -(y * height / hy_y - z * dy / hy_y + y_offset)
    
    # Reflect the y-plane across the XZ plane (y=0)
    y_plane_reflected = fpm.reflect_axis(y_plane, 'y')
    y_planes = fpm.max(y_plane, y_plane_reflected)
    
    # Combine the planes to form the pyramid
    planes = fpm.max(x_planes, y_planes)
    
    # Add the base plane (floor at z=0)
    pyramid = fpm.max(planes, -z)
    
    # Move the base to the specified base_z position
    if base_z != 0:
        pyramid = fpm.translate(pyramid, 0, 0, base_z)
    
    return pyramid

def rhombus(la=0.5, lb=0.5, h=0.5, ra=0.1):
    """
    Create a rhombus SDF based on Inigo Quilez' formula.

    Args:
        la: Length of first axis in the xz-plane (must be positive)
        lb: Length of second axis in the xz-plane (must be positive)
        h: Half-height along the y-axis (must be positive)
        ra: Rounding radius (must be non-negative)

    Returns:
        An SDF expression representing a rhombus

    Raises:
        ValueError: If la, lb, or h are negative or zero, or if ra is negative.
    """
    if la <= 0 or lb <= 0 or h <= 0:
        raise ValueError("Rhombus dimensions la, lb, and h must be positive")
    if ra < 0:
        raise ValueError("Rhombus rounding radius ra cannot be negative")

    px, py, pz = fp.x(), fp.y(), fp.z()

    # p = abs(p);
    px_abs = px.abs()
    py_abs = py.abs() # Note: GLSL uses p.y later, not abs(p.y)
    pz_abs = pz.abs()
    p_xz = [px_abs, pz_abs]

    # vec2 b = vec2(la,lb);
    b = [la, lb]

    # float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
    # ndot(a,b) = a.x*b.x - a.y*b.y in GLSL reference. Implementing this directly.
    b_minus_2p_xz = [b[0] - 2.0 * p_xz[0], b[1] - 2.0 * p_xz[1]]
    # Calculate ndot(b, b - 2.0 * p.xz)
    ndot_val = fpm.ndot(b, b_minus_2p_xz) # Use the dedicated ndot function
    dot_b_b = fpm.dot(b, b) # Denominator: dot(b, b) = la*la + lb*lb
    # Use a small epsilon to prevent potential division by zero if dot_b_b is symbolic and could be zero
    epsilon = 1e-9
    f = fpm.clamp(ndot_val / (dot_b_b + epsilon), -1.0, 1.0)

    # vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    scale_vec = [1.0 - f, 1.0 + f]
    scaled_b_half = [0.5 * b[0] * scale_vec[0], 0.5 * b[1] * scale_vec[1]]
    p_xz_minus_scaled_b_half = [p_xz[0] - scaled_b_half[0], p_xz[1] - scaled_b_half[1]]
    
    len_term = fpm.length(p_xz_minus_scaled_b_half)
    
    # Use original px_abs, pz_abs for sign calculation
    sign_term = fpm.sign(px_abs * b[1] + pz_abs * b[0] - b[0] * b[1])
    
    qx = len_term * sign_term - ra
    qy = py_abs - h # Use absolute value of y as per p = abs(p)

    q = [qx, qy]

    # return min(max(q.x,q.y),0.0) + length(max(q,0.0));
    inside_term = fpm.min(fpm.max(q[0], q[1]), 0.0)
    
    max_q_zero = [fpm.max(q[0], 0.0), fpm.max(q[1], 0.0)]
    outside_term = fpm.length(max_q_zero)

    return inside_term + outside_term