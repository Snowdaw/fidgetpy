"""
Basic primitive shapes for Fidget.

This module provides fundamental 2D and 3D primitive shapes for SDF expressions, including:
- Simple primitives (sphere, box, plane)
- 2D shapes (circle, rectangle, triangle, polygon)
- 3D geometric primitives (torus, octahedron, hexagonal_prism)
- Advanced primitives (gyroid, half_space)

These shapes serve as the building blocks for more complex shapes and can be
combined using operations from the fidgetpy.ops module.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def sphere(radius=1.0, center_x=0.0, center_y=0.0, center_z=0.0):
    """
    Create a sphere centered at the specified point.
    
    This function creates an exact signed distance field for a sphere.
    The distance field is negative inside the sphere and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        radius: The radius of the sphere (must be positive)
        center_x: The x-coordinate of the center of the sphere
        center_y: The y-coordinate of the center of the sphere
        center_z: The z-coordinate of the center of the sphere
        
    Returns:
        An SDF expression representing a sphere
        
    Raises:
        ValueError: If radius is negative or zero
        
    Examples:
        # Create a basic sphere at the origin
        s1 = fps.sphere(1.0)  # Creates a sphere with radius 1.0 at origin
        
        # Create a sphere at a specific position
        s2 = fps.sphere(2.0, 1.0, 3.0, -2.0)  # Creates a sphere with radius 2.0 centered at (1,3,-2)
    """
    if radius <= 0:
        raise ValueError("Sphere radius must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    return ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2).sqrt() - radius

def box_exact(width=1.0, height=1.0, depth=1.0):
    """
    Create a box centered at the origin with exact Euclidean distance.
    
    This function creates an exact signed distance field for a box.
    The distance field is negative inside the box and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        width: The width of the box along the x-axis (must be positive)
        height: The height of the box along the y-axis (must be positive)
        depth: The depth of the box along the z-axis (must be positive)
        
    Returns:
        An SDF expression representing a box centered at the origin
        
    Raises:
        ValueError: If any dimension is negative or zero
        
    Examples:
        # Create a cube
        cube = fps.box_exact(1.0, 1.0, 1.0)  # Creates a 1x1x1 cube
        
        # Create a rectangular box
        rect_box = fps.box_exact(2.0, 1.0, 3.0)  # Creates a 2x1x3 box
        
        # Create a flat panel
        panel = fps.box_exact(5.0, 5.0, 0.1)  # Creates a thin 5x5 panel
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Box dimensions must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    half_width = width / 2.0
    half_height = height / 2.0
    half_depth = depth / 2.0
    
    dx = x.abs() - half_width
    dy = y.abs() - half_height
    dz = z.abs() - half_depth
    
    # Using the exact same formula as libfive's box_exact_centered
    inside = fpm.min(0, fpm.max(dx, fpm.max(dy, dz)))
    outside = fpm.sqrt(fpm.pow(fpm.max(dx, 0), 2) +
                       fpm.pow(fpm.max(dy, 0), 2) +
                       fpm.pow(fpm.max(dz, 0), 2))
    
    return inside + outside

def box_mitered(width=1.0, height=1.0, depth=1.0):
    """
    Create a box centered at the origin with mitered edges.
    
    This function creates a signed distance field for a box with mitered edges.
    The distance field is negative inside the box and positive outside.
    Unlike box_exact, this version maintains sharp edges when offset.
    
    Args:
        width: The width of the box along the x-axis (must be positive)
        height: The height of the box along the y-axis (must be positive)
        depth: The depth of the box along the z-axis (must be positive)
        
    Returns:
        An SDF expression representing a box centered at the origin
        
    Raises:
        ValueError: If any dimension is negative or zero
        
    Examples:
        # Create a mitered cube
        cube = fps.box_mitered(1.0, 1.0, 1.0)
        
        # Create a mitered rectangular box
        rect_box = fps.box_mitered(2.0, 1.0, 3.0)
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Box dimensions must be positive")
    
    x, y, z = fp.x(), fp.y(), fp.z()
    half_width = width / 2.0
    half_height = height / 2.0
    half_depth = depth / 2.0
    
    # Calculate the distance to each face
    dx = x.abs() - half_width
    dy = y.abs() - half_height
    dz = z.abs() - half_depth
    
    # The mitered distance is the maximum of the distances to each face
    return fpm.max(dx, fpm.max(dy, dz))

def rectangle(width=1.0, height=1.0):
    """
    Create a 2D rectangle in the XY plane centered at the origin.
    
    Args:
        width: The width of the rectangle along the x-axis (must be positive)
        height: The height of the rectangle along the y-axis (must be positive)
        
    Returns:
        An SDF expression representing a 2D rectangle
        
    Raises:
        ValueError: If any dimension is negative or zero
    """
    if width <= 0 or height <= 0:
        raise ValueError("Rectangle dimensions must be positive")
        
    x, y = fp.x(), fp.y()
    half_width = width / 2.0
    half_height = height / 2.0
    
    dx = x.abs() - half_width
    dy = y.abs() - half_height
    
    return fpm.max(dx, dy)


def plane(normal=(0, 1, 0), offset=0):
    """
    Create an infinite plane.
    
    This function creates an exact signed distance field for an infinite plane.
    The distance field is negative on one side of the plane and positive on the other,
    with the value representing the exact distance to the plane.
    
    Args:
        normal: The normal vector of the plane as (x, y, z) (will be normalized)
        offset: The offset of the plane from the origin along the normal direction
        
    Returns:
        An SDF expression representing an infinite plane
        
    Raises:
        ValueError: If the normal vector is (0,0,0)
        
    Examples:
        # Create a horizontal plane (floor)
        floor = fps.plane((0, 1, 0), 0)  # Creates a plane at y=0 with normal (0,1,0)
        
        # Create a vertical plane (wall)
        wall = fps.plane((1, 0, 0), -5)  # Creates a plane at x=-5 with normal (1,0,0)
        
        # Create an angled plane
        angled = fps.plane((1, 1, 0), 2)  # Creates a plane with normal (1,1,0) and offset 2
    """
    nx, ny, nz = normal
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    
    if length == 0:
        raise ValueError("Normal vector cannot be (0,0,0)")
        
    nx, ny, nz = nx/length, ny/length, nz/length
    
    x, y, z = fp.x(), fp.y(), fp.z()
    return x * nx + y * ny + z * nz + offset

def bounded_plane(normal=(0, 1, 0), offset=0, bounds=(2.0, 2.0, 2.0)):
    """
    Create a bounded plane (a finite section of a plane).
    
    This function creates a bounded plane by intersecting an infinite plane
    with a box. The resulting shape has clean edges and maintains proper
    distance field properties.
    
    Args:
        normal: The normal vector of the plane (will be normalized)
        offset: The offset of the plane from the origin along the normal
        bounds: The dimensions of the bounding box (width, height, depth)
        
    Returns:
        An SDF expression representing a bounded plane
        
    Raises:
        ValueError: If normal is (0,0,0) or if any bound is negative or zero
        
    Examples:
        # Create a bounded horizontal plane
        floor_section = fps.bounded_plane((0, 1, 0), 0, (3.0, 0.1, 3.0))
        
        # Create a bounded vertical plane
        wall_section = fps.bounded_plane((1, 0, 0), 2, (0.1, 4.0, 4.0))
    """
    if isinstance(bounds, (int, float)):
        bounds = (bounds, bounds, bounds)
    
    width, height, depth = bounds
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Bounds must be positive")
    
    # Create the infinite plane
    infinite_plane = plane(normal, offset)
    
    # Create the bounding box using box_exact for clean edges
    bounding_box = box_exact(width, height, depth)
    
    # Intersect the plane with the box
    return fpm.max(infinite_plane, bounding_box)

def torus_z(major_radius=1.0, minor_radius=0.25, center_x=0.0, center_y=0.0, center_z=0.0):
    """
    Create a torus centered at the specified point, aligned with the z-axis.
    
    This function creates an exact signed distance field for a torus.
    The torus has its axis aligned with the z-axis, with its center
    at the specified point.
    
    Args:
        major_radius: The major radius of the torus (distance from center to center of tube)
                     Must be positive and greater than minor_radius
        minor_radius: The minor radius of the torus (radius of the tube)
                     Must be positive
        center_x: The x-coordinate of the center of the torus
        center_y: The y-coordinate of the center of the torus
        center_z: The z-coordinate of the center of the torus
        
    Returns:
        An SDF expression representing a torus
        
    Raises:
        ValueError: If either radius is negative or zero, or if major_radius <= minor_radius
        
    Examples:
        # Create a torus at the origin
        t1 = fps.torus_z(1.0, 0.25)  # Creates a torus with major radius 1.0 and minor radius 0.25
        
        # Create a torus at a specific position
        t2 = fps.torus_z(2.0, 0.5, 0.0, 0.0, 1.0)  # Creates a torus at position (0,0,1)
    """
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("Torus radii must be positive")
    if major_radius <= minor_radius:
        raise ValueError("Major radius must be greater than minor radius")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Adjust for center position
    x_adj = x - center_x
    y_adj = y - center_y
    z_adj = z - center_z
    
    # Calculate distance to torus
    q = ((major_radius - fpm.sqrt(x_adj**2 + y_adj**2))**2 + z_adj**2).sqrt() - minor_radius
    
    return q

def torus(major_radius=1.0, minor_radius=0.25):
    """
    Create a torus centered at the origin and aligned with the xz-plane.
    
    This function creates an exact signed distance field for a torus.
    The torus is created by sweeping a circle of radius minor_radius
    around a circle of radius major_radius in the xz-plane.
    
    Args:
        major_radius: The major radius of the torus (distance from center to center of tube)
                     Must be positive and greater than minor_radius
        minor_radius: The minor radius of the torus (radius of the tube)
                     Must be positive
        
    Returns:
        An SDF expression representing a torus centered at the origin
        
    Raises:
        ValueError: If either radius is negative or zero, or if major_radius <= minor_radius
        
    Examples:
        # Create a standard torus
        torus = fps.torus(1.0, 0.25)  # Creates a torus with major radius 1.0 and minor radius 0.25
        
        # Create a thicker torus
        thick_torus = fps.torus(2.0, 0.5)  # Creates a torus with major radius 2.0 and minor radius 0.5
    """
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("Torus radii must be positive")
    if major_radius <= minor_radius:
        raise ValueError("Major radius must be greater than minor radius")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    q_x = (x**2 + z**2).sqrt() - major_radius
    q = (q_x**2 + y**2).sqrt() - minor_radius
    
    return q

def octahedron(size=1.0):
    """
    Create an octahedron centered at the origin.
    
    This function creates a signed distance field for an octahedron.
    An octahedron is a regular polyhedron with 8 faces, 12 edges, and 6 vertices.
    It can be visualized as two square pyramids attached at their bases.
    
    Args:
        size: The size of the octahedron (distance from center to any vertex)
             Must be positive
    
    Returns:
        An SDF expression representing an octahedron centered at the origin
        
    Raises:
        ValueError: If size is negative or zero
        
    Examples:
        # Create a standard octahedron
        octa = fps.octahedron(1.0)  # Creates an octahedron with size 1.0
        
        # Create a larger octahedron
        large_octa = fps.octahedron(2.5)  # Creates an octahedron with size 2.5
    """
    if size <= 0:
        raise ValueError("Octahedron size must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Take absolute values for symmetry
    p_abs = [x.abs(), y.abs(), z.abs()]
    
    # Compute m = |x| + |y| + |z| - size
    m = p_abs[0] + p_abs[1] + p_abs[2] - size
    
    # Scale factor for the normalized distance
    k = 0.57735027  # 1/sqrt(3)
    
    # Return the distance to the octahedron
    return m * k

def hexagonal_prism(radius=1.0, height=1.0):
    """
    Create a hexagonal prism centered at the origin and aligned with the z-axis.
    
    This function creates a signed distance field for a hexagonal prism.
    The hexagonal base is in the xy-plane, and the prism extends along the z-axis.
    
    Args:
        size: A tuple (radius, height) where:
             - radius is the distance from center to outer edge of the hexagon (must be positive)
             - height is the total height of the prism (must be positive)
    
    Returns:
        An SDF expression representing a hexagonal prism centered at the origin
        
    Raises:
        ValueError: If radius or height is negative or zero
        
    Examples:
        # Create a standard hexagonal prism
        hex_prism = fps.hexagonal_prism(1.0, 2.0)  # Creates a hexagonal prism with radius 1.0 and height 2.0
        
        # Create a thin hexagonal tile
        hex_tile = fps.hexagonal_prism(2.0, 0.2)  # Creates a thin hexagonal tile
        
        # Create a tall hexagonal column
        hex_column = fps.hexagonal_prism(0.5, 5.0)  # Creates a tall, thin hexagonal column
    """
    
    if radius <= 0 or height <= 0:
        raise ValueError("Hexagonal prism radius and height must be positive")

    x, y, z = fp.x(), fp.y(), fp.z()
    half_height = height / 2.0

    # Constants for hexagon calculations (from GLSL k)
    k_x = -0.8660254  # -sqrt(3)/2
    k_y = 0.5
    k_z = 0.57735     # 1/sqrt(3)

    # Work with absolute coordinates for symmetry
    p_x = x.abs()
    p_y = y.abs()
    p_z = z.abs()

    # Hexagonal folding in XY plane (Translating: p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy)
    # Calculate dot product: dot(k.xy, p.xy)
    dot_prod = k_x * p_x + k_y * p_y
    # Calculate fold factor: 2.0 * min(dot_prod, 0.0)
    fold_factor = 2.0 * dot_prod.min(0.0)
    # Apply fold: p.xy -= fold_factor * k.xy
    p_x = p_x - fold_factor * k_x
    p_y = p_y - fold_factor * k_y

    # Calculate distance components (Translating vec2 d = ...)
    # d.x component: length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x)
    clamped_px = p_x.clamp(-k_z * radius, k_z * radius)
    dx_comp = p_x - clamped_px
    dy_comp = p_y - radius
    # Use fp.sqrt if available, otherwise assume **0.5 works for fp objects
    # Use sign() method if available on fp objects
    len_xy = (dx_comp*dx_comp + dy_comp*dy_comp).sqrt() # Or use fp.sqrt(...)
    sign_y = (p_y - radius).sign() # Or use fp.sign(...)
    d_xy = len_xy * sign_y

    # d.y component: p.z - h.y
    d_z = p_z - half_height

    # Combine distances (Translating: return min(max(d.x,d.y),0.0) + length(max(d,0.0));)
    # Use fp.sqrt if available, otherwise assume **0.5 works for fp objects
    # Use max() method if available on fp objects
    inside_dist = d_xy.max(d_z).min(0.0)
    outside_dist_vec_x = d_xy.max(0.0)
    outside_dist_vec_y = d_z.max(0.0)
    outside_dist = (outside_dist_vec_x*outside_dist_vec_x + outside_dist_vec_y*outside_dist_vec_y).sqrt() # Or use fp.sqrt(...)

    return inside_dist + outside_dist

def circle(radius=1.0, center=(0, 0)):
    """
    Create a 2D circle in the xy-plane.
    
    This function creates an exact signed distance field for a circle in the xy-plane.
    The distance field is negative inside the circle and positive outside,
    with the value representing the exact Euclidean distance to the perimeter.
    
    Args:
        radius: The radius of the circle (must be positive)
        center: The center of the circle as (x, y) coordinates
        
    Returns:
        An SDF expression representing a circle in the xy-plane
        
    Raises:
        ValueError: If radius is negative or zero
        
    Examples:
        # Create a basic circle
        c = fps.circle(1.0)  # Creates a circle with radius 1.0 at origin
        
        # Create a circle at a specific position
        c = fps.circle(2.0, (1.0, 3.0))  # Creates a circle with radius 2.0 centered at (1.0, 3.0)
    """
    if radius <= 0:
        raise ValueError("Circle radius must be positive")
        
    x, y = fp.x(), fp.y()
    cx, cy = center
    
    # Calculate the distance to the circle
    return ((x - cx)**2 + (y - cy)**2).sqrt() - radius

def ring(outer_radius=1.0, inner_radius=0.5, center_x=0.0, center_y=0.0):
    """
    Create a ring (annulus) in the xy-plane.
    
    This function creates an exact signed distance field for a ring (annulus) in the xy-plane.
    The ring is created by subtracting a smaller circle from a larger circle.
    
    Args:
        outer_radius: The outer radius of the ring (must be positive)
        inner_radius: The inner radius of the ring (must be positive and smaller than outer_radius)
        center_x: The x-coordinate of the center of the ring
        center_y: The y-coordinate of the center of the ring
        
    Returns:
        An SDF expression representing a ring in the xy-plane
        
    Raises:
        ValueError: If any radius is negative or zero, or if inner_radius >= outer_radius
        
    Examples:
        # Create a basic ring
        r = fps.ring(1.0, 0.5)  # Creates a ring with outer radius 1.0 and inner radius 0.5
        
        # Create a thin ring at a specific position
        thin_r = fps.ring(2.0, 1.8, 1.0, 1.0)  # Creates a thin ring at position (1,1)
    """
    if outer_radius <= 0 or inner_radius <= 0:
        raise ValueError("Ring radii must be positive")
    if inner_radius >= outer_radius:
        raise ValueError("Inner radius must be smaller than outer radius")
        
    # Create the outer circle
    outer_circle = circle(outer_radius, (center_x, center_y))
    
    # Create the inner circle
    inner_circle = circle(inner_radius, (center_x, center_y))
    
    # Return the difference (outer - inner)
    return fpm.max(outer_circle, -inner_circle)

def polygon(radius=1.0, sides=6, center=(0, 0)):
    """
    Create a regular polygon in the xy-plane.
    
    This function creates a signed distance field for a regular polygon with
    the specified number of sides. The polygon is centered at the origin or
    at the specified center coordinates.
    
    Args:
        radius: The radius of the polygon (distance from center to vertices) (must be positive)
        sides: The number of sides of the polygon (must be at least 3)
        center: The center of the polygon as (x, y) coordinates
        
    Returns:
        An SDF expression representing a regular polygon
        
    Raises:
        ValueError: If radius is negative or zero, or if sides < 3
        
    Examples:
        # Create a hexagon
        hex = fps.polygon(1.0, 6)  # Creates a hexagon with radius 1.0
        
        # Create a triangle
        tri = fps.polygon(1.0, 3)  # Creates a triangle with radius 1.0
    """
    if radius <= 0:
        raise ValueError("Polygon radius must be positive")
    if sides < 3:
        raise ValueError("Polygon must have at least 3 sides")
        
    x, y = fp.x(), fp.y()
    cx, cy = center
    
    # Adjust for center position
    x = x - cx
    y = y - cy
    
    # Adjust radius by cos(π/n) to get the correct radius
    adjusted_radius = radius * math.cos(math.pi / sides)
    
    # Create the first half-plane
    half = y - adjusted_radius
    
    # Rotate and intersect to create the polygon
    for i in range(1, sides):
        angle = 2 * math.pi * i / sides
        c, s = math.cos(angle), math.sin(angle)
        # Rotate the half-plane: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
        rotated_half = s * x + c * y - adjusted_radius
        half = fpm.max(half, rotated_half)
        
    return half

def half_plane(ax=0.0, ay=0.0, bx=1.0, by=0.0):
    """
    Create a half-plane defined by two points.
    
    This function creates a signed distance field for a half-plane.
    The half-plane is defined by the line passing through points (ax,ay) and (bx,by),
    with positive distances on one side and negative on the other.
    
    Args:
        ax: The x-coordinate of the first point on the line
        ay: The y-coordinate of the first point on the line
        bx: The x-coordinate of the second point on the line
        by: The y-coordinate of the second point on the line
        
    Returns:
        An SDF expression representing a half-plane
        
    Examples:
        # Create a horizontal half-plane (above the x-axis)
        h1 = fps.half_plane(0.0, 0.0, 1.0, 0.0)  # Everything above y=0 is positive
        
        # Create a vertical half-plane (right of the y-axis)
        h2 = fps.half_plane(0.0, 0.0, 0.0, 1.0)  # Everything right of x=0 is positive
    """
    x, y = fp.x(), fp.y()
    
    # Calculate the signed distance to the line
    return (by - ay) * (x - ax) - (bx - ax) * (y - ay)

def triangle(a=(-0.5, -0.5), b=(0.5, -0.5), c=(0, 0.5)):
    """
    Create a triangle defined by three points in the xy-plane.
    
    This function creates a signed distance field for a triangle.
    The triangle is defined by the three specified points.
    
    Args:
        a: The first vertex of the triangle as (x, y) coordinates
        b: The second vertex of the triangle as (x, y) coordinates
        c: The third vertex of the triangle as (x, y) coordinates
        
    Returns:
        An SDF expression representing a triangle
        
    Examples:
        # Create a default triangle
        t1 = fps.triangle()  # Creates a triangle with default vertices
        
        # Create a custom triangle
        t2 = fps.triangle((0, 0), (1, 0), (0.5, 1))  # Creates a triangle with specified vertices
    """
    # Create half-planes for each edge of the triangle
    h1 = half_plane(a, b)
    h2 = half_plane(b, c)
    h3 = half_plane(c, a)
    
    # The triangle is the intersection of the three half-planes
    # We don't know which way the triangle is wound, so we take the union
    # of both possible windings
    winding1 = fpm.max(fpm.max(h1, h2), h3)
    winding2 = fpm.max(fpm.max(-h1, -h2), -h3)
    
    return fpm.min(winding1, winding2)

def half_space(normal=(0, 1, 0), point=(0, 0, 0)):
    """
    Create an infinite half-space.
    
    This function creates an exact signed distance field for an infinite half-space.
    The half-space is defined by a plane with the specified normal vector passing
    through the specified point. The distance field is negative on one side of the
    plane and positive on the other, with the value representing the exact distance
    to the plane.
    
    Args:
        normal: The normal vector of the plane as (x, y, z) (will be normalized)
        point: A point on the plane as (x, y, z)
        
    Returns:
        An SDF expression representing a half-space
        
    Examples:
        # Create a horizontal half-space (above the xz-plane)
        h1 = fps.half_space((0, 1, 0), (0, 0, 0))  # Everything above y=0 is positive
        
        # Create a half-space with an angled normal
        h2 = fps.half_space((1, 1, 0), (1, 0, 0))  # Half-space with normal (1,1,0) through (1,0,0)
    """
    nx, ny, nz = normal
    px, py, pz = point
    
    # Calculate the squared length of the normal vector
    length_sq = nx*nx + ny*ny + nz*nz
    
    # Check if the normal vector is non-zero
    if length_sq == 0:
        raise ValueError("Normal vector cannot be (0,0,0)")
        
    # Normalize the normal vector
    length = math.sqrt(length_sq)
    nx, ny, nz = nx/length, ny/length, nz/length
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the signed distance to the plane
    return (x - px) * nx + (y - py) * ny + (z - pz) * nz

def rectangle_centered_exact(width=1.0, height=1.0, center_x=0.0, center_y=0.0):
    """
    Create a rectangle with exact distance, centered at the specified point in the xy-plane.
    
    This function creates an exact signed distance field for a rectangle.
    The distance field is negative inside the rectangle and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        width: The width of the rectangle along the x-axis (must be positive)
        height: The height of the rectangle along the y-axis (must be positive)
        center_x: The x-coordinate of the center of the rectangle
        center_y: The y-coordinate of the center of the rectangle
        
    Returns:
        An SDF expression representing a rectangle with exact distance
        
    Raises:
        ValueError: If any dimension is negative or zero
        
    Examples:
        # Create a default centered rectangle
        r1 = fps.rectangle_centered_exact()  # Creates a 1x1 rectangle at the origin
        
        # Create a custom centered rectangle
        r2 = fps.rectangle_centered_exact(2.0, 1.0, 1.0, 2.0)  # Creates a 2x1 rectangle at (1,2)
    """
    if width <= 0 or height <= 0:
        raise ValueError("Rectangle dimensions must be positive")
        
    x, y = fp.x(), fp.y()
    
    # Calculate the distance components
    dx = (x - center_x).abs() - width/2
    dy = (y - center_y).abs() - height/2
    
    # Combine the distances for the exact rectangle SDF
    inside = fpm.min(fpm.max(dx, dy), 0)
    outside = fpm.sqrt(fpm.pow(fpm.max(dx, 0), 2) + fpm.pow(fpm.max(dy, 0), 2))
    
    return inside + outside

def box_mitered_centered(width=1.0, height=1.0, depth=1.0, center_x=0.0, center_y=0.0, center_z=0.0):
    """
    Create a box with mitered edges, centered at the specified point.
    
    This function creates a signed distance field for a box with mitered edges.
    The distance field is negative inside the box and positive outside.
    Unlike box_exact, this version maintains sharp edges when offset.
    
    Args:
        width: The width of the box along the x-axis (must be positive)
        height: The height of the box along the y-axis (must be positive)
        depth: The depth of the box along the z-axis (must be positive)
        center_x: The x-coordinate of the center of the box
        center_y: The y-coordinate of the center of the box
        center_z: The z-coordinate of the center of the box
        
    Returns:
        An SDF expression representing a box with mitered edges
        
    Raises:
        ValueError: If any dimension is negative or zero
        
    Examples:
        # Create a default centered mitered box
        b1 = fps.box_mitered_centered()  # Creates a 1x1x1 box at the origin
        
        # Create a custom centered mitered box
        b2 = fps.box_mitered_centered(2.0, 1.0, 3.0, 0.0, 1.0, 0.0)  # Creates a 2x1x3 box at (0,1,0)
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Box dimensions must be positive")
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the distance to each face
    dx = (x - center_x).abs() - width/2
    dy = (y - center_y).abs() - height/2
    dz = (z - center_z).abs() - depth/2
    
    # The mitered distance is the maximum of the distances to each face
    return fpm.max(dx, fpm.max(dy, dz))

def gyroid(period=(1.0, 1.0, 1.0), thickness=0.1):
    """
    Create a gyroid surface.
    
    This function creates a signed distance field approximation of a gyroid surface.
    The gyroid is a triply periodic minimal surface with interesting properties.
    
    Args:
        period: The period of the gyroid as (x_period, y_period, z_period)
        thickness: The thickness of the gyroid shell
        
    Returns:
        An SDF expression representing a gyroid surface
        
    Examples:
        # Create a default gyroid
        g1 = fps.gyroid()  # Creates a gyroid with default parameters
        
        # Create a gyroid with custom parameters
        g2 = fps.gyroid((2.0, 2.0, 2.0), 0.2)  # Creates a gyroid with period 2 and thickness 0.2
    """
    px, py, pz = period
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Constants
    tau = 2 * math.pi
    
    # The gyroid implicit surface
    gyroid_surface = (
        fpm.sin(x * tau / px) * fpm.cos(y * tau / py) +
        fpm.sin(y * tau / py) * fpm.cos(z * tau / pz) +
        fpm.sin(z * tau / pz) * fpm.cos(x * tau / px)
    )
    
    # Create a shell around the surface
    return gyroid_surface.abs() - thickness

def cylinder_z(radius=1.0, height=2.0, base=(0, 0, 0)):
    """
    Create a cylinder along the z-axis with its base at the specified point.
    
    This function creates a signed distance field for a cylinder aligned with the z-axis.
    The cylinder is positioned with its base at the specified coordinates
    and extends along the positive z direction.
    
    Args:
        radius: The radius of the cylinder (must be positive)
        height: The height of the cylinder (must be positive)
        base: The center of the base of the cylinder as (x, y, z) coordinates
        
    Returns:
        An SDF expression representing a cylinder
        
    Raises:
        ValueError: If radius or height is negative or zero
        
    Examples:
        # Create a cylinder at the origin
        cyl = fps.cylinder_z(1.0, 2.0)  # Creates a cylinder with radius 1.0 and height 2.0
        
        # Create a cylinder at a specific position
        cyl2 = fps.cylinder_z(0.5, 3.0, (1, 2, 0))  # Creates a cylinder at position (1,2,0)
    """
    if radius <= 0 or height <= 0:
        raise ValueError("Cylinder radius and height must be positive")
        
    cx, cy, cz = base
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Create a 2D circle in the xy plane for the cylinder's cross-section
    circle_sdf = ((x - cx)**2 + (y - cy)**2).sqrt() - radius
    
    # Extend along the z-axis
    z_min = cz
    z_max = cz + height
    z_sdf = fpm.max(z_min - z, z - z_max)
    
    # Combine to create the cylinder
    return fpm.max(circle_sdf, z_sdf)

def cone_z(radius=1.0, height=2.0, base=(0, 0, 0)):
    """
    Create a cone along the z-axis with its base at the specified point.
    
    This function creates a signed distance field for a cone with its
    circular base at the specified coordinates and its apex at z = base.z + height.
    
    Args:
        radius: The radius of the base of the cone (must be positive)
        height: The height of the cone (must be positive)
        base: The center of the base of the cone as (x, y, z) coordinates
        
    Returns:
        An SDF expression representing a cone
        
    Raises:
        ValueError: If radius or height is negative or zero
        
    Examples:
        # Create a cone at the origin
        cone = fps.cone_z(1.0, 2.0)  # Creates a cone with base radius 1.0 and height 2.0
        
        # Create a cone at a specific position
        cone2 = fps.cone_z(0.5, 3.0, (1, 2, 0))  # Creates a cone at position (1,2,0)
    """
    if radius <= 0 or height <= 0:
        raise ValueError("Cone radius and height must be positive")
        
    # Calculate the angle from the cone's height and radius
    angle = math.atan2(radius, height)
    
    # Use the cone_ang_z function
    return cone_ang_z(angle, height, base)

def cone_ang_z(angle=math.pi/6, height=2.0, base=(0, 0, 0)):
    """
    Create a cone along the z-axis with the specified angle and base location.
    
    This function creates a signed distance field for a cone with a specified
    apex angle, height, and base location.
    
    Args:
        angle: The angle between the cone surface and the z-axis in radians
        height: The height of the cone (must be positive)
        base: The center of the base of the cone as (x, y, z) coordinates
        
    Returns:
        An SDF expression representing a cone
        
    Raises:
        ValueError: If height is negative or zero, or if angle is not in (0, π/2)
        
    Examples:
        # Create a cone with a 30-degree angle
        cone = fps.cone_ang_z(math.pi/6, 2.0)  # Creates a cone with a 30-degree angle
        
        # Create a cone with a 45-degree angle at a specific position
        cone2 = fps.cone_ang_z(math.pi/4, 3.0, (1, 2, 0))  # Creates a cone at position (1,2,0)
    """
    if height <= 0:
        raise ValueError("Cone height must be positive")
    if angle <= 0 or angle >= math.pi/2:
        raise ValueError("Cone angle must be between 0 and π/2 radians")
        
    cx, cy, cz = base
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Adjust for base position
    x_adj = x - cx
    y_adj = y - cy
    z_adj = z - cz
    
    # Calculate distance to cone
    return fpm.max(-z_adj,
                  fpm.cos(angle) * fpm.sqrt(x_adj**2 + y_adj**2) +
                  fpm.sin(angle) * z_adj -
                  height)

def extrude_z(shape, zmin=0.0, zmax=1.0):
    """
    Extrude a 2D shape along the z-axis.
    
    This function creates a 3D shape by extruding a 2D shape (defined in the xy-plane)
    along the z-axis between the specified z coordinate bounds.
    
    Args:
        shape: The 2D shape to extrude (SDF function in the xy-plane)
        zmin: The minimum z coordinate of the extrusion
        zmax: The maximum z coordinate of the extrusion (must be greater than zmin)
        
    Returns:
        An SDF expression representing the extruded shape
        
    Raises:
        ValueError: If zmax <= zmin
        
    Examples:
        # Extrude a circle to create a cylinder
        circ = fps.circle(1.0)
        cyl = fps.extrude_z(circ, 0.0, 2.0)  # Creates a cylinder of height 2.0
        
        # Extrude a rectangle to create a box
        rect = fps.rectangle((-0.5, -0.5), (0.5, 0.5))
        box = fps.extrude_z(rect, -0.5, 0.5)  # Creates a 1x1x1 box
    """
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin")
        
    z = fp.z()
    
    # Combine the 2D shape with the z bounds
    return fpm.max(shape, fpm.max(zmin - z, z - zmax))

def pyramid_z(a=(-0.5, -0.5), b=(0.5, 0.5), zmin=0.0, height=1.0):
    """
    Create a pyramid with a rectangular base in the xy-plane.
    
    This function creates a signed distance field for a pyramid with
    its rectangular base defined by corners a and b in the xy-plane at z=zmin,
    and its apex at z=zmin+height directly above the center of the base.
    
    Args:
        a: The first corner of the rectangular base as (x, y) coordinates
        b: The opposite corner of the rectangular base as (x, y) coordinates
        zmin: The z-coordinate of the base
        height: The height of the pyramid (must be positive)
        
    Returns:
        An SDF expression representing a pyramid
        
    Raises:
        ValueError: If height is negative or zero
        
    Examples:
        # Create a pyramid with a 1x1 base and height 1.0
        pyramid = fps.pyramid_z((-0.5, -0.5), (0.5, 0.5), 0.0, 1.0)
        
        # Create a pyramid with a rectangular base
        rect_pyramid = fps.pyramid_z((0, 0), (2, 1), 0.0, 2.0)
    """
    if height <= 0:
        raise ValueError("Pyramid height must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    
    ax, ay = a
    bx, by = b
    
    # Calculate the center of the base
    px = (ax + bx) / 2
    py = (ay + by) / 2
    
    # Calculate half-dimensions of the base
    dx = (bx - ax) / 2
    dy = (by - ay) / 2
    
    # Calculate the hypotenuse for each triangular side
    hy_x = fpm.sqrt(dx**2 + height**2)
    hy_y = fpm.sqrt(dy**2 + height**2)
    
    # Calculate parameters for the triangular sides using Heron's formula
    short_x = fpm.min(dx, height)
    mid_x = fpm.max(dx, height)
    area_x = fpm.sqrt((hy_x + (mid_x + short_x)) *
                     (short_x - (hy_x - mid_x)) *
                     (short_x + (hy_x - mid_x)) *
                     (hy_x + (mid_x - short_x))) / 4
    x_offset = 2 * area_x / hy_x
    
    # The actual xz planes
    x_plane = -(x * height / hy_x - (z - zmin) * dx / hy_x + x_offset)
    x_planes = fpm.max(x_plane, fpm.reflect_axis(x_plane, 'x'))
    
    # Calculate parameters for the yz planes
    short_y = fpm.min(dy, height)
    mid_y = fpm.max(dy, height)
    area_y = fpm.sqrt((hy_y + (mid_y + short_y)) *
                     (short_y - (hy_y - mid_y)) *
                     (short_y + (hy_y - mid_y)) *
                     (hy_y + (mid_y - short_y))) / 4
    y_offset = 2 * area_y / hy_y
    
    # The actual yz planes
    y_plane = -(y * height / hy_y - (z - zmin) * dy / hy_y + y_offset)
    y_planes = fpm.max(y_plane, fpm.reflect_axis(y_plane, 'y'))
    
    # Combine all planes to form the pyramid
    planes = fpm.max(x_planes, y_planes)
    
    # Add the base plane and adjust for position
    out = fpm.max(planes, zmin - z)
    
    # Translate to the specified position (center x and y)
    return fpm.translate(out, px, py, 0)