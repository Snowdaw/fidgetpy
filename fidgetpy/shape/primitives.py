"""
Basic primitive shapes for Fidget.

This module provides fundamental 3D primitive shapes for SDF expressions, including:
- Simple primitives (sphere, box, plane)
- Geometric primitives (torus, octahedron, hexagonal_prism)

These shapes serve as the building blocks for more complex shapes and can be
combined using operations from the fidgetpy.ops module.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def sphere(radius=1.0):
    """
    Create a sphere centered at the origin.
    
    This function creates an exact signed distance field for a sphere.
    The distance field is negative inside the sphere and positive outside,
    with the value representing the exact Euclidean distance to the surface.
    
    Args:
        radius: The radius of the sphere (must be positive)
        
    Returns:
        An SDF expression representing a sphere centered at the origin
        
    Raises:
        ValueError: If radius is negative or zero
        
    Examples:
        # Create a basic sphere
        sphere = fps.sphere(1.0)  # Creates a sphere with radius 1.0
        
        # Create a larger sphere
        large_sphere = fps.sphere(5.0)  # Creates a sphere with radius 5.0
        
        # Combine with other shapes
        import fidgetpy.ops as fpo
        combined = fpo.union(fps.sphere(1.0), fps.box(1.0, 1.0, 1.0))
    """
    if radius <= 0:
        raise ValueError("Sphere radius must be positive")
        
    x, y, z = fp.x(), fp.y(), fp.z()
    return (x**2 + y**2 + z**2).sqrt() - radius

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

def box(width=1.0, height=1.0, depth=1.0):
    """
    Create a box centered at the origin.
    
    This is an alias for box_exact, which creates an exact signed distance field.
    See box_exact for more details.
    
    Args:
        width: The width of the box along the x-axis (must be positive)
        height: The height of the box along the y-axis (must be positive)
        depth: The depth of the box along the z-axis (must be positive)
        
    Returns:
        An SDF expression representing a box centered at the origin
    """
    return box_exact(width, height, depth)

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

def torus(major_radius=1.0, minor_radius=0.25):
    """
    Create a torus centered at the origin and aligned with the y-axis.
    
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
        
        # Create a thin ring
        thin_ring = fps.torus(3.0, 0.1)  # Creates a thin ring with major radius 3.0
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