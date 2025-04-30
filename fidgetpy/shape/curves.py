"""
Curve-based shape primitives for Fidget.

This module provides shape primitives based on curves, such as line segments,
bezier curves, and polylines.
"""

import math
import fidgetpy as fp
import fidgetpy.math as fpm

def line_segment(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0), radius=0.1):
    """
    Create a line segment SDF with rounded caps.
    
    Args:
        start: The starting point of the line segment (x, y, z)
        end: The ending point of the line segment (x, y, z)
        radius: The radius (thickness) of the line segment
        
    Returns:
        An SDF expression representing a line segment
    
    Example:
        >>> import fidgetpy as fp
        >>> line = fp.shape.line_segment((-1, 0, 0), (1, 0, 0), 0.1)
        >>> mesh = fp.mesh(line)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Convert tuples to values
    start_x, start_y, start_z = start
    end_x, end_y, end_z = end
    
    # Vector from start to end
    seg_x = end_x - start_x
    seg_y = end_y - start_y
    seg_z = end_z - start_z
    
    # Vector from start to query point
    to_point_x = x - start_x
    to_point_y = y - start_y
    to_point_z = z - start_z
    
    # Project query point onto segment
    seg_len_sq = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z
    
    # Avoid division by zero for degenerate segments
    # Use max function instead of method
    seg_len_sq = max(seg_len_sq, 1e-8)
    
    # Calculate projection parameter (t along segment)
    t_proj = ((to_point_x*seg_x + to_point_y*seg_y + to_point_z*seg_z) / seg_len_sq).clamp(0.0, 1.0)
    
    # Calculate closest point on segment
    closest_x = start_x + t_proj * seg_x
    closest_y = start_y + t_proj * seg_y
    closest_z = start_z + t_proj * seg_z
    
    # Calculate squared distance to closest point
    dx = x - closest_x
    dy = y - closest_y
    dz = z - closest_z
    dist_sq = dx*dx + dy*dy + dz*dz
    
    # Return the SDF (distance to line minus radius)
    return dist_sq.sqrt() - radius

def quadratic_bezier(p0=(0.0, 0.0, 0.0), p1=(0.5, 1.0, 0.0), p2=(1.0, 0.0, 0.0), radius=0.1, segments=30):
    """
    Create a quadratic bezier curve SDF.
    
    The curve is defined by three control points and approximated using line segments.
    
    Args:
        p0: First control point (x, y, z)
        p1: Second control point (handle) (x, y, z)
        p2: Third control point (x, y, z)
        radius: The radius (thickness) of the curve
        segments: Number of line segments to use for approximation (default: 30)
        
    Returns:
        An SDF expression representing a quadratic bezier curve
    
    Example:
        >>> import fidgetpy as fp
        >>> curve = fp.shape.quadratic_bezier((-1, 0, 0), (0, 1, 0), (1, 0, 0), 0.1)
        >>> mesh = fp.mesh(curve)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Convert tuples to values
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    p2_x, p2_y, p2_z = p2
    
    # Helper function to evaluate bezier curve at parameter t
    def bezier(t):
        t1 = 1.0 - t
        t1_sq = t1 * t1
        t_t1_2 = 2.0 * t1 * t
        t_sq = t * t
        
        px = t1_sq * p0_x + t_t1_2 * p1_x + t_sq * p2_x
        py = t1_sq * p0_y + t_t1_2 * p1_y + t_sq * p2_y
        pz = t1_sq * p0_z + t_t1_2 * p1_z + t_sq * p2_z
        
        return px, py, pz
    
    # We'll use a line segment approximation approach
    # Initialize minimum distance to a large value
    min_dist_sq = 1000.0
    
    # Start with the first point
    prev_x, prev_y, prev_z = bezier(0.0)
    
    # Iterate through segments
    for i in range(1, segments + 1):
        # Parameter t in [0, 1]
        t = i / segments
        
        # Get current point on curve
        curr_x, curr_y, curr_z = bezier(t)
        
        # Calculate distance to line segment
        # Vector from previous point to current point (segment direction)
        seg_x = curr_x - prev_x
        seg_y = curr_y - prev_y
        seg_z = curr_z - prev_z
        
        # Vector from previous point to query point
        to_point_x = x - prev_x
        to_point_y = y - prev_y
        to_point_z = z - prev_z
        
        # Project query point onto segment
        seg_len_sq = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z
        
        # Avoid division by zero for degenerate segments
        # Use max function instead of method
        seg_len_sq = max(seg_len_sq, 1e-8)
        
        # Calculate projection parameter (t along segment)
        t_proj = fpm.clamp((to_point_x*seg_x + to_point_y*seg_y + to_point_z*seg_z) / seg_len_sq, 0.0, 1.0)
        
        # Calculate closest point on segment
        closest_x = prev_x + t_proj * seg_x
        closest_y = prev_y + t_proj * seg_y
        closest_z = prev_z + t_proj * seg_z
        
        # Calculate squared distance to closest point
        dx = x - closest_x
        dy = y - closest_y
        dz = z - closest_z
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Update minimum distance
        min_dist_sq = fpm.min(min_dist_sq, dist_sq)
        
        # Current point becomes previous point for next iteration
        prev_x, prev_y, prev_z = curr_x, curr_y, curr_z
    
    # Return the SDF (distance to curve minus radius)
    return min_dist_sq.sqrt() - radius

def cubic_bezier(p0=(0.0, 0.0, 0.0), p1=(0.0, 1.0, 0.0), p2=(1.0, 1.0, 0.0), p3=(1.0, 0.0, 0.0), radius=0.1, segments=40):
    """
    Create a cubic bezier curve SDF.
    
    The curve is defined by four control points and approximated using line segments.
    
    Args:
        p0: First control point (x, y, z)
        p1: Second control point (handle) (x, y, z)
        p2: Third control point (handle) (x, y, z)
        p3: Fourth control point (x, y, z)
        radius: The radius (thickness) of the curve
        segments: Number of line segments to use for approximation (default: 40)
        
    Returns:
        An SDF expression representing a cubic bezier curve
    
    Example:
        >>> import fidgetpy as fp
        >>> curve = fp.shape.cubic_bezier((-1, 0, 0), (-0.5, 1, 0), (0.5, -1, 0), (1, 0, 0), 0.1)
        >>> mesh = fp.mesh(curve)
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Convert tuples to values
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    p2_x, p2_y, p2_z = p2
    p3_x, p3_y, p3_z = p3
    
    # Helper function to evaluate cubic bezier curve at parameter t
    def bezier(t):
        t1 = 1.0 - t
        t1_sq = t1 * t1
        t1_cu = t1_sq * t1
        
        t_sq = t * t
        t_cu = t_sq * t
        
        # Cubic Bezier formula: (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        c1 = t1_cu
        c2 = 3.0 * t1_sq * t
        c3 = 3.0 * t1 * t_sq
        c4 = t_cu
        
        px = c1 * p0_x + c2 * p1_x + c3 * p2_x + c4 * p3_x
        py = c1 * p0_y + c2 * p1_y + c3 * p2_y + c4 * p3_y
        pz = c1 * p0_z + c2 * p1_z + c3 * p2_z + c4 * p3_z
        
        return px, py, pz
    
    # We'll use a line segment approximation approach
    # Initialize minimum distance to a large value
    min_dist_sq = 1000.0
    
    # Start with the first point
    prev_x, prev_y, prev_z = bezier(0.0)
    
    # Iterate through segments
    for i in range(1, segments + 1):
        # Parameter t in [0, 1]
        t = i / segments
        
        # Get current point on curve
        curr_x, curr_y, curr_z = bezier(t)
        
        # Calculate distance to line segment
        # Vector from previous point to current point (segment direction)
        seg_x = curr_x - prev_x
        seg_y = curr_y - prev_y
        seg_z = curr_z - prev_z
        
        # Vector from previous point to query point
        to_point_x = x - prev_x
        to_point_y = y - prev_y
        to_point_z = z - prev_z
        
        # Project query point onto segment
        seg_len_sq = seg_x*seg_x + seg_y*seg_y + seg_z*seg_z
        
        # Avoid division by zero for degenerate segments
        # Use max function instead of method
        seg_len_sq = max(seg_len_sq, 1e-8)
        
        # Calculate projection parameter (t along segment)
        t_proj = ((to_point_x*seg_x + to_point_y*seg_y + to_point_z*seg_z) / seg_len_sq).clamp(0.0, 1.0)
        
        # Calculate closest point on segment
        closest_x = prev_x + t_proj * seg_x
        closest_y = prev_y + t_proj * seg_y
        closest_z = prev_z + t_proj * seg_z
        
        # Calculate squared distance to closest point
        dx = x - closest_x
        dy = y - closest_y
        dz = z - closest_z
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Update minimum distance
        min_dist_sq = fpm.min(min_dist_sq, dist_sq)
        
        # Current point becomes previous point for next iteration
        prev_x, prev_y, prev_z = curr_x, curr_y, curr_z
    
    # Return the SDF (distance to curve minus radius)
    return min_dist_sq.sqrt() - radius

def polyline(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], radius=0.1, closed=False):
    """
    Create a polyline SDF (connected line segments).
    
    Args:
        points: List of points defining the polyline [(x1, y1, z1), (x2, y2, z2), ...]
        radius: The radius (thickness) of the polyline
        closed: Whether to close the polyline by connecting the last point to the first (default: False)
        
    Returns:
        An SDF expression representing a polyline
    
    Example:
        >>> import fidgetpy as fp
        >>> points = [(-1, 0, 0), (0, 1, 0), (1, 0, 0)]
        >>> polyline = fp.shape.polyline(points, 0.1)
        >>> mesh = fp.mesh(polyline)
    """
    if len(points) < 2:
        raise ValueError("Polyline must have at least 2 points")
    
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Initialize minimum distance to a large value
    min_dist_sq = 1000.0
    
    # Process each segment
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        # Calculate distance to this segment using line_segment function
        segment_sdf = line_segment(start, end, 0.0)
        segment_dist_sq = segment_sdf * segment_sdf
        
        # Update minimum distance
        min_dist_sq = fpm.min(min_dist_sq, segment_dist_sq)
    
    # If closed, add the segment connecting the last point to the first
    if closed and len(points) > 2:
        start = points[-1]
        end = points[0]
        
        segment_sdf = line_segment(start, end, 0.0)
        segment_dist_sq = segment_sdf * segment_sdf
        
        min_dist_sq = fpm.min(min_dist_sq, segment_dist_sq)
    
    # Return the SDF (square root of minimum distance squared minus radius)
    return min_dist_sq.sqrt() - radius