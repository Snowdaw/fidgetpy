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


def bezier_spline(points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                left_handles=None, right_handles=None,
                radius=0.1, segments=20, closed=False):
   """
   Create a Blender-like bezier spline with handles for each point.
   
   This function creates a bezier spline where each point has two handles
   (left/incoming and right/outgoing) that control how the curve bends,
   similar to Blender's bezier curves.
   
   Args:
       points: List of points (knots) defining the spline [(x1, y1, z1), (x2, y2, z2), ...]
       left_handles: List of left/incoming handles for each point. If None, handles are
                   automatically placed at 1/3 distance toward the previous point
       right_handles: List of right/outgoing handles for each point. If None, handles are
                    automatically placed at 1/3 distance toward the next point
       radius: The radius (thickness) of the spline
       segments: Number of line segments to use for approximating each curve segment
       closed: Whether to close the spline by connecting the last point to the first (default: False)
       
   Returns:
       An SDF expression representing a bezier spline
       
   Example:
       >>> import fidgetpy as fp
       >>> # Create a simple bezier spline with two points and explicit handles
       >>> points = [(-1, 0, 0), (1, 0, 0)]
       >>> left_handles = [(-1, 0, 0), (0, -1, 0)]  # Left handle of second point curves down
       >>> right_handles = [(0, 1, 0), (1, 0, 0)]   # Right handle of first point curves up
       >>> curve = fp.shape.bezier_spline(points, left_handles, right_handles, 0.1)
       >>> mesh = fp.mesh(curve)
   """
   if len(points) < 2:
       raise ValueError("Bezier spline must have at least 2 points")
   
   x, y, z = fp.x(), fp.y(), fp.z()
   
   # If handles aren't provided, generate default ones
   if left_handles is None:
       left_handles = []
       for i in range(len(points)):
           # For first point or if not closed, use point itself as left handle
           if i == 0 and not closed:
               left_handles.append(points[i])
           else:
               # Previous point index (wrap around if closed)
               prev_idx = i - 1 if i > 0 else len(points) - 1
               
               # Create handle at 2/3 from previous point to current point
               px, py, pz = points[prev_idx]
               cx, cy, cz = points[i]
               
               # Calculate 2/3 distance from previous to current
               hx = px + (cx - px) * 2/3
               hy = py + (cy - py) * 2/3
               hz = pz + (cz - pz) * 2/3
               
               left_handles.append((hx, hy, hz))
   
   if right_handles is None:
       right_handles = []
       for i in range(len(points)):
           # For last point or if not closed, use point itself as right handle
           if i == len(points) - 1 and not closed:
               right_handles.append(points[i])
           else:
               # Next point index (wrap around if closed)
               next_idx = (i + 1) % len(points)
               
               # Create handle at 1/3 from current point to next point
               cx, cy, cz = points[i]
               nx, ny, nz = points[next_idx]
               
               # Calculate 1/3 distance from current to next
               hx = cx + (nx - cx) * 1/3
               hy = cy + (ny - cy) * 1/3
               hz = cz + (nz - cz) * 1/3
               
               right_handles.append((hx, hy, hz))
   
   # Validate that all lists have the same length
   if len(points) != len(left_handles) or len(points) != len(right_handles):
       raise ValueError("Points, left_handles, and right_handles must have same length")
   
   # Initialize minimum distance to a large value
   min_dist_sq = 1000.0
   
   # Number of curve segments
   num_segments = len(points) - 1
   if closed and len(points) > 2:
       num_segments = len(points)
   
   # Process each curve segment
   for i in range(num_segments):
       # Get current point and next point (with wrapping for closed curves)
       curr_idx = i
       next_idx = (i + 1) % len(points)
       
       p0 = points[curr_idx]           # Current point
       p1 = right_handles[curr_idx]    # Right handle of current point
       p2 = left_handles[next_idx]     # Left handle of next point
       p3 = points[next_idx]           # Next point
       
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
           
           # Convert tuples to value components
           p0_x, p0_y, p0_z = p0
           p1_x, p1_y, p1_z = p1
           p2_x, p2_y, p2_z = p2
           p3_x, p3_y, p3_z = p3
           
           px = c1 * p0_x + c2 * p1_x + c3 * p2_x + c4 * p3_x
           py = c1 * p0_y + c2 * p1_y + c3 * p2_y + c4 * p3_y
           pz = c1 * p0_z + c2 * p1_z + c3 * p2_z + c4 * p3_z
           
           return px, py, pz
       
       # Start with the first point of this segment
       prev_x, prev_y, prev_z = bezier(0.0)
       
       # Iterate through subsegments
       for j in range(1, segments + 1):
           # Parameter t in [0, 1]
           t = j / segments
           
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


# --- Helper Vector Functions for FidgetPy expressions ---
# These are kept internal to this module.
def _fp_vec_sub(v1, v2):
   return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def _fp_vec_add(v1, v2):
   return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

def _fp_vec_scale(v, s):
   return (v[0] * s, v[1] * s, v[2] * s)

def _fp_vec_dot(v1, v2):
   return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def _fp_vec_length_sq(v):
   return _fp_vec_dot(v, v)

# --- Helper Bezier Curve Evaluation for FidgetPy expressions ---
def _fp_eval_bezier(p0, p1, p2, p3, t):
   # p0, p1, p2, p3 are expected to be Python tuples of numbers
   # t is a Python float
   # Returns a Python tuple of numbers
   t1 = 1.0 - t
   t1_sq = t1 * t1
   t1_cub = t1_sq * t1
   t_sq = t * t
   t_cub = t_sq * t

   term0 = _fp_vec_scale(p0, t1_cub)
   term1 = _fp_vec_scale(p1, 3.0 * t1_sq * t)
   term2 = _fp_vec_scale(p2, 3.0 * t1 * t_sq)
   term3 = _fp_vec_scale(p3, t_cub)
   
   return _fp_vec_add(_fp_vec_add(term0, term1), _fp_vec_add(term2, term3))


def cubic_bezier_spline_variable_radius(
   points,
   left_handles,
   right_handles,
   radii,
   segments=20,
   closed=False
):
   """
   Create a cubic Bezier spline with variable radius at each control point.

   The spline is composed of multiple cubic Bezier segments. Each segment is
   approximated by a series of smaller line segments. The distance to these
   line segments is calculated, and a radius is interpolated along the original
   cubic curve to define the thickness.

   Args:
       points: List of control points (knots) defining the spline [(x,y,z), ...].
       left_handles: List of left/incoming handles for each control point.
       right_handles: List of right/outgoing handles for each control point.
       radii: List of radius values for each control point.
       segments: Number of line sub-segments to approximate each cubic Bezier curve.
       closed: Whether to close the spline by connecting the last point to the first.

   Returns:
       An SDF expression representing the variable radius cubic Bezier spline.
   """
   if len(points) < 2:
       raise ValueError("Spline must have at least 2 points")
   if not (len(points) == len(left_handles) == \
           len(right_handles) == len(radii)):
       raise ValueError("All control point, handle, and radii lists must have the same length.")

   x_sample, y_sample, z_sample = fp.x(), fp.y(), fp.z()
   sample_pos = (x_sample, y_sample, z_sample)

   num_main_segments = len(points) - 1
   if closed:
       if len(points) < 2:
           return 1000.0
       num_main_segments = len(points)

   overall_sdf = 1e6

   for i in range(num_main_segments):
       p0_idx = i
       p3_idx = (i + 1) % len(points) if closed else i + 1
       
       seg_p0 = points[p0_idx]
       seg_p1 = right_handles[p0_idx]
       seg_p2 = left_handles[p3_idx]
       seg_p3 = points[p3_idx]
       
       seg_r0 = radii[p0_idx]
       seg_r3 = radii[p3_idx]

       sdf_for_this_cubic_segment = 1e10

       P_prev_sub_seg = _fp_eval_bezier(seg_p0, seg_p1, seg_p2, seg_p3, 0.0)
       t_prev_cubic = 0.0

       for k in range(1, segments + 1):
           t_curr_cubic = float(k) / segments
           P_curr_sub_seg = _fp_eval_bezier(seg_p0, seg_p1, seg_p2, seg_p3, t_curr_cubic)
           
           line_sub_seg_vec = _fp_vec_sub(P_curr_sub_seg, P_prev_sub_seg)
           sample_to_P_prev_sub = _fp_vec_sub(sample_pos, P_prev_sub_seg)
           
           dot_line_sub_seg_vec_self = _fp_vec_dot(line_sub_seg_vec, line_sub_seg_vec)
           t_proj_on_line_sub_seg = _fp_vec_dot(sample_to_P_prev_sub, line_sub_seg_vec) / \
                                    (dot_line_sub_seg_vec_self + 1e-9)
           
           t_proj_clamped = fpm.clamp(t_proj_on_line_sub_seg, 0.0, 1.0)

           closest_pt_on_small_sub_seg = _fp_vec_add(P_prev_sub_seg, _fp_vec_scale(line_sub_seg_vec, t_proj_clamped))
           
           dist_sq_to_small_sub_seg = _fp_vec_length_sq(_fp_vec_sub(sample_pos, closest_pt_on_small_sub_seg))

           t_cubic_approx_for_radius = t_prev_cubic + t_proj_clamped * (t_curr_cubic - t_prev_cubic)
           
           interpolated_radius = (1.0 - t_cubic_approx_for_radius) * seg_r0 + \
                                 t_cubic_approx_for_radius * seg_r3
           
           current_sub_segment_sdf = dist_sq_to_small_sub_seg.sqrt() - interpolated_radius
           
           sdf_for_this_cubic_segment = fpm.min(sdf_for_this_cubic_segment, current_sub_segment_sdf)

           P_prev_sub_seg = P_curr_sub_seg
           t_prev_cubic = t_curr_cubic
           
       overall_sdf = fpm.min(overall_sdf, sdf_for_this_cubic_segment)
       
   return overall_sdf
