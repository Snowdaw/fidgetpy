"""
Blending operations for Fidget.

This module provides blending operations for SDF expressions, including:
- Basic blending (mix, lerp, blend)
- Various smooth minimum/maximum algorithms (smooth_min, smooth_max)
- Advanced blending techniques (exponential_smooth_min, power_smooth_min)
- Utility blending operations (soft_clamp, quad_bezier_blend)

All operations in this module are for direct function calls only.
"""

import fidgetpy as fp
import fidgetpy.math as fpm
import math

def blend(a, b, t):
    """
    Linear interpolation between two SDFs.
    
    This function creates a smooth transition between two shapes based on
    the interpolation factor t. When t=0, the result is identical to a,
    and when t=1, the result is identical to b.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        t: Interpolation factor (0-1, values outside this range will extrapolate)
        
    Returns:
        An SDF expression representing the linear blend between a and b
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Blend between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.blend(sphere, box, 0.3)  # 70% sphere, 30% box
        
        # Create a sequence of blended shapes
        shape1 = fp.shape.sphere(1.0)
        shape2 = fp.shape.box(1.0, 1.0, 1.0)
        blend_25 = fpo.blend(shape1, shape2, 0.25)
        blend_50 = fpo.blend(shape1, shape2, 0.50)
        blend_75 = fpo.blend(shape1, shape2, 0.75)
        
    IMPORTANT: Only for direct function calls: fpo.blend(a, b, t)
    Method calls are not supported for operations.
    """
    return fpm.mix(a, b, t)

def mix(a, b, t):
    """
    Alias for blend. Linear interpolation between two SDFs.
    
    This function is identical to blend() and is provided as an alias
    for consistency with common graphics programming terminology.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        t: Interpolation factor (0-1, values outside this range will extrapolate)
        
    Returns:
        An SDF expression representing the linear blend between a and b
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Mix between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.mix(sphere, box, 0.3)  # 70% sphere, 30% box
        
    IMPORTANT: Only for direct function calls: fpo.mix(a, b, t)
    Method calls are not supported for operations.
    """
    return blend(a, b, t)

def lerp(a, b, t):
    """
    Alias for blend. Linear interpolation between two SDFs.
    
    This function is identical to blend() and is provided as an alias
    for consistency with common graphics programming terminology (LERP = Linear intERPolation).
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        t: Interpolation factor (0-1, values outside this range will extrapolate)
        
    Returns:
        An SDF expression representing the linear blend between a and b
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Linear interpolation between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.lerp(sphere, box, 0.3)  # 70% sphere, 30% box
        
    IMPORTANT: Only for direct function calls: fpo.lerp(a, b, t)
    Method calls are not supported for operations.
    """
    return blend(a, b, t)

def smooth_min(a, b, k):
    """
    Smoothly blends the minimum of two SDF values.
    
    Uses the polynomial smooth minimum algorithm to create a smooth blend
    between two shapes. This is similar to a union operation but with a
    smooth transition between the shapes.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp min, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth minimum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is negative
        
    Examples:
        # Smooth blend between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.smooth_min(sphere, box, 0.2)  # Creates a smooth blend in the union
        
        # Create a blob-like shape from multiple spheres
        sphere1 = fp.shape.sphere(1.0).translate(-0.5, 0.0, 0.0)
        sphere2 = fp.shape.sphere(0.8).translate(0.5, 0.0, 0.0)
        blob = fpo.smooth_min(sphere1, sphere2, 0.5)
        
    IMPORTANT: Only for direct function calls: fpo.smooth_min(a, b, k)
    Method calls are not supported for operations.
    """
    h = fpm.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return fpm.mix(b, a, h) - k * h * (1.0 - h)

def smooth_max(a, b, k):
    """
    Smoothly blends the maximum of two SDF values.
    
    Uses the polynomial smooth maximum algorithm to create a smooth blend
    between two shapes. This is similar to an intersection operation but with a
    smooth transition between the shapes.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp max, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth maximum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is negative
        
    Examples:
        # Smooth intersection between a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.smooth_max(sphere, box, 0.2)  # Creates a smooth blend in the intersection
        
        # Create a smooth intersection of two cylinders
        cyl1 = fp.shape.cylinder(0.5, 2.0).rotate_x(90)
        cyl2 = fp.shape.cylinder(0.5, 2.0).rotate_z(90)
        smooth_cross = fpo.smooth_max(cyl1, cyl2, 0.1)
        
    IMPORTANT: Only for direct function calls: fpo.smooth_max(a, b, k)
    Method calls are not supported for operations.
    """
    h = fpm.clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return fpm.mix(b, a, h) + k * h * (1.0 - h)

def exponential_smooth_min(a, b, k):
    """
    Smooth minimum using the exponential smoothing algorithm.
    
    This function provides an alternative smooth minimum implementation using
    exponential functions. It often produces a more natural-looking blend
    than the polynomial version, especially for organic shapes.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp min, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth minimum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is too small (approaches zero)
        
    Examples:
        # Exponential smooth blend between shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.exponential_smooth_min(sphere, box, 0.2)
        
        # Create organic-looking connections between shapes
        shape1 = fp.shape.sphere(0.8).translate(-1.0, 0.0, 0.0)
        shape2 = fp.shape.sphere(0.6).translate(1.0, 0.0, 0.0)
        organic = fpo.exponential_smooth_min(shape1, shape2, 0.4)
        
    IMPORTANT: Only for direct function calls: fpo.exponential_smooth_min(a, b, k)
    Method calls are not supported for operations.
    """
    # Prevent division by zero by using a small positive value for k
    # when k approaches zero
    k = fpm.max(k, 1e-6)
            
    # Exponential smooth min formula from Inigo Quilez
    # https://iquilezles.org/articles/smin/
    a_exp = fpm.exp(-a / k)
    b_exp = fpm.exp(-b / k)
    return -k * fpm.ln(a_exp + b_exp)

def exponential_smooth_max(a, b, k):
    """
    Smooth maximum using the exponential smoothing algorithm.
    
    This function provides an alternative smooth maximum implementation using
    exponential functions. It often produces a more natural-looking blend
    than the polynomial version, especially for organic shapes.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp max, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth maximum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is too small (approaches zero)
        
    Examples:
        # Exponential smooth intersection between shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.exponential_smooth_max(sphere, box, 0.2)
        
        # Create organic-looking intersections
        shape1 = fp.shape.sphere(1.2)
        shape2 = fp.shape.box(1.0, 1.0, 1.0)
        organic_intersection = fpo.exponential_smooth_max(shape1, shape2, 0.3)
        
    IMPORTANT: Only for direct function calls: fpo.exponential_smooth_max(a, b, k)
    Method calls are not supported for operations.
    """
    # Smooth max is just a negated smooth min of negated inputs
    return -exponential_smooth_min(-a, -b, k)

def power_smooth_min(a, b, k):
    """
    Smooth minimum using the power smoothing algorithm.
    
    This function provides yet another smooth minimum implementation using
    power functions. It offers different blending characteristics than the
    polynomial or exponential versions, with parameter k controlling the
    sharpness of the transition.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing exponent (higher = sharper transition)
           Must be positive and typically > 1
        
    Returns:
        An SDF expression representing the smooth minimum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is not positive
        
    Examples:
        # Power-based smooth blend between shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.power_smooth_min(sphere, box, 8.0)  # Higher k = sharper transition
        
        # Compare different k values
        blend_k2 = fpo.power_smooth_min(sphere, box, 2.0)  # Very smooth
        blend_k8 = fpo.power_smooth_min(sphere, box, 8.0)  # Medium smoothness
        blend_k32 = fpo.power_smooth_min(sphere, box, 32.0)  # Almost sharp
        
    IMPORTANT: Only for direct function calls: fpo.power_smooth_min(a, b, k)
    Method calls are not supported for operations.
    """
    # Power smooth min formula from Inigo Quilez
    a_pow = fpm.pow(a, k)
    b_pow = fpm.pow(b, k)
    return fpm.pow((a_pow * b_pow) / (a_pow + b_pow), 1.0 / k)

def power_smooth_max(a, b, k):
    """
    Smooth maximum using the power smoothing algorithm.
    
    This function provides a power-based smooth maximum implementation.
    It offers different blending characteristics than the polynomial or
    exponential versions, with parameter k controlling the sharpness of
    the transition.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing exponent (higher = sharper transition)
           Must be positive and typically > 1
        
    Returns:
        An SDF expression representing the smooth maximum
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is not positive
        
    Examples:
        # Power-based smooth intersection between shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.power_smooth_max(sphere, box, 8.0)  # Higher k = sharper transition
        
        # Compare different k values for intersection
        intersect_k2 = fpo.power_smooth_max(sphere, box, 2.0)  # Very smooth
        intersect_k8 = fpo.power_smooth_max(sphere, box, 8.0)  # Medium smoothness
        intersect_k32 = fpo.power_smooth_max(sphere, box, 32.0)  # Almost sharp
        
    IMPORTANT: Only for direct function calls: fpo.power_smooth_max(a, b, k)
    Method calls are not supported for operations.
    """
    # Smooth max is just a negated smooth min of negated inputs
    return -power_smooth_min(-a, -b, k)

def soft_clamp(x, min_val, max_val, k):
    """
    Softly clamps a value between min and max with smoothing factor k.
    
    This function restricts a value to a specified range, but with smooth
    transitions at the boundaries. Unlike a hard clamp, which creates a
    sharp cutoff, this produces a gradual approach to the limits.
    
    Args:
        x: The value to clamp (SDF expression or numeric)
        min_val: The minimum value
        max_val: The maximum value (should be > min_val)
        k: Smoothing factor (0 = hard clamp, larger = smoother transition)
           Must be positive for meaningful results
    
    Returns:
        An SDF expression representing the soft-clamped value
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If min_val > max_val or k is negative
        
    Examples:
        # Softly clamp a distance field
        sdf = fp.shape.sphere(1.0)
        clamped = fpo.soft_clamp(sdf, -0.5, 0.5, 0.1)  # Clamps values between -0.5 and 0.5
        
        # Create a soft boundary effect
        plane = fp.shape.plane((0, 1, 0), 0)  # Ground plane
        soft_boundary = fpo.soft_clamp(plane, -0.2, 0.0, 0.1)  # Soft transition at ground
        
    IMPORTANT: Only for direct function calls: fpo.soft_clamp(x, min_val, max_val, k)
    Method calls are not supported for operations.
    """
    return smooth_max(min_val, smooth_min(x, max_val, k), k)

def quad_bezier_blend(a, b, c, t):
    """
    Blend using quadratic Bezier curve with control point c.
    
    This function creates a non-linear blend between two values using a
    quadratic Bezier curve. Unlike linear interpolation, this allows for
    curved transitions controlled by a third point.
    
    Args:
        a: Start value (SDF expression or numeric)
        b: End value (SDF expression or numeric)
        c: Control point value (SDF expression or numeric)
        t: Parameter (0-1, values outside this range will extrapolate)
    
    Returns:
        An SDF expression representing the Bezier-interpolated value
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Bezier blend between a sphere and a box with control point
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        control = fp.shape.torus(1.0, 0.3)
        result = fpo.quad_bezier_blend(sphere, box, control, 0.5)
        
        # Create a sequence of Bezier-blended shapes
        shape1 = fp.shape.sphere(1.0)
        shape2 = fp.shape.box(1.0, 1.0, 1.0)
        control = fp.shape.cylinder(0.5, 1.0)
        blend_25 = fpo.quad_bezier_blend(shape1, shape2, control, 0.25)
        blend_50 = fpo.quad_bezier_blend(shape1, shape2, control, 0.50)
        blend_75 = fpo.quad_bezier_blend(shape1, shape2, control, 0.75)
        
    IMPORTANT: Only for direct function calls: fpo.quad_bezier_blend(a, b, c, t)
    Method calls are not supported for operations.
    """
    t_sq = t * t
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    
    return a * one_minus_t_sq + 2.0 * c * one_minus_t * t + b * t_sq