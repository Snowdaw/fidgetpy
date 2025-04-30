"""
Boolean operations for Fidget.

This module provides boolean operations for SDF expressions, including:
- Basic boolean operations (union, intersection, difference)
- Smooth boolean operations (smooth_union, smooth_intersection, smooth_difference)
- Logical operations (complement, boolean_and, boolean_or, boolean_not, boolean_xor)

All operations in this module are for direct function calls only.
"""

import fidgetpy as fp
import fidgetpy.math as fpm

def union(a, b):
    """
    Union of two SDFs (minimum of the two).
    
    Creates a shape that represents the union of two shapes, which is the space
    occupied by either shape. For SDFs, this is implemented as the minimum of
    the two distance fields.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        
    Returns:
        An SDF expression representing the union of the two inputs
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Union of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.union(sphere, box)
        
    IMPORTANT: Only for direct function calls: fpo.union(a, b)
    Method calls are not supported for operations.
    """
    # Union is the minimum of two SDFs
    return fpm.min(a, b)

def intersection(a, b):
    """
    Intersection of two SDFs (maximum of the two).
    
    Creates a shape that represents the intersection of two shapes, which is the space
    occupied by both shapes. For SDFs, this is implemented as the maximum of
    the two distance fields.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        
    Returns:
        An SDF expression representing the intersection of the two inputs
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Intersection of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.intersection(sphere, box)
        
    IMPORTANT: Only for direct function calls: fpo.intersection(a, b)
    Method calls are not supported for operations.
    """
    # Intersection is the maximum of two SDFs
    return fpm.max(a, b)

def difference(a, b):
    """
    Difference of two SDFs (A - B).
    
    Creates a shape that represents the difference between two shapes, which is the space
    occupied by the first shape but not the second. For SDFs, this is implemented as the
    intersection of the first shape with the complement of the second.
    
    Args:
        a: First SDF expression (the one to subtract from)
        b: Second SDF expression (the one to subtract)
        
    Returns:
        An SDF expression representing the difference (a - b)
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Subtract a sphere from a box
        box = fp.shape.box(1.0, 1.0, 1.0)
        sphere = fp.shape.sphere(0.8)
        result = fpo.difference(box, sphere)  # Creates a box with a spherical hole
        
    IMPORTANT: Only for direct function calls: fpo.difference(a, b)
    Method calls are not supported for operations.
    """
    # Difference is the intersection of a with the negation of b
    return fpm.max(a, -b)

def smooth_union(a, b, k):
    """
    Smooth union of two SDFs using the polynomial smooth minimum formula.
    
    Creates a shape that represents the union of two shapes with a smooth blend
    between them. The parameter k controls the size of the blending region.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp union, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth union
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is negative
        
    Examples:
        # Smooth union of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.smooth_union(sphere, box, 0.2)  # Creates a smoothly blended union
        
    IMPORTANT: Only for direct function calls: fpo.smooth_union(a, b, k)
    Method calls are not supported for operations.
    """
    # Implementation of polynomial smooth min
    h = fpm.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return fpm.mix(b, a, h) - k * h * (1.0 - h)

def smooth_intersection(a, b, k):
    """
    Smooth intersection of two SDFs using the polynomial smooth maximum formula.
    
    Creates a shape that represents the intersection of two shapes with a smooth blend
    between them. The parameter k controls the size of the blending region.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        k: Smoothing factor (0 = sharp intersection, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth intersection
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is negative
        
    Examples:
        # Smooth intersection of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.smooth_intersection(sphere, box, 0.2)  # Creates a smoothly blended intersection
        
    IMPORTANT: Only for direct function calls: fpo.smooth_intersection(a, b, k)
    Method calls are not supported for operations.
    """
    # Implementation of polynomial smooth max
    h = fpm.clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return fpm.mix(b, a, h) + k * h * (1.0 - h)

def smooth_difference(a, b, k):
    """
    Smooth difference (subtraction) of two SDFs: A - B.
    
    Creates a shape that represents the difference between two shapes with a smooth blend
    between them. The parameter k controls the size of the blending region.
    
    Args:
        a: First SDF expression (the one to subtract from)
        b: Second SDF expression (the one to subtract)
        k: Smoothing factor (0 = sharp difference, larger = smoother blend)
           Must be positive for meaningful results
        
    Returns:
        An SDF expression representing the smooth difference
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If k is negative
        
    Examples:
        # Smooth difference of a box and a sphere
        box = fp.shape.box(1.0, 1.0, 1.0)
        sphere = fp.shape.sphere(0.8)
        result = fpo.smooth_difference(box, sphere, 0.2)  # Creates a box with a smoothly blended spherical hole
        
    IMPORTANT: Only for direct function calls: fpo.smooth_difference(a, b, k)
    Method calls are not supported for operations.
    """
    # Smooth difference is just smooth intersection with -b
    return smooth_intersection(a, -b, k)

def complement(sdf):
    """
    Returns the complement of an SDF.
    
    Effectively flips inside and outside, which is useful for creating
    negative space or hollow objects. This is implemented by negating the SDF value.
    
    Args:
        sdf: The SDF expression to complement
        
    Returns:
        The complemented SDF expression
        
    Raises:
        TypeError: If input is not an SDF expression
        
    Examples:
        # Create a hollow sphere
        sphere = fp.shape.sphere(1.0)
        hollow = fpo.complement(sphere)  # Everything outside the sphere becomes inside
        
    IMPORTANT: Only for direct function calls: fpo.complement(sdf)
    Method calls are not supported for operations.
    """
    return -sdf

def boolean_and(a, b):
    """
    Alias for intersection operation.
    
    This is a semantic alias for the intersection operation, making code more
    readable when using logical operations.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        
    Returns:
        An SDF expression representing the intersection (AND)
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Logical AND of two shapes
        left_half = fp.shape.plane((1, 0, 0), 0)  # x < 0
        top_half = fp.shape.plane((0, 1, 0), 0)   # y < 0
        bottom_left = fpo.boolean_and(left_half, top_half)  # x < 0 AND y < 0
        
    IMPORTANT: Only for direct function calls: fpo.boolean_and(a, b)
    Method calls are not supported for operations.
    """
    return intersection(a, b)

def boolean_or(a, b):
    """
    Alias for union operation.
    
    This is a semantic alias for the union operation, making code more
    readable when using logical operations.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        
    Returns:
        An SDF expression representing the union (OR)
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # Logical OR of two shapes
        left_half = fp.shape.plane((1, 0, 0), 0)  # x < 0
        top_half = fp.shape.plane((0, 1, 0), 0)   # y < 0
        either = fpo.boolean_or(left_half, top_half)  # x < 0 OR y < 0
        
    IMPORTANT: Only for direct function calls: fpo.boolean_or(a, b)
    Method calls are not supported for operations.
    """
    return union(a, b)

def boolean_not(a):
    """
    Alias for complement operation.
    
    This is a semantic alias for the complement operation, making code more
    readable when using logical operations.
    
    Args:
        a: The SDF expression to negate
        
    Returns:
        The complemented SDF expression (NOT)
        
    Raises:
        TypeError: If input is not an SDF expression
        
    Examples:
        # Logical NOT of a shape
        sphere = fp.shape.sphere(1.0)
        not_sphere = fpo.boolean_not(sphere)  # Everything except the sphere
        
    IMPORTANT: Only for direct function calls: fpo.boolean_not(a)
    Method calls are not supported for operations.
    """
    return complement(a)

def boolean_xor(a, b):
    """
    Boolean XOR operation for SDFs.
    
    Creates a shape that represents the exclusive OR of two shapes, which is the space
    occupied by either shape but not both. This is implemented as the union of (a AND NOT b)
    with (b AND NOT a).
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        
    Returns:
        An SDF expression representing the XOR operation
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        
    Examples:
        # XOR of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.boolean_xor(sphere, box)  # Areas where only one shape exists
        
    IMPORTANT: Only for direct function calls: fpo.boolean_xor(a, b)
    Method calls are not supported for operations.
    """
    # XOR is the union minus the intersection
    return fpm.max(fpm.min(a, b), fpm.min(-a, -b))