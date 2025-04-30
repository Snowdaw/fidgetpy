"""
Interpolation functions for Fidget.

This module provides interpolation operations for SDF expressions, including:
- Linear interpolation (mix/lerp)
- Smoothstep
- Step function
"""

from .basic_math import clamp

def mix(a, b, t):
    """
    Linearly interpolates between a and b.
    
    Can be used as either:
    - fpm.mix(a, b, t)
    - a.mix(b, t) (via extension)
    
    Args:
        a: First value
        b: Second value
        t: Interpolation factor (0-1)
        
    Returns:
        Interpolated value
    """
    return a * (1.0 - t) + b * t

def lerp(a, b, t):
    """
    Linearly interpolates between a and b (alias for mix).

    Can be used as either:
    - fpm.lerp(a, b, t)
    - a.lerp(b, t) (via extension)

    Args:
        a: First value
        b: Second value
        t: Interpolation factor (0-1)

    Returns:
        Interpolated value
    """
    return mix(a, b, t)

def smoothstep(edge0, edge1, x):
    """
    Smooth Hermite interpolation between 0 and 1.

    IMPORTANT: Only for direct function calls: fpm.smoothstep(edge0, edge1, x)
    Method calls (x.smoothstep()) are not supported due to parameter order issues.

    Args:
        edge0: Lower edge
        edge1: Upper edge
        x: Value to interpolate

    Returns:
        0 if x <= edge0, 1 if x >= edge1, and smooth interpolation otherwise
    """
    # Calculate normalized t, clamping to valid range
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def step(edge, x):
    """
    Step function: returns 0 if x < edge, 1 otherwise.

    IMPORTANT: Only for direct function calls: fpm.step(edge, x)
    Method calls (x.step()) are not supported due to parameter order issues.

    Args:
        edge: Edge value
        x: Value to test

    Returns:
        An SDF expression evaluating to 0 if x < edge, 1 otherwise
    """
    from .logical import logical_not  # Import locally to avoid circular dependency
    
    # Standard implementation for direct function calls
    if hasattr(x, '_is_sdf_expr') or hasattr(edge, '_is_sdf_expr'):
        # At least one is an SDF expression
        return logical_not(x < edge)  # Returns 1 if x >= edge, 0 if x < edge
    else:
        # Both are likely numbers
        return 0.0 if x < edge else 1.0

def smootherstep(edge0, edge1, x):
    """
    Enhanced smooth interpolation with zero 1st and 2nd order derivatives at endpoints.

    This provides even smoother interpolation than smoothstep, using a 5th-order polynomial.

    IMPORTANT: Only for direct function calls: fpm.smootherstep(edge0, edge1, x)
    Method calls (x.smootherstep()) are not supported due to parameter order issues.

    Args:
        edge0: Lower edge
        edge1: Upper edge
        x: Value to interpolate

    Returns:
        0 if x <= edge0, 1 if x >= edge1, and smoother interpolation otherwise
    """
    # Calculate normalized t, clamping to valid range
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)