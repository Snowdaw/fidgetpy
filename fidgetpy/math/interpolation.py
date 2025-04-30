"""
Interpolation functions for Fidget.

This module provides interpolation operations for SDF expressions, including:
- Linear interpolation (mix/lerp)
- Smoothstep and smootherstep functions
- Step function
"""

from .basic_math import clamp

def mix(a, b, t):
    """
    Linearly interpolates between a and b.
    
    This function performs linear interpolation between values a and b
    using the formula: a * (1-t) + b * t
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        t: Interpolation factor (typically 0-1, but can be any value)
        
    Returns:
        Interpolated value
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Interpolate between two values
        result = fpm.mix(0.0, 10.0, 0.5)  # Returns 5.0
        
        # Interpolate between two SDF expressions
        sphere1 = fp.shape.sphere(1.0)
        sphere2 = fp.shape.sphere(2.0)
        blended = fpm.mix(sphere1, sphere2, 0.3)  # 30% sphere2, 70% sphere1
        
    Can be used as either:
    - fpm.mix(a, b, t)
    - a.mix(b, t) (via extension)
    """
    # Validate inputs - we can't strictly validate types since SDF expressions
    # might not have a common base class, but we can check for basic compatibility
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [a, b, t]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    return a * (1.0 - t) + b * t

def lerp(a, b, t):
    """
    Linearly interpolates between a and b (alias for mix).
    
    This function is identical to mix() and is provided as an alias
    for compatibility with common graphics programming conventions.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        t: Interpolation factor (typically 0-1, but can be any value)
        
    Returns:
        Interpolated value
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Interpolate between two values
        result = fpm.lerp(0.0, 10.0, 0.5)  # Returns 5.0
        
    Can be used as either:
    - fpm.lerp(a, b, t)
    - a.lerp(b, t) (via extension)
    """
    return mix(a, b, t)

def interpolate(x, edge0, edge1):
    """
    Linearly interpolates x from [edge0, edge1] range to [0, 1] range.
    
    This function is similar to smoothstep but without the smoothing.
    It's useful when you need to remap a value from one range to another.
    The result is clamped to the [0, 1] range.
    
    Args:
        x: Value to interpolate
        edge0: Lower edge of the input range
        edge1: Upper edge of the input range
        
    Returns:
        Normalized and clamped value in [0, 1] range
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Remap a value from [-1, 1] to [0, 1]
        result = fpm.interpolate(0.0, -1.0, 1.0)  # Returns 0.5
        
    Can be used as either:
    - fpm.interpolate(x, edge0, edge1)
    - x.interpolate(edge0, edge1) (via extension)
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x, edge0, edge1]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    # Calculate normalized t, clamping to valid range
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)

def smoothstep(edge0, edge1, x):
    """
    Smooth Hermite interpolation between 0 and 1.
    
    This function performs smooth interpolation using a cubic Hermite polynomial.
    It maps x from [edge0, edge1] to [0, 1] with smooth derivatives at endpoints.
    
    Args:
        edge0: Lower edge of the input range
        edge1: Upper edge of the input range
        x: Value to interpolate
        
    Returns:
        0 if x <= edge0, 1 if x >= edge1, and smooth interpolation otherwise
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Smooth transition from 0 to 1 as x goes from 0 to 1
        result = fpm.smoothstep(0.0, 1.0, 0.5)  # Returns 0.5, but with smooth curve
        
    IMPORTANT: Only for direct function calls: fpm.smoothstep(edge0, edge1, x)
    Method calls (x.smoothstep()) are not supported due to parameter order issues.
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [edge0, edge1, x]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    # Calculate normalized t, clamping to valid range
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def smootherstep(edge0, edge1, x):
    """
    Enhanced smooth interpolation with zero 1st and 2nd order derivatives at endpoints.
    
    This provides even smoother interpolation than smoothstep, using a 5th-order polynomial.
    It maps x from [edge0, edge1] to [0, 1] with smooth 1st and 2nd derivatives at endpoints.
    
    Args:
        edge0: Lower edge of the input range
        edge1: Upper edge of the input range
        x: Value to interpolate
        
    Returns:
        0 if x <= edge0, 1 if x >= edge1, and smoother interpolation otherwise
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Even smoother transition from 0 to 1 as x goes from 0 to 1
        result = fpm.smootherstep(0.0, 1.0, 0.5)  # Returns ~0.5, with very smooth curve
        
    IMPORTANT: Only for direct function calls: fpm.smootherstep(edge0, edge1, x)
    Method calls (x.smootherstep()) are not supported due to parameter order issues.
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [edge0, edge1, x]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    # Calculate normalized t, clamping to valid range
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def step(edge, x):
    """
    Step function: returns 0 if x < edge, 1 otherwise.
    
    This function creates a sharp transition from 0 to 1 at the edge value.
    
    Args:
        edge: Edge value where the transition occurs
        x: Value to test
        
    Returns:
        An SDF expression or value evaluating to 0 if x < edge, 1 otherwise
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Create a step function at x=0.5
        result = fpm.step(0.5, 0.7)  # Returns 1.0
        result = fpm.step(0.5, 0.3)  # Returns 0.0
        
    IMPORTANT: Only for direct function calls: fpm.step(edge, x)
    Method calls (x.step()) are not supported due to parameter order issues.
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [edge, x]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    from .logical import logical_not  # Import locally to avoid circular dependency
    
    # Standard implementation for direct function calls
    if hasattr(x, '_is_sdf_expr') or hasattr(edge, '_is_sdf_expr'):
        # At least one is an SDF expression
        return logical_not(x < edge)  # Returns 1 if x >= edge, 0 if x < edge
    else:
        # Both are likely numbers
        return 0.0 if x < edge else 1.0

def threshold(x, threshold_value, low_value=0.0, high_value=1.0):
    """
    Threshold function: returns low_value if x < threshold_value, high_value otherwise.
    
    This is a more flexible version of the step function that allows you to specify
    the output values for both sides of the threshold.
    
    Args:
        x: Value to test
        threshold_value: Value where the transition occurs
        low_value: Value to return when x < threshold_value (default: 0.0)
        high_value: Value to return when x >= threshold_value (default: 1.0)
        
    Returns:
        An SDF expression or value evaluating to low_value if x < threshold_value, 
        high_value otherwise
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Create a threshold function at x=0.5 with custom output values
        result = fpm.threshold(0.7, 0.5, -1.0, 1.0)  # Returns 1.0
        result = fpm.threshold(0.3, 0.5, -1.0, 1.0)  # Returns -1.0
        
    Can be used as either:
    - fpm.threshold(x, threshold_value, low_value, high_value)
    - x.threshold(threshold_value, low_value, high_value) (via extension)
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x, threshold_value, low_value, high_value]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    # Use step function and mix to implement threshold
    s = step(threshold_value, x)
    return mix(low_value, high_value, s)

def pulse(edge0, edge1, x):
    """
    Pulse function: returns 1 when edge0 <= x < edge1, 0 otherwise.
    
    This function creates a rectangular pulse that is 1 inside the range
    [edge0, edge1) and 0 outside.
    
    Args:
        edge0: Lower edge where the pulse begins
        edge1: Upper edge where the pulse ends
        x: Value to test
        
    Returns:
        An SDF expression or value evaluating to 1 when edge0 <= x < edge1, 0 otherwise
        
    Raises:
        TypeError: If inputs are not compatible types
        
    Examples:
        # Create a pulse function between x=0.25 and x=0.75
        result = fpm.pulse(0.25, 0.75, 0.5)  # Returns 1.0
        result = fpm.pulse(0.25, 0.75, 0.2)  # Returns 0.0
        result = fpm.pulse(0.25, 0.75, 0.8)  # Returns 0.0
        
    IMPORTANT: Only for direct function calls: fpm.pulse(edge0, edge1, x)
    Method calls are not supported due to parameter order issues.
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [edge0, edge1, x]):
        raise TypeError("Arguments must be numbers or SDF expressions")
        
    # Use step functions to create a pulse
    return step(edge0, x) - step(edge1, x)