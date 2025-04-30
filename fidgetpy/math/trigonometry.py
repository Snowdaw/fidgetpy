"""
Trigonometric functions for Fidget.

This module provides trigonometric operations for SDF expressions, including:
- sin, cos, tan
- asin, acos, atan, atan2
"""

import math as py_math

def sin(x):
    """
    Returns the sine of x.
    
    This function computes the sine of an angle in radians.
    
    Args:
        x: The angle in radians (number or SDF expression)
        
    Returns:
        The sine of x, a value between -1 and 1
        
    Raises:
        TypeError: If input is not a numeric value or SDF expression
        
    Examples:
        # Sine of a number
        result = fpm.sin(py_math.pi/2)  # Returns 1.0
        
        # Create a sine wave pattern in an SDF
        wave_pattern = fpm.sin(fp.x() * 5.0)
        
    Can be used as either:
    - fpm.sin(x)
    - x.sin() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'sin'):
        return x.sin()
    elif isinstance(x, (int, float)):
        return py_math.sin(x)
    else:
        raise TypeError("sin requires a numeric input or SDF expression")

def cos(x):
    """
    Returns the cosine of x.
    
    This function computes the cosine of an angle in radians.
    
    Args:
        x: The angle in radians (number or SDF expression)
        
    Returns:
        The cosine of x, a value between -1 and 1
        
    Raises:
        TypeError: If input is not a numeric value or SDF expression
        
    Examples:
        # Cosine of a number
        result = fpm.cos(0.0)  # Returns 1.0
        
        # Create a cosine wave pattern in an SDF
        wave_pattern = fpm.cos(fp.x() * 5.0)
        
    Can be used as either:
    - fpm.cos(x)
    - x.cos() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'cos'):
        return x.cos()
    elif isinstance(x, (int, float)):
        return py_math.cos(x)
    else:
        raise TypeError("cos requires a numeric input or SDF expression")

def tan(x):
    """
    Returns the tangent of x.
    
    This function computes the tangent of an angle in radians.
    The tangent is undefined at angles that are odd multiples of π/2.
    
    Args:
        x: The angle in radians (number or SDF expression)
        
    Returns:
        The tangent of x
        
    Raises:
        TypeError: If input is not a numeric value or SDF expression
        
    Examples:
        # Tangent of a number
        result = fpm.tan(py_math.pi/4)  # Returns approximately 1.0
        
        # Create a tangent pattern in an SDF
        pattern = fpm.tan(fp.x())
        
    Can be used as either:
    - fpm.tan(x)
    - x.tan() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'tan'):
        return x.tan()
    elif isinstance(x, (int, float)):
        return py_math.tan(x)
    else:
        raise TypeError("tan requires a numeric input or SDF expression")

def asin(x):
    """
    Returns the arcsine of x.
    
    Domain: x must be between -1 and 1
    Range: Result is between -π/2 and π/2
    
    Can be used as either:
    - fpm.asin(x)
    - x.asin() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'asin'):
        return x.asin()
    elif isinstance(x, (int, float)):
        if x < -1 or x > 1:
            raise ValueError("asin domain error: input must be between -1 and 1")
        return py_math.asin(x)
    else:
        raise TypeError("asin requires a numeric input or SDF expression")

def acos(x):
    """
    Returns the arccosine of x.
    
    Domain: x must be between -1 and 1
    Range: Result is between 0 and π
    
    Can be used as either:
    - fpm.acos(x)
    - x.acos() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'acos'):
        return x.acos()
    elif isinstance(x, (int, float)):
        if x < -1 or x > 1:
            raise ValueError("acos domain error: input must be between -1 and 1")
        return py_math.acos(x)
    else:
        raise TypeError("acos requires a numeric input or SDF expression")

def atan(x):
    """
    Returns the arctangent of x.
    
    Range: Result is between -π/2 and π/2
    
    Can be used as either:
    - fpm.atan(x)
    - x.atan() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'atan'):
        return x.atan()
    elif isinstance(x, (int, float)):
        return py_math.atan(x)
    else:
        raise TypeError("atan requires a numeric input or SDF expression")

def atan2(y, x):
    """
    Returns the arctangent of y/x, using the signs of both arguments to determine the quadrant.
    
    Range: Result is between -π and π
    
    Can be used as either:
    - fpm.atan2(y, x)
    - y.atan2(x) (via extension)
    """
    if hasattr(y, '_is_sdf_expr') and hasattr(y, 'atan2'):
        return y.atan2(x)
    elif isinstance(y, (int, float)) and isinstance(x, (int, float)):
        return py_math.atan2(y, x)
    else:
        raise TypeError("atan2 requires numeric inputs or SDF expressions")