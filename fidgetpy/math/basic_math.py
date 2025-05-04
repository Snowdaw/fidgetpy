"""
Basic math functions for Fidget.

This module provides basic mathematical operations for SDF expressions, including:
- Comparison (min, max, clamp)
- Functions (abs, sign, floor, ceil, round)
- Arithmetic (fract, mod, pow, sqrt)

Functions can be used both as standalone functions and as methods on expressions:
- fpm.min(a, b) or a.min(b)
"""

import builtins
import math as py_math
import fidgetpy as fp

def add(a, b):
    """
    Returns the addition of a and b.
    
    This function returns the addition of two values.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The addition value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Addition of two numbers
        result = fpm.add(5.0, 10.0)  # Returns 15.0
        
        # Addition of expression and variable
        sphere = fp.shape.sphere(1.0)
        added_expr = fpm.add(sphere, fp.x())
        
    Can be used as either:
    - fpm.add(a, b)
    - a.add(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'add'):
        return a.add(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'add'):
        # If 'a' is not SDF but 'b' is, call b.add(a)
        return b.add(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("add requires numeric inputs or SDF expressions")


def sub(a, b):
    """
    Returns the subtraction of a and b.
    
    This function returns the subtraction of two values.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The subtraction value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Subtraction of two numbers
        result = fpm.sub(5.0, 10.0)  # Returns -5.0
        
        # Subtraction of expression and variable
        sphere = fp.shape.sphere(1.0)
        sub_expr = fpm.sub(sphere, fp.x())
        
    Can be used as either:
    - fpm.sub(a, b)
    - a.sub(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'sub'):
        return a.sub(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'sub'):
        # If 'a' is not SDF but 'b' is, call b.sub(a)
        return b.sub(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("sub requires numeric inputs or SDF expressions")


def mul(a, b):
    """
    Returns the multiplication of a and b.
    
    This function returns the multiplication of two values.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The multiplication value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Multiplication of two numbers
        result = fpm.mul(5.0, 10.0)  # Returns -5.0
        
        # Multiplication of expression and variable
        sphere = fp.shape.sphere(1.0)
        mul_expr = fpm.mul(sphere, fp.x())
        
    Can be used as either:
    - fpm.mul(a, b)
    - a.mul(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'mul'):
        return a.mul(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'mul'):
        # If 'a' is not SDF but 'b' is, call b.mul(a)
        return b.mul(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("mul requires numeric inputs or SDF expressions")



def div(a, b):
    """
    Returns the division of a and b.
    
    This function returns the division of two values.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The division value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Division of two numbers
        result = fpm.div(5.0, 10.0)  # Returns -5.0
        
        # Division of expression and variable
        sphere = fp.shape.sphere(1.0)
        div_expr = fpm.div(sphere, fp.x())
        
    Can be used as either:
    - fpm.div(a, b)
    - a.div(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'div'):
        return a.div(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'div'):
        # If 'a' is not SDF but 'b' is, call b.div(a)
        return b.div(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a / b
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("div requires numeric inputs or SDF expressions")


def min(a, b):
    """
    Returns the minimum of a and b.
    
    This function returns the smaller of two values. For SDF expressions,
    this is equivalent to a union operation.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The minimum value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Minimum of two numbers
        result = fpm.min(5.0, 10.0)  # Returns 5.0
        
        # Union of two shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        union = fpm.min(sphere, box)
        
    Can be used as either:
    - fpm.min(a, b)
    - a.min(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'min'):
        return a.min(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'min'):
        # If 'a' is not SDF but 'b' is, call b.min(a)
        return b.min(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return builtins.min(a, b)
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("min requires numeric inputs or SDF expressions")

def max(a, b):
    """
    Returns the maximum of a and b.
    
    This function returns the larger of two values. For SDF expressions,
    this is equivalent to an intersection operation.
    
    Args:
        a: First value (number or SDF expression)
        b: Second value (number or SDF expression)
        
    Returns:
        The maximum value of a and b
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Maximum of two numbers
        result = fpm.max(5.0, 10.0)  # Returns 10.0
        
        # Intersection of two shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        intersection = fpm.max(sphere, box)
        
    Can be used as either:
    - fpm.max(a, b)
    - a.max(b) (via extension)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'max'):
        return a.max(b)
    elif hasattr(b, '_is_sdf_expr') and hasattr(b, 'max'):
        # If 'a' is not SDF but 'b' is, call b.max(a)
        return b.max(a)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return builtins.max(a, b)
    else:
        # Raise error if types are incompatible and not SDF expressions
        raise TypeError("max requires numeric inputs or SDF expressions")

def clamp(x, min_val, max_val):
    """
    Clamps the value x between min_val and max_val.
    
    This function restricts a value to a specified range. It's equivalent
    to min(max(x, min_val), max_val).
    
    Args:
        x: The value to clamp (number or SDF expression)
        min_val: The lower bound of the range
        max_val: The upper bound of the range (should be >= min_val)
        
    Returns:
        The clamped value: min_val if x < min_val, max_val if x > max_val, otherwise x
        
    Raises:
        TypeError: If inputs are not numeric or SDF expressions
        
    Examples:
        # Clamp a value between 0 and 1
        result = fpm.clamp(-0.5, 0.0, 1.0)  # Returns 0.0
        result = fpm.clamp(0.5, 0.0, 1.0)   # Returns 0.5
        result = fpm.clamp(1.5, 0.0, 1.0)   # Returns 1.0
        
        # Clamp an SDF expression
        sphere = fp.shape.sphere(1.0)
        clamped_sphere = fpm.clamp(sphere, -0.5, 0.5)
        
    Can be used as either:
    - fpm.clamp(x, min_val, max_val)
    - x.clamp(min_val, max_val) (via extension)
    """
    # If SDF expression, rely on min/max operators working correctly.
    # Avoid calling x.clamp() directly to prevent recursion.
    if hasattr(x, '_is_sdf_expr') or hasattr(min_val, '_is_sdf_expr') or hasattr(max_val, '_is_sdf_expr'):
         # Use the fpm.min/max which handle SDFs correctly
         # This assumes min/max themselves don't cause recursion issues now.
         # If min/max also recurse, they need fixing too.
         # Let's assume min/max are okay for now or fixed separately.
        return min(max(x, min_val), max_val)
    elif isinstance(x, (int, float)) and isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
        # Use Python's min/max for numeric types
        return builtins.min(builtins.max(x, min_val), max_val)
    else:
        # Raise error if types are incompatible
        raise TypeError("clamp requires numeric inputs or SDF expressions")


def abs(x):
    """
    Returns the absolute value of x.
    
    For scalar values, this returns the magnitude of the value.
    For SDF expressions, this can be used to create symmetric shapes.
    
    Args:
        x: The value to get the absolute value of (number or SDF expression)
        
    Returns:
        The absolute value of x
        
    Raises:
        TypeError: If input is not a numeric value or SDF expression
        
    Examples:
        # Absolute value of a number
        result = fpm.abs(-5.0)  # Returns 5.0
        
        # Create a symmetric shape
        plane = fp.shape.plane((1, 0, 0), 0)  # Plane at x=0
        symmetric_shape = fpm.abs(plane)  # Creates a V-shaped valley
        
    Can be used as either:
    - fpm.abs(x)
    - x.abs() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'abs'):
        return x.abs()
    elif isinstance(x, (int, float)):
        return builtins.abs(x)
    else:
        raise TypeError("abs requires a numeric input or SDF expression")

def sign(x):
    """
    Returns the sign of x (-1 if x < 0, 0 if x == 0, 1 if x > 0).
    
    This function extracts just the sign of a value, discarding the magnitude.
    
    Args:
        x: The value to get the sign of (number or SDF expression)
        
    Returns:
        -1 if x < 0, 0 if x == 0, 1 if x > 0
        
    Raises:
        TypeError: If input is not a numeric value or SDF expression
        
    Examples:
        # Sign of a number
        result = fpm.sign(-5.0)  # Returns -1.0
        result = fpm.sign(0.0)   # Returns 0.0
        result = fpm.sign(5.0)   # Returns 1.0
        
        # Create a step function from an SDF
        plane = fp.shape.plane((1, 0, 0), 0)  # Plane at x=0
        step_function = 0.5 * (1.0 - fpm.sign(plane))  # 1 when x < 0, 0 when x > 0
        
    Can be used as either:
    - fpm.sign(x)
    - x.sign() (via extension)
    """
    # Avoid calling x.sign() directly to prevent recursion.
    # Implement using division, relying on abs and operators.
    if hasattr(x, '_is_sdf_expr'):
        # Add small epsilon to prevent division by zero in SDF graph
        epsilon = 1e-10
        # Use fpm.abs which handles SDFs
        return x / (abs(x) + epsilon)
    elif isinstance(x, (int, float)):
        # Handle numeric case directly
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0.0
    else:
        raise TypeError("sign requires a numeric input or SDF expression")

def floor(x):
    """
    Returns the largest integer less than or equal to x.

    Can be used as either:
    - fpm.floor(x)
    - x.floor() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'floor'):
        return x.floor()
    elif isinstance(x, (int, float)):
        return py_math.floor(x)
    else:
        raise TypeError("floor requires a numeric input or SDF expression")

def ceil(x):
    """
    Returns the smallest integer greater than or equal to x.

    Can be used as either:
    - fpm.ceil(x)
    - x.ceil() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'ceil'):
        return x.ceil()
    elif isinstance(x, (int, float)):
        return py_math.ceil(x)
    else:
        raise TypeError("ceil requires a numeric input or SDF expression")

def round(x):
    """
    Returns the nearest integer to x (rounds half to even).

    Can be used as either:
    - fpm.round(x)
    - x.round() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'round'):
        return x.round()
    elif isinstance(x, (int, float)):
        return builtins.round(x)
    else:
        raise TypeError("round requires a numeric input or SDF expression")

def fract(x):
    """
    Returns the fractional part of x (x - floor(x)).

    Result is always non-negative.

    Can be used as either:
    - fpm.fract(x)
    - x.fract() (via extension)
    """
    # If SDF expression, rely on floor and subtraction operators.
    # Avoid calling x.fract() directly to prevent recursion.
    if hasattr(x, '_is_sdf_expr'):
        return x - floor(x) # Use fpm.floor which handles SDFs
    elif isinstance(x, (int, float)):
        return x - py_math.floor(x)
    else:
        raise TypeError("fract requires a numeric input or SDF expression")

def mod(x, y):
    """
    Returns the remainder of x divided by y (x % y).

    Can be used as either:
    - fpm.mod(x, y)
    - x.modulo(y) (via extension, uses 'modulo' method name)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'modulo'):
        return x.modulo(y)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x % y
    else:
        raise TypeError("mod requires numeric inputs or SDF expressions")

def pow(x, y):
    """
    Returns x raised to the power of y (x ** y).

    Can be used as either:
    - fpm.pow(x, y)
    - x ** y (operator overload)
    - x.pow(y) (via extension)
    """
    # If SDF expression, rely on the __pow__ operator overload.
    # Avoid calling x.pow() directly to prevent recursion.
    if hasattr(x, '_is_sdf_expr') or hasattr(y, '_is_sdf_expr'):
         # Check domain for SDF case if possible? Difficult without evaluation.
         # Rely on the underlying Rust implementation to handle domains or errors.
        return x ** y
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        # Add domain check for negative base with non-integer exponent
        if x < 0 and not float(y).is_integer():
             raise ValueError("pow domain error: negative base requires integer exponent")
        # Use py_math.pow for potential C-level speedup and consistency
        return py_math.pow(x, y)
    else:
        raise TypeError("pow requires numeric inputs or SDF expressions")

def sqrt(x):
    """
    Returns the non-negative square root of x.

    Domain: x must be non-negative (>= 0)

    Can be used as either:
    - fpm.sqrt(x)
    - x.sqrt() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'sqrt'):
        return x.sqrt()
    elif isinstance(x, (int, float)):
        if x < 0:
            raise ValueError("sqrt domain error: input must be non-negative")
        return py_math.sqrt(x)
    else:
        raise TypeError("sqrt requires a numeric input or SDF expression")

def exp(x):
    """
    Returns e (Euler's number) raised to the power of x.

    Can be used as either:
    - fpm.exp(x)
    - x.exp() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'exp'):
        return x.exp()
    elif isinstance(x, (int, float)):
        return py_math.exp(x)
    else:
        raise TypeError("exp requires a numeric input or SDF expression")

def ln(x):
    """
    Returns the natural logarithm (base e) of x.

    Domain: x must be positive (> 0)

    Can be used as either:
    - fpm.ln(x)
    - x.ln() (via extension)
    """
    if hasattr(x, '_is_sdf_expr') and hasattr(x, 'ln'):
        return x.ln()
    elif isinstance(x, (int, float)):
        if x <= 0:
            raise ValueError("ln domain error: input must be positive")
        return py_math.log(x)
    else:
        raise TypeError("ln requires a numeric input or SDF expression")