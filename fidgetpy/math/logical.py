"""
Logical operations for SDF expressions.

This module provides function equivalents to the logical operators available
in Fidget (& for AND, | for OR, ~ for NOT).
"""

def logical_and(a, b):
    """
    Return the logical AND of two expressions.
    
    For SDF expressions, this is equivalent to an intersection operation,
    returning the maximum of the two values.

    Can be used as either:
    - fpm.logical_and(a, b)
    - a & b (operator overload)

    Args:
        a: First expression (SDF expression)
        b: Second expression (SDF expression)

    Returns:
        A new expression representing the logical AND of a and b

    Raises:
        TypeError: If inputs are not SDF expressions or do not support the '&' operator
        
    Examples:
        # Create an intersection of two shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        
        # Using logical_and function
        intersection = fpm.logical_and(sphere, box)
        
        # Using operator overload
        intersection = sphere & box
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, '__and__'):
        return a & b
    else:
        # Check if b is SDF and supports __rand__ might be needed for a & b if a is not SDF
        # However, for consistency, we primarily expect SDF expressions here.
        raise TypeError("logical_and requires SDF expressions or objects supporting the '&' operator")


def logical_or(a, b):
    """
    Return the logical OR of two expressions.
    
    For SDF expressions, this is equivalent to a union operation,
    returning the minimum of the two values.

    Can be used as either:
    - fpm.logical_or(a, b)
    - a | b (operator overload)

    Args:
        a: First expression (SDF expression)
        b: Second expression (SDF expression)

    Returns:
        A new expression representing the logical OR of a and b

    Raises:
        TypeError: If inputs are not SDF expressions or do not support the '|' operator
        
    Examples:
        # Create a union of two shapes
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        
        # Using logical_or function
        union = fpm.logical_or(sphere, box)
        
        # Using operator overload
        union = sphere | box
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, '__or__'):
        return a | b
    else:
        raise TypeError("logical_or requires SDF expressions or objects supporting the '|' operator")


def logical_not(a):
    """
    Return the logical NOT of an expression.

    Can be used as either:
    - fpm.logical_not(a)
    - ~a (operator overload)

    Args:
        a: The expression to negate

    Returns:
        A new expression representing the logical NOT of a

    Raises:
        TypeError: If input is not an SDF expression or does not support the '~' operator
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, '__invert__'):
        return ~a
    else:
        raise TypeError("logical_not requires an SDF expression or object supporting the '~' operator")


def logical_xor(a, b):
    """
    Return the logical XOR of two expressions.

    Implemented as (a | b) & ~(a & b).

    Can be used as either:
    - fpm.logical_xor(a, b)
    - (a | b) & ~(a & b) (using operators directly)

    Args:
        a: First expression
        b: Second expression

    Returns:
        A new expression representing the logical XOR of a and b

    Raises:
        TypeError: If inputs are not SDF expressions or do not support logical operators
    """
    # Rely on the underlying logical_and, logical_or, logical_not to raise TypeErrors
    # if a or b are incompatible.
    try:
        return logical_and(logical_or(a, b), logical_not(logical_and(a, b)))
    except TypeError:
         # Re-raise a more specific error for xor context
         raise TypeError("logical_xor requires SDF expressions or objects supporting '&', '|', and '~' operators")


def python_and(a, b):
    """
    Return the Python-style logical AND of two expressions.

    Can be used as either:
    - fpm.python_and(a, b)
    - a.python_and(b) (via extension)

    Args:
        a: First expression (must have .python_and method)
        b: Second expression

    Returns:
        A new expression representing the Python-style AND of a and b

    Raises:
        AttributeError: If first input doesn't have a python_and method
        TypeError: If the types are otherwise incompatible for the operation
    """
    if hasattr(a, 'python_and'):
        return a.python_and(b)
    else:
        # More specific error than TypeError
        raise AttributeError("First argument must have a 'python_and' method")


def python_or(a, b):
    """
    Return the Python-style logical OR of two expressions.

    Can be used as either:
    - fpm.python_or(a, b)
    - a.python_or(b) (via extension)

    Args:
        a: First expression (must have .python_or method)
        b: Second expression

    Returns:
        A new expression representing the Python-style OR of a and b

    Raises:
        AttributeError: If first input doesn't have a python_or method
        TypeError: If the types are otherwise incompatible for the operation
    """
    if hasattr(a, 'python_or'):
        return a.python_or(b)
    else:
        # More specific error than TypeError
        raise AttributeError("First argument must have a 'python_or' method")


def logical_if(condition, true_value, false_value):
    """
    Return a conditional expression based on a logical condition.

    Evaluates to true_value where condition is true (non-zero),
    and false_value where condition is false (zero).

    Can be used as:
    - fpm.logical_if(condition, true_value, false_value)

    Args:
        condition: SDF Expression to use as condition (should evaluate to 0 or non-zero)
        true_value: SDF Expression or numeric value for the true case
        false_value: SDF Expression or numeric value for the false case

    Returns:
        A new expression representing the conditional expression.

    Raises:
        TypeError: If inputs are not SDF expressions or compatible numeric types,
                   or if they don't support the required logical operators.
    """
    # Check types - allow SDF expressions or basic numeric types
    def is_compatible(val):
        return hasattr(val, '_is_sdf_expr') or isinstance(val, (int, float))

    if not is_compatible(condition) or not is_compatible(true_value) or not is_compatible(false_value):
         raise TypeError("logical_if requires SDF expressions or numeric types for all arguments")

    # Implement using mix for proper SDF conditional selection.
    # Assumes 'condition' evaluates to 0 for false and >= 0 (typically 1) for true.
    # The 'round_cone' usage provides a condition from fpm.step, which is 0 or 1.
    try:
        # Import mix locally to avoid circular dependency if logical.py is imported before others in math
        from .interpolation import mix
        return mix(false_value, true_value, condition)
    except (TypeError, AttributeError):
        # Re-raise a more specific error for logical_if context
        raise TypeError("Arguments to logical_if must be SDF expressions or numeric types compatible with fpm.mix")