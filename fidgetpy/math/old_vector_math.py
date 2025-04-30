"""
Vector math functions for Fidget.

This module provides vector math operations for SDF expressions, including:
- length, distance
- dot product, cross product
- normalization
"""

import math as py_math
import fidgetpy as fp

from .basic_math import sqrt, abs # Import abs explicitly

def length(v):
    """
    Returns the length (magnitude) of a vector or the absolute value of a scalar.

    - For lists/tuples (2D/3D): Computes the Euclidean norm sqrt(x*x + y*y + ...).
    - For SDF expressions: Tries v.length(), then sqrt(v.dot(v)), then abs(v).
    - For numeric scalars: Computes abs(v).

    Can be used as either:
    - fpm.length(v)
    - v.length() (via extension, if available)
    """
    # Avoid calling v.length() directly to prevent recursion.
    # Rely on dot product and sqrt for SDF expressions.
    if hasattr(v, '_is_sdf_expr'):
        # Check if it's likely a vector (has dot) or scalar (has abs)
        if hasattr(v, 'dot'):
            # Compute sqrt(dot(v, v))
            return sqrt(dot(v, v)) # Use fpm.dot and fpm.sqrt
        elif hasattr(v, 'abs'):
             # Fallback to abs for scalar-like SDF expressions
             return abs(v) # Use fpm.abs
        else:
             # If it's an SDF expression but none of the above, it's unclear how to get length
             raise TypeError("SDF expression type does not support length calculation")
    elif isinstance(v, (list, tuple)):
        # Use dot2 for potentially simpler/faster calculation
        d2 = dot2(v) # dot2 handles dimension checks and SDF components
        if hasattr(d2, '_is_sdf_expr'):
            return sqrt(d2) # Use Fidget math sqrt if result is SDF
        else:
            # Ensure d2 is non-negative before Python sqrt
            if d2 < 0:
                 raise ValueError("Cannot calculate length of vector with negative squared magnitude")
            return py_math.sqrt(d2) # Use Python math sqrt otherwise
    elif isinstance(v, (int, float)):
        # Handle numeric scalar input
        return py_math.fabs(v) # Use fabs for consistency
    else:
        raise TypeError("Unsupported type for length calculation")


def distance(a, b):
    """
    Returns the Euclidean distance between two points a and b.

    Points can be represented as lists/tuples (2D/3D) or compatible SDF expressions.

    Can be used as either:
    - fpm.distance(a, b)
    - a.distance(b) (via extension, if available)
    """
    # Avoid calling a.distance() directly to prevent recursion.
    # Rely on length(a - b).
    # Check if both are lists/tuples first for specific list logic
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for distance calculation")
        if len(a) == 2:
            # Create diff list, components might be SDF or numeric
            diff = [a[0] - b[0], a[1] - b[1]]
        elif len(a) == 3:
            diff = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        else:
            raise ValueError("Distance calculation only supports 2D or 3D vectors")
        return length(diff) # length handles list input with potentially mixed types
    # Check if types are compatible for subtraction (SDF or numeric)
    elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
         (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
         # Assume they are compatible for subtraction and length calculation
         # This covers scalar distance (abs(a-b)) and SDF expression distance
         # Relies on '-' operator and length function working correctly.
         return length(a - b)
    else:
        raise TypeError("Unsupported types for distance calculation")


def dot(a, b):
    """
    Returns the dot product of two vectors or the product of two scalars.

    - For lists/tuples (2D/3D): Computes a[0]*b[0] + a[1]*b[1] + ...
    - For SDF expressions: Relies on '*' and '+' operators.
    - For numeric scalars: Computes a * b.

    Can be used as either:
    - fpm.dot(a, b)
    - a.dot(b) (via extension, if available)
    """
    # Avoid calling a.dot(b) directly to prevent recursion.
    # Implement using component-wise multiplication and addition.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for dot product")
        if len(a) == 2:
            # Rely on '*' and '+' operators working for SDF/numeric components
            return a[0] * b[0] + a[1] * b[1]
        elif len(a) == 3:
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        else:
            raise ValueError("Dot product only supports 2D or 3D vectors")
    # Check if types are compatible for multiplication (SDF or numeric)
    elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
         (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
         # Assume scalar multiplication or compatible SDF multiplication
         # Relies on '*' operator working correctly.
         return a * b
    else:
        raise TypeError("Unsupported types for dot product calculation")


def dot2(v):
    """
    Returns the dot product of a vector with itself (v.dot(v)), or square for scalars.

    Equivalent to length(v)**2 for vectors.

    Can be used as either:
    - fpm.dot2(v)
    - v.dot(v) (using dot directly)
    """
    # Use the main dot function for consistency (which no longer recurses)
    return dot(v, v)


def ndot(a, b):
    """
    Returns the "negative dot product" of two 2D vectors: a.x*b.x - a.y*b.y.

    This is useful for certain 2D distance functions. Strictly for 2D.

    Can be used as:
    - fpm.ndot(a, b)

    Args:
        a: First 2D vector (list/tuple or object with x, y)
        b: Second 2D vector (list/tuple or object with x, y)
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != 2 or len(b) != 2:
            raise ValueError("ndot requires 2D vectors (lists/tuples)")
        return a[0] * b[0] - a[1] * b[1]
    # Check for attribute access *after* list/tuple check
    elif hasattr(a, 'x') and hasattr(a, 'y') and hasattr(b, 'x') and hasattr(b, 'y'):
        # Handle objects with x, y properties (like potential Vec2 SDF types)
        return a.x * b.x - a.y * b.y
    else:
        raise TypeError("ndot requires 2D vectors as lists/tuples or objects with x,y attributes")


def cross(a, b):
    """
    Returns the cross product of two 3D vectors. Strictly for 3D.

    Can be used as either:
    - fpm.cross(a, b)
    - a.cross(b) (via extension, if available)

    Args:
        a: First 3D vector (list/tuple or SDF expression)
        b: Second 3D vector (list/tuple or SDF expression)

    Returns:
        Cross product vector (list or SDF expression)
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'cross'):
        return a.cross(b)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != 3 or len(b) != 3:
            raise ValueError("Cross product requires 3D vectors (lists/tuples)")
        # Check if components are SDF expressions to return SDF results
        is_sdf = any(hasattr(c, '_is_sdf_expr') for c in a + b)
        
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        
        # If inputs were lists of numbers, return list of numbers.
        # If inputs involved SDF expressions, return list of SDF expressions.
        return [x, y, z]
    else:
        raise TypeError("Cross product requires 3D vector inputs as lists/tuples or SDF expressions")


def normalize(v):
    """
    Returns a normalized vector (unit length) or the sign of a scalar.

    - For vectors (lists/tuples, SDF expressions): Returns v / length(v).
    - For scalars: Returns 1.0 if v >= 0 else -1.0.
    - Handles potential division by zero for numeric types. SDF division by zero
      behavior depends on the underlying implementation.

    Can be used as either:
    - fpm.normalize(v)
    - v.normalize() (via extension, if available)
    """
    # Avoid calling v.normalize() directly to prevent recursion.
    # Rely on length function and division operator.
    l = length(v) # length() handles different types

    # Check for SDF types
    if hasattr(v, '_is_sdf_expr') or hasattr(l, '_is_sdf_expr'):
        # Use logical_if to handle potential division by zero within the SDF graph
        epsilon = 1e-10
        # Check if length is close to zero (using SDF comparison)
        # Need to ensure '<' operator works correctly for SDFs
        is_zero = l < epsilon

        # Import necessary functions locally
        from .logical import logical_if
        from .basic_math import sign # For scalar SDF case

        # If length is zero, return v itself (which should be zero vector/scalar)
        # Otherwise, return v / l.
        # Need to apply this logic component-wise for lists.
        if isinstance(v, (list, tuple)):
            # Create a 'safe' length that adds epsilon only if l is near zero
            safe_l = l + logical_if(is_zero, epsilon, 0.0)
            if len(v) == 2:
                # Divide components by safe_l
                return [v[0] / safe_l, v[1] / safe_l]
            elif len(v) == 3:
                return [v[0] / safe_l, v[1] / safe_l, v[2] / safe_l]
            else:
                 # This path shouldn't be reached if length() worked correctly
                 raise ValueError("Normalize operation only supports 2D or 3D SDF vectors represented as lists")
        else: # Assume v is a scalar SDF expression
             # For scalar, normalize is just sign, but handle zero case using fpm.sign
             return sign(v)

    # Handle non-SDF cases (numeric scalars or lists of numbers)
    elif isinstance(v, (list, tuple)):
        # Length l must be numeric here
        if abs(l) < 1e-10: # Check for zero length
            return [0.0] * len(v) # Return zero vector
        else:
            if len(v) == 2: return [v[0]/l, v[1]/l]
            if len(v) == 3: return [v[0]/l, v[1]/l, v[2]/l]
            # Dimension check should happen in length(), but double-check
            raise ValueError("Normalize operation only supports 2D or 3D vectors")
    elif isinstance(v, (int, float)):
        # Handle scalar: return sign or 0
        if abs(v) < 1e-10:
            return 0.0
        else:
            return 1.0 if v >= 0 else -1.0
    else:
        raise TypeError("Unsupported type for normalize operation")