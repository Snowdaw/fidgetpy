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

def length_2d(x, y):
    """
    Returns the length (magnitude) of a 2D vector.
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        
    Returns:
        The Euclidean length of the vector
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in [x, y]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    return sqrt(x*x + y*y)

def length_3d(x, y, z):
    """
    Returns the length (magnitude) of a 3D vector.
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        z: The z component of the vector
        
    Returns:
        The Euclidean length of the vector
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in [x, y, z]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    return sqrt(x*x + y*y + z*z)

def length(v):
    """
    Returns the length (magnitude) of a vector or the absolute value of a scalar.
    
    This is a legacy function that supports both scalar values and vector lists/tuples.
    For new code, prefer using length_2d or length_3d with explicit components.

    Args:
        v: A scalar value, SDF expression, or vector as list/tuple
        
    Returns:
        The length of the vector or absolute value of the scalar
        
    Raises:
        TypeError: If the input is not a supported type
        ValueError: If the vector has invalid dimensions
    """
    # Handle SDF expressions
    if hasattr(v, '_is_sdf_expr'):
        if hasattr(v, 'dot'):
            return sqrt(dot(v, v))
        elif hasattr(v, 'abs'):
            return abs(v)
        else:
            raise TypeError("SDF expression type does not support length calculation")
    
    # Handle lists/tuples
    elif isinstance(v, (list, tuple)):
        if len(v) == 2:
            return length_2d(v[0], v[1])
        elif len(v) == 3:
            return length_3d(v[0], v[1], v[2])
        else:
            raise ValueError("Vector length calculation only supports 2D or 3D vectors")
    
    # Handle scalar values
    elif isinstance(v, (int, float)):
        return py_math.fabs(v)
    else:
        raise TypeError("Unsupported type for length calculation")

def distance_2d(x1, y1, x2, y2):
    """
    Returns the Euclidean distance between two 2D points.
    
    Args:
        x1: The x component of the first point
        y1: The y component of the first point
        x2: The x component of the second point
        y2: The y component of the second point
        
    Returns:
        The Euclidean distance between the points
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, x2, y2]):
        raise TypeError("Point components must be numbers or SDF expressions")
        
    return length_2d(x1 - x2, y1 - y2)

def distance_3d(x1, y1, z1, x2, y2, z2):
    """
    Returns the Euclidean distance between two 3D points.
    
    Args:
        x1: The x component of the first point
        y1: The y component of the first point
        z1: The z component of the first point
        x2: The x component of the second point
        y2: The y component of the second point
        z2: The z component of the second point
        
    Returns:
        The Euclidean distance between the points
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, z1, x2, y2, z2]):
        raise TypeError("Point components must be numbers or SDF expressions")
        
    return length_3d(x1 - x2, y1 - y2, z1 - z2)

def distance(a, b):
    """
    Returns the Euclidean distance between two points a and b.
    
    This is a legacy function that supports both scalar values and vector lists/tuples.
    For new code, prefer using distance_2d or distance_3d with explicit components.
    
    Args:
        a: First point (scalar, SDF expression, or vector as list/tuple)
        b: Second point (scalar, SDF expression, or vector as list/tuple)
        
    Returns:
        The Euclidean distance between the points
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors have invalid or mismatched dimensions
    """
    # Handle lists/tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for distance calculation")
        
        if len(a) == 2:
            return distance_2d(a[0], a[1], b[0], b[1])
        elif len(a) == 3:
            return distance_3d(a[0], a[1], a[2], b[0], b[1], b[2])
        else:
            raise ValueError("Distance calculation only supports 2D or 3D vectors")
    
    # Handle scalar or SDF expression types
    elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
         (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
        return length(a - b)
    else:
        raise TypeError("Unsupported types for distance calculation")

def dot_2d(x1, y1, x2, y2):
    """
    Returns the dot product of two 2D vectors.
    
    Args:
        x1: The x component of the first vector
        y1: The y component of the first vector
        x2: The x component of the second vector
        y2: The y component of the second vector
        
    Returns:
        The dot product of the vectors
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, x2, y2]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    return x1 * x2 + y1 * y2

def dot_3d(x1, y1, z1, x2, y2, z2):
    """
    Returns the dot product of two 3D vectors.
    
    Args:
        x1: The x component of the first vector
        y1: The y component of the first vector
        z1: The z component of the first vector
        x2: The x component of the second vector
        y2: The y component of the second vector
        z2: The z component of the second vector
        
    Returns:
        The dot product of the vectors
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, z1, x2, y2, z2]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    return x1 * x2 + y1 * y2 + z1 * z2

def dot(a, b):
    """
    Returns the dot product of two vectors or the product of two scalars.
    
    This is a legacy function that supports both scalar values and vector lists/tuples.
    For new code, prefer using dot_2d or dot_3d with explicit components.
    
    Args:
        a: First vector (scalar, SDF expression, or vector as list/tuple)
        b: Second vector (scalar, SDF expression, or vector as list/tuple)
        
    Returns:
        The dot product of the vectors or product of the scalars
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors have invalid or mismatched dimensions
    """
    # Handle lists/tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension for dot product")
        
        # Perform element-wise multiplication and summation for lists/tuples
        if len(a) == 2:
            # Ensure components are treated as expressions/numbers
            a0, a1 = a
            b0, b1 = b
            return a0 * b0 + a1 * b1
        elif len(a) == 3:
            # Ensure components are treated as expressions/numbers
            a0, a1, a2 = a
            b0, b1, b2 = b
            return a0 * b0 + a1 * b1 + a2 * b2
        else:
            raise ValueError("Dot product only supports 2D or 3D vectors")
    
    # Handle scalar or SDF expression types (if not lists)
    elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
         (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
        # This case handles single SDF expressions or numbers, not vectors represented as lists
        return a * b
    else:
        raise TypeError("Unsupported types for dot product calculation")

def dot2_2d(x, y):
    """
    Returns the dot product of a 2D vector with itself (squared length).
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        
    Returns:
        The squared length of the vector
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    return dot_2d(x, y, x, y)

def dot2_3d(x, y, z):
    """
    Returns the dot product of a 3D vector with itself (squared length).
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        z: The z component of the vector
        
    Returns:
        The squared length of the vector
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    return dot_3d(x, y, z, x, y, z)

def dot2(v):
    """
    Returns the dot product of a vector with itself (v.dot(v)), or square for scalars.
    
    This is a legacy function that supports both scalar values and vector lists/tuples.
    For new code, prefer using dot2_2d or dot2_3d with explicit components.
    
    Args:
        v: A scalar value, SDF expression, or vector as list/tuple
        
    Returns:
        The squared length of the vector or square of the scalar
        
    Raises:
        TypeError: If the input is not a supported type
        ValueError: If the vector has invalid dimensions
    """
    return dot(v, v)

def ndot_2d(x1, y1, x2, y2):
    """
    Returns the "negative dot product" of two 2D vectors: x1*x2 - y1*y2.
    
    This is useful for certain 2D distance functions.
    
    Args:
        x1: The x component of the first vector
        y1: The y component of the first vector
        x2: The x component of the second vector
        y2: The y component of the second vector
        
    Returns:
        The negative dot product of the vectors
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, x2, y2]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    return x1 * x2 - y1 * y2

def ndot(a, b):
    """
    Returns the "negative dot product" of two 2D vectors: a.x*b.x - a.y*b.y.
    
    This is a legacy function that supports vector lists/tuples and objects with x,y attributes.
    For new code, prefer using ndot_2d with explicit components.
    
    Args:
        a: First 2D vector (list/tuple or object with x, y attributes)
        b: Second 2D vector (list/tuple or object with x, y attributes)
        
    Returns:
        The negative dot product of the vectors
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors are not 2D
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != 2 or len(b) != 2:
            raise ValueError("ndot requires 2D vectors (lists/tuples)")
        return ndot_2d(a[0], a[1], b[0], b[1])
    elif hasattr(a, 'x') and hasattr(a, 'y') and hasattr(b, 'x') and hasattr(b, 'y'):
        return ndot_2d(a.x, a.y, b.x, b.y)
    else:
        raise TypeError("ndot requires 2D vectors as lists/tuples or objects with x,y attributes")

def cross_3d(x1, y1, z1, x2, y2, z2):
    """
    Returns the cross product of two 3D vectors.
    
    Args:
        x1: The x component of the first vector
        y1: The y component of the first vector
        z1: The z component of the first vector
        x2: The x component of the second vector
        y2: The y component of the second vector
        z2: The z component of the second vector
        
    Returns:
        The cross product as a tuple (x, y, z)
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
              for v in [x1, y1, z1, x2, y2, z2]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    
    return (x, y, z)

def cross(a, b):
    """
    Returns the cross product of two 3D vectors.
    
    This is a legacy function that supports vector lists/tuples and SDF expressions.
    For new code, prefer using cross_3d with explicit components.
    
    Args:
        a: First 3D vector (list/tuple or SDF expression)
        b: Second 3D vector (list/tuple or SDF expression)
        
    Returns:
        The cross product vector (list or SDF expression)
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors are not 3D
    """
    if hasattr(a, '_is_sdf_expr') and hasattr(a, 'cross'):
        return a.cross(b)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != 3 or len(b) != 3:
            raise ValueError("Cross product requires 3D vectors (lists/tuples)")
            
        result = cross_3d(a[0], a[1], a[2], b[0], b[1], b[2])
        return list(result)  # Convert to list for backward compatibility
    else:
        raise TypeError("Cross product requires 3D vector inputs as lists/tuples or SDF expressions")

def normalize_2d(x, y):
    """
    Returns a normalized 2D vector (unit length).
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        
    Returns:
        The normalized vector as a tuple (x, y)
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in [x, y]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    l = length_2d(x, y)
    
    # Handle SDF expressions
    if hasattr(x, '_is_sdf_expr') or hasattr(y, '_is_sdf_expr') or hasattr(l, '_is_sdf_expr'):
        # Import necessary functions locally
        from .logical import logical_if
        
        # Use logical_if to handle potential division by zero
        epsilon = 1e-10
        is_zero = l < epsilon
        safe_l = l + logical_if(is_zero, epsilon, 0.0)
        
        return (x / safe_l, y / safe_l)
    
    # Handle numeric values
    else:
        if abs(l) < 1e-10:
            return (0.0, 0.0)
        else:
            return (x / l, y / l)

def normalize_3d(x, y, z):
    """
    Returns a normalized 3D vector (unit length).
    
    Args:
        x: The x component of the vector
        y: The y component of the vector
        z: The z component of the vector
        
    Returns:
        The normalized vector as a tuple (x, y, z)
        
    Raises:
        TypeError: If the inputs are not compatible types
    """
    # Validate inputs
    if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in [x, y, z]):
        raise TypeError("Vector components must be numbers or SDF expressions")
        
    l = length_3d(x, y, z)
    
    # Handle SDF expressions
    if hasattr(x, '_is_sdf_expr') or hasattr(y, '_is_sdf_expr') or hasattr(z, '_is_sdf_expr') or hasattr(l, '_is_sdf_expr'):
        # Import necessary functions locally
        from .logical import logical_if
        
        # Use logical_if to handle potential division by zero
        epsilon = 1e-10
        is_zero = l < epsilon
        safe_l = l + logical_if(is_zero, epsilon, 0.0)
        
        return (x / safe_l, y / safe_l, z / safe_l)
    
    # Handle numeric values
    else:
        if abs(l) < 1e-10:
            return (0.0, 0.0, 0.0)
        else:
            return (x / l, y / l, z / l)

def normalize(v):
    """
    Returns a normalized vector (unit length) or the sign of a scalar.
    
    This is a legacy function that supports both scalar values and vector lists/tuples.
    For new code, prefer using normalize_2d or normalize_3d with explicit components.
    
    Args:
        v: A scalar value, SDF expression, or vector as list/tuple
        
    Returns:
        The normalized vector or sign of the scalar
        
    Raises:
        TypeError: If the input is not a supported type
        ValueError: If the vector has invalid dimensions
    """
    # Get the length
    l = length(v)
    
    # Handle SDF expressions
    if hasattr(v, '_is_sdf_expr') or hasattr(l, '_is_sdf_expr'):
        # Import necessary functions locally
        from .logical import logical_if
        from .basic_math import sign
        
        # Use logical_if to handle potential division by zero
        epsilon = 1e-10
        is_zero = l < epsilon
        
        # Handle vector vs scalar SDF expressions
        if isinstance(v, (list, tuple)):
            safe_l = l + logical_if(is_zero, epsilon, 0.0)
            
            if len(v) == 2:
                return [v[0] / safe_l, v[1] / safe_l]
            elif len(v) == 3:
                return [v[0] / safe_l, v[1] / safe_l, v[2] / safe_l]
            else:
                raise ValueError("Normalize operation only supports 2D or 3D SDF vectors represented as lists")
        else:
            # For scalar, normalize is just sign
            return sign(v)
    
    # Handle lists/tuples of numeric values
    elif isinstance(v, (list, tuple)):
        if abs(l) < 1e-10:
            return [0.0] * len(v)
        else:
            if len(v) == 2:
                return [v[0] / l, v[1] / l]
            elif len(v) == 3:
                return [v[0] / l, v[1] / l, v[2] / l]
            else:
                raise ValueError("Normalize operation only supports 2D or 3D vectors")
    
    # Handle scalar numeric values
    elif isinstance(v, (int, float)):
        if abs(v) < 1e-10:
            return 0.0
        else:
            return 1.0 if v >= 0 else -1.0
    else:
        raise TypeError("Unsupported type for normalize operation")