"""
Vector math functions for Fidget.

This module provides vector math operations for SDF expressions, including:
- length, distance
- dot product, cross product
- normalization
"""

import math as py_math
import fidgetpy as fp

from .basic_math import sqrt, abs  # Import abs explicitly

def length(*components):
    """
    Returns the length (magnitude) of a vector or the absolute value of a scalar.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *components: Either:
            - Individual vector components (x, y, z)
            - A single scalar value
            - A single vector as list/tuple
        
    Returns:
        The Euclidean length of the vector or absolute value of the scalar
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vector has invalid dimensions
    """
    # Handle case when a single argument is provided
    if len(components) == 1:
        v = components[0]
        
        # Handle SDF expressions
        if hasattr(v, '_is_sdf_expr'):
            if hasattr(v, 'dot'):
                return sqrt(dot(v, v))
            elif hasattr(v, 'abs'):
                return abs(v)
            else:
                raise TypeError("SDF expression type does not support length calculation")
        
        # Handle lists/tuples (vectors)
        elif isinstance(v, (list, tuple)):
            # Recursively call with unpacked components
            return length(*v)
        
        # Handle scalar values
        elif isinstance(v, (int, float)):
            return py_math.fabs(v)
        else:
            raise TypeError("Unsupported type for length calculation")
    
    # Handle individual components (2D, 3D, or higher)
    else:
        # Validate inputs
        if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in components):
            raise TypeError("Vector components must be numbers or SDF expressions")
            
        # Compute sum of squares
        sum_sq = 0
        for component in components:
            sum_sq += component * component
            
        return sqrt(sum_sq)

def distance(*args):
    """
    Returns the Euclidean distance between two points.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *args: Either:
            - Two points as vectors: (a, b)
            - Individual components of two points: (x1, y1, x2, y2) for 2D
              or (x1, y1, z1, x2, y2, z2) for 3D
        
    Returns:
        The Euclidean distance between the points
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors have invalid or mismatched dimensions
    """
    # Case 1: Two arguments, each being a point (as vector or scalar)
    if len(args) == 2:
        a, b = args
        
        # Handle lists/tuples (vectors)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                raise ValueError("Vectors must have the same dimension for distance calculation")
            
            # Calculate differences for each component
            diffs = [a[i] - b[i] for i in range(len(a))]
            
            # Return length of the difference vector
            return length(diffs)
        
        # Handle scalar or SDF expression types
        elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
             (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
            return length(a - b)
        else:
            raise TypeError("Unsupported types for distance calculation")
    
    # Case 2: Individual components for two points (4 args for 2D, 6 args for 3D)
    elif len(args) == 4:  # 2D points: x1, y1, x2, y2
        x1, y1, x2, y2 = args
        return length(x1 - x2, y1 - y2)
    elif len(args) == 6:  # 3D points: x1, y1, z1, x2, y2, z2
        x1, y1, z1, x2, y2, z2 = args
        return length(x1 - x2, y1 - y2, z1 - z2)
    else:
        raise ValueError("distance() requires either 2 point arguments or 4/6 component arguments")

def dot(*args):
    """
    Returns the dot product of two vectors or the product of two scalars.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *args: Either:
            - Two vectors: (a, b)
            - Individual components of two vectors: (x1, y1, x2, y2) for 2D
              or (x1, y1, z1, x2, y2, z2) for 3D
        
    Returns:
        The dot product of the vectors or product of the scalars
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors have invalid or mismatched dimensions
    """
    # Case 1: Two arguments, each being a vector or scalar
    if len(args) == 2:
        a, b = args
        
        # Handle lists/tuples (vectors)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                raise ValueError("Vectors must have the same dimension for dot product")
            
            # Handle empty vectors
            if len(a) == 0:
                return 0
                
            # Calculate dot product
            return sum(a[i] * b[i] for i in range(len(a)))
        
        # Handle scalar or SDF expression types
        elif (hasattr(a, '_is_sdf_expr') or isinstance(a, (int, float))) and \
             (hasattr(b, '_is_sdf_expr') or isinstance(b, (int, float))):
            return a * b
        else:
            raise TypeError("Unsupported types for dot product calculation")
    
    # Case 2: Individual components for two vectors (4 args for 2D, 6 args for 3D)
    elif len(args) == 4:  # 2D vectors: x1, y1, x2, y2
        x1, y1, x2, y2 = args
        return x1 * x2 + y1 * y2
    elif len(args) == 6:  # 3D vectors: x1, y1, z1, x2, y2, z2
        x1, y1, z1, x2, y2, z2 = args
        return x1 * x2 + y1 * y2 + z1 * z2
    else:
        raise ValueError("dot() requires either 2 vector arguments or 4/6 component arguments")

def dot2(*components):
    """
    Returns the dot product of a vector with itself (squared length).
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *components: Either:
            - Individual vector components (x, y, z)
            - A single vector as list/tuple
        
    Returns:
        The squared length of the vector or square of the scalar
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vector has invalid dimensions
    """
    # Handle case when a single argument is provided
    if len(components) == 1:
        v = components[0]
        
        # If it's a list/tuple, use dot product with itself
        if isinstance(v, (list, tuple)):
            return dot(v, v)
        else:
            # For scalars or SDF expressions, square it
            return v * v
    
    # Handle individual components (recursively call dot with the same components twice)
    else:
        return dot(*(list(components) + list(components)))

def ndot(*args):
    """
    Returns the "negative dot product" of two 2D vectors: a.x*b.x - a.y*b.y.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *args: Either:
            - Two 2D vectors: (a, b)
            - Individual components of two 2D vectors: (x1, y1, x2, y2)
        
    Returns:
        The negative dot product of the vectors
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors are not 2D
    """
    # Case 1: Two arguments, each being a 2D vector
    if len(args) == 2:
        a, b = args
        
        # Handle lists/tuples
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != 2 or len(b) != 2:
                raise ValueError("ndot requires 2D vectors (lists/tuples)")
            return a[0] * b[0] - a[1] * b[1]
        
        # Handle objects with x,y attributes
        elif hasattr(a, 'x') and hasattr(a, 'y') and hasattr(b, 'x') and hasattr(b, 'y'):
            return a.x * b.x - a.y * b.y
        else:
            raise TypeError("ndot requires 2D vectors as lists/tuples or objects with x,y attributes")
    
    # Case 2: Individual components for two 2D vectors
    elif len(args) == 4:  # 2D vectors: x1, y1, x2, y2
        x1, y1, x2, y2 = args
        # Validate inputs
        if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
                  for v in [x1, y1, x2, y2]):
            raise TypeError("Vector components must be numbers or SDF expressions")
            
        return x1 * x2 - y1 * y2
    else:
        raise ValueError("ndot() requires either 2 vector arguments or 4 component arguments")

def cross(*args):
    """
    Returns the cross product of two vectors.
    
    For 2D vectors, it returns the z-component of the cross product (scalar).
    For 3D vectors, it returns the full cross product vector.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *args: Either:
            - Two vectors: (a, b)
            - Individual components of two vectors: (x1, y1, x2, y2) for 2D
              or (x1, y1, z1, x2, y2, z2) for 3D
        
    Returns:
        The cross product (vector for 3D, scalar for 2D)
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vectors have invalid dimensions
    """
    # Case 1: Two arguments, each being a vector
    if len(args) == 2:
        a, b = args
        
        # Handle SDF expressions with cross method
        if hasattr(a, '_is_sdf_expr') and hasattr(a, 'cross'):
            return a.cross(b)
        
        # Handle lists/tuples
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            # 2D cross product (returns scalar)
            if len(a) == 2 and len(b) == 2:
                return a[0] * b[1] - a[1] * b[0]
            
            # 3D cross product (returns vector)
            elif len(a) == 3 and len(b) == 3:
                x = a[1] * b[2] - a[2] * b[1]
                y = a[2] * b[0] - a[0] * b[2]
                z = a[0] * b[1] - a[1] * b[0]
                return [x, y, z]
            else:
                raise ValueError("Cross product requires 2D or 3D vectors (lists/tuples)")
        else:
            raise TypeError("Cross product requires vector inputs as lists/tuples or SDF expressions")
    
    # Case 2: Individual components for two vectors
    elif len(args) == 4:  # 2D vectors: x1, y1, x2, y2
        x1, y1, x2, y2 = args
        # Validate inputs
        if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
                  for v in [x1, y1, x2, y2]):
            raise TypeError("Vector components must be numbers or SDF expressions")
            
        # 2D cross product (returns scalar)
        return x1 * y2 - y1 * x2
        
    elif len(args) == 6:  # 3D vectors: x1, y1, z1, x2, y2, z2
        x1, y1, z1, x2, y2, z2 = args
        # Validate inputs
        if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') 
                  for v in [x1, y1, z1, x2, y2, z2]):
            raise TypeError("Vector components must be numbers or SDF expressions")
            
        # 3D cross product (returns vector)
        x = y1 * z2 - z1 * y2
        y = z1 * x2 - x1 * z2
        z = x1 * y2 - y1 * x2
        
        return (x, y, z)
    else:
        raise ValueError("cross() requires either 2 vector arguments or 4/6 component arguments")

def normalize(*components):
    """
    Returns a normalized vector (unit length) or the sign of a scalar.
    
    This function handles both explicit components and vector lists/tuples.
    
    Args:
        *components: Either:
            - Individual vector components (x, y, z)
            - A single vector as list/tuple
            - A single scalar value
        
    Returns:
        The normalized vector or sign of the scalar
        
    Raises:
        TypeError: If the inputs are not compatible types
        ValueError: If the vector has invalid dimensions
    """
    # Handle case when a single argument is provided
    if len(components) == 1:
        v = components[0]
        
        # Handle scalar SDF expressions (normalize is sign)
        if hasattr(v, '_is_sdf_expr') and not isinstance(v, (list, tuple)):
            from .basic_math import sign
            return sign(v)
                
        # Handle lists/tuples (vectors)
        elif isinstance(v, (list, tuple)):
            # Recursively call with unpacked components
            normalized = normalize(*v)
            
            # Re-pack the normalized components into the same container type
            return list(normalized) if isinstance(v, list) else normalized
        
        # Handle scalar values
        elif isinstance(v, (int, float)):
            if abs(v) < 1e-10:
                return 0.0
            else:
                return 1.0 if v >= 0 else -1.0
        else:
            raise TypeError("Unsupported type for normalize operation")
    
    # Handle individual components (2D, 3D, or higher)
    else:
        # Validate inputs
        if not all(isinstance(v, (int, float)) or hasattr(v, '_is_sdf_expr') for v in components):
            raise TypeError("Vector components must be numbers or SDF expressions")
            
        # Get the length
        l = length(*components)
        
        # Handle SDF expressions with special consideration for division-by-zero
        if any(hasattr(v, '_is_sdf_expr') for v in components) or hasattr(l, '_is_sdf_expr'):
            # For SDF expressions: Let the backend handle division by zero
            # This is preferable in many SDF systems as they handle this case optimally
            # in their compiled output, without needing explicit branching
            return tuple(c / l for c in components)
        
        # Handle numeric values with explicit zero check
        else:
            if abs(l) < 1e-10:
                return tuple(0.0 for _ in components)
            else:
                return tuple(c / l for c in components)