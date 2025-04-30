# Documentation Style Guide for Fidget Shape Module

This document outlines the standardized documentation format for all files in the Fidget shape module to ensure consistency.

## Module Docstrings

Each module should have a docstring with the following structure:

```python
"""
[Module name] shapes for Fidget.

This module provides [module name] shapes for SDF expressions, including:
- [Category 1] (function1, function2, ...)
- [Category 2] (function3, function4, ...)
...
"""
```

## Function Docstrings

Each function should have a docstring with the following structure:

```python
def function_name(param1, param2, ...):
    """
    [Brief description of what the shape is].
    
    [Additional details about the shape, its geometry, or mathematical properties]
    
    [Optional: Domain/range constraints if applicable]
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        ...
    
    Returns:
        Description of return value
    
    Raises:
        TypeError: Conditions that trigger TypeError
        ValueError: Conditions that trigger ValueError
        [Other exceptions as applicable]
    
    Examples:
        # Simple example
        shape = fps.function_name(param1, param2, ...)  # Creates a [shape description]
        
        # [Optional: Additional examples for complex cases]
    """
```

## Parameter Documentation

Parameters should be documented with:

1. Type information (implicit or explicit)
2. Valid ranges or constraints
3. Default values and their meaning (for optional parameters)

Example:
```
Args:
    radius: The radius of the sphere (must be positive)
    height: The height of the cylinder (must be positive)
    angle: The angle in degrees (range: 0-90)
```

## Return Value Documentation

Return values should be documented with:

1. What the value represents
2. Type information (implicit or explicit)
3. Range or special values if applicable

Example:
```
Returns:
    An SDF expression representing a sphere centered at the origin
```

## Error Documentation

Error cases should be documented with:

1. The exception type
2. The specific conditions that trigger the exception

Example:
```
Raises:
    TypeError: If inputs are not numeric types
    ValueError: If radius is negative
```

## Examples

Include at least one example for each function, with:

1. Simple usage showing typical parameters
2. Expected result as a comment
3. Additional examples for complex cases or different parameter combinations

Example:
```
Examples:
    # Create a basic sphere
    sphere = fps.sphere(1.0)  # Creates a sphere with radius 1.0
    
    # Create a larger sphere
    large_sphere = fps.sphere(5.0)  # Creates a sphere with radius 5.0
```

## Consistency Checklist

- [ ] Module docstring follows standard format
- [ ] All functions have complete docstrings
- [ ] Parameter documentation is consistent and complete
- [ ] Return value documentation is clear
- [ ] Error cases are documented
- [ ] Examples are provided