# Documentation Style Guide for Fidget Ops Module

This document outlines the standardized documentation format for all files in the Fidget ops module to ensure consistency.

## Module Docstrings

Each module should have a docstring with the following structure:

```python
"""
[Module name] operations for Fidget.

This module provides [module name] operations for SDF expressions, including:
- [Category 1] (function1, function2, ...)
- [Category 2] (function3, function4, ...)
...

All operations in this module are for direct function calls only.
"""
```

## Function Docstrings

Each function should have a docstring with the following structure:

```python
def function_name(param1, param2, ...):
    """
    [Brief description of what the function does].
    
    [Additional details about behavior, implementation, or mathematical properties]
    
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
        shape1 = fp.shape.sphere(1.0)
        shape2 = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.function_name(shape1, shape2, ...)  # Expected result description
        
        # [Optional: Additional examples for complex cases]
    
    IMPORTANT: Only for direct function calls: fpo.function_name(param1, param2, ...)
    Method calls are not supported for operations.
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
    sdf: The SDF expression to modify
    radius: The radius of the operation (must be positive)
    amount: A tuple/list/vector of (x, y, z) values
```

## Return Value Documentation

Return values should be documented with:

1. What the value represents
2. Type information (implicit or explicit)
3. Range or special values if applicable

Example:
```
Returns:
    An SDF expression representing the modified shape
```

## Error Documentation

Error cases should be documented with:

1. The exception type
2. The specific conditions that trigger the exception

Example:
```
Raises:
    TypeError: If inputs are not SDF expressions or numeric types
    ValueError: If radius is negative
```

## Examples

Include at least one example for each function, with:

1. Simple usage showing typical parameters
2. Expected result as a comment
3. Additional examples for complex functions or edge cases

Example:
```
Examples:
    # Union of a sphere and a box
    sphere = fp.shape.sphere(1.0)
    box = fp.shape.box(1.0, 1.0, 1.0)
    result = fpo.union(sphere, box)  # Creates a shape that is the union of both
    
    # Create a complex shape with multiple operations
    shape1 = fp.shape.sphere(1.0)
    shape2 = fp.shape.box(1.0, 1.0, 1.0).translate(0.8, 0.0, 0.0)
    result = fpo.smooth_union(shape1, shape2, 0.2)  # Creates a smoothly blended union
```

## Usage Patterns

Always document that ops functions can only be used as direct function calls:

```
IMPORTANT: Only for direct function calls: fpo.function_name(param1, param2, ...)
Method calls are not supported for operations.
```

## Consistency Checklist

- [ ] Module docstring follows standard format
- [ ] All functions have complete docstrings
- [ ] Parameter documentation is consistent and complete
- [ ] Return value documentation is clear
- [ ] Error cases are documented
- [ ] Examples are provided
- [ ] Usage pattern note is included