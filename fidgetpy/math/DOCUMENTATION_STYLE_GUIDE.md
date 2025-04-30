# Documentation Style Guide for Fidget Math Module

This document outlines the standardized documentation format for all files in the Fidget math module to ensure consistency.

## Module Docstrings

Each module should have a docstring with the following structure:

```python
"""
[Module name] functions for Fidget.

This module provides [module name] operations for SDF expressions, including:
- [Category 1] (function1, function2, ...)
- [Category 2] (function3, function4, ...)
...

[Optional: Brief usage overview if needed]
"""
```

## Function Docstrings

Each function should have a docstring with the following structure:

```python
def function_name(param1, param2, ...):
    """
    [Brief description of what the function does].
    
    [Optional: Additional details about behavior, implementation, or mathematical properties]
    
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
        result = fpm.function_name(1.0, 2.0)  # Returns expected_result
        
        # [Optional: Additional examples for complex cases]
    
    Can be used as either:
    - fpm.function_name(param1, param2, ...)
    - param1.function_name(param2, ...) (via extension, if applicable)
    [Or note if only function call style is supported]
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
    edge0: Lower edge of the input range
    edge1: Upper edge of the input range (must be > edge0)
```

## Return Value Documentation

Return values should be documented with:

1. What the value represents
2. Type information (implicit or explicit)
3. Range or special values if applicable

Example:
```
Returns:
    A new SDF expression representing the sphere with the given radius
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
    # Create a sphere with radius 1.0
    sphere = fpm.sphere(1.0)
    
    # Create a sphere and translate it
    moved_sphere = fpm.sphere(2.0).translate(1.0, 0.0, 0.0)
```

## Usage Patterns

Always document how the function can be used, especially if it supports both function call and method call styles:

```
Can be used as either:
- fpm.function_name(param1, param2)
- param1.function_name(param2) (via extension)
```

Or if only function call style is supported:

```
IMPORTANT: Only for direct function calls: fpm.function_name(param1, param2)
Method calls are not supported due to parameter order issues.
```

## Consistency Checklist

- [ ] Module docstring follows standard format
- [ ] All functions have complete docstrings
- [ ] Parameter documentation is consistent and complete
- [ ] Return value documentation is clear
- [ ] Error cases are documented
- [ ] Examples are provided
- [ ] Usage patterns are documented