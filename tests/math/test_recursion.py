import pytest
import fidgetpy as fp
import fidgetpy.math as fpm

def test_math_recursion():
    x = fp.x()

    # Basic math functions
    fpm.min(x, 1)
    fpm.max(x, 1)
    fpm.abs(x)
    fpm.floor(x)
    fpm.ceil(x)
    fpm.round(x)
    fpm.sqrt(x)

    # Trigonometric functions
    fpm.sin(x)
    fpm.cos(x)
    fpm.tan(x)
    fpm.asin(x)
    fpm.acos(x)
    fpm.atan(x)
    fpm.atan2(x, 1)

    # Transformations
    fpm.translate(x, 1, 1, 1)
    fpm.scale(x, 2, 2, 2)
    fpm.rotate(x, 1, 1, 1)

    # Vector math
    fpm.length([x, x, x])
    fpm.distance([x, x, x], [1, 1, 1])
    fpm.dot([x, x, x], [1, 1, 1])
    fpm.dot2([x, x, x])
    fpm.normalize([x, x, x])

    # Domain manipulation
    fpm.repeat(x, 1, 1, 1)
    fpm.mirror(x, True, True, True)

    # Interpolation
    fpm.mix(x, 1, 0.5)
    fpm.smoothstep(0, 1, x)
    fpm.step(0.5, x)

    # Logical
    fpm.logical_and(x, True)
    fpm.logical_or(x, True)
    fpm.logical_not(x)
    fpm.logical_xor(x, True)

    # Basic math (cont.)
    fpm.clamp(x, 0, 1)
    fpm.sign(x)
    fpm.fract(x)
    fpm.mod(x, 2)
    fpm.pow(x, 2)
    fpm.exp(x)
    fpm.ln(x + 1.1) # Ensure input > 0

    # Transformations (cont.)
    fpm.remap_xyz(x, x, x, x)
    fpm.remap_affine(x, [1,0,0, 0,1,0, 0,0,1, 0,0,0]) # Identity matrix

    # Vector math (cont.)
    vec = [x, x, x]
    vec2d = [x, x]
    fpm.ndot(vec2d, [1, 1])
    fpm.cross(vec, [1, 0, 0])

    # Domain manipulation (cont.)
    fpm.repeat(x, 1, 1, 1)
    fpm.symmetry(x, True, True, True)

    # Interpolation (cont.)
    fpm.lerp(x, 1, 0.5)
    fpm.smootherstep(0, 1, x)

    # Logical (cont.)
    cond = x > 0 # Create a boolean SDF expression
    fpm.logical_if(cond, 1, 0)
    # Assuming x might have these methods via extension
    if hasattr(x, 'python_and'):
        fpm.python_and(x, 1)
    if hasattr(x, 'python_or'):
        fpm.python_or(x, 1)