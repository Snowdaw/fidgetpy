"""
Color space conversion utilities for fidgetpy expressions.

All functions accept floats or fidgetpy SDF expressions as arguments,
making them usable for procedural per-point colour patterns.
"""


def hsl(h, s=1.0, l=0.5):
    """
    Convert HSL values to an RGB channel dict using expression-compatible math.

    Each argument can be a float or a fidgetpy expression (evaluated per
    surface point), making this ideal for procedural colour patterns.

    Args:
        h: Hue in [0, 1].  0 and 1 both map to red; 1/3 = green; 2/3 = blue.
           For angles from atan2, normalize first:
               h = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
        s: Saturation in [0, 1].  0 = greyscale, 1 = fully saturated.
        l: Lightness in [0, 1].   0 = black, 0.5 = pure hue, 1 = white.

    Returns:
        dict with keys 'r', 'g', 'b', each a float or fidgetpy expression.
        Assign directly to a Container:
            rgb = fpm.hsl(angle, saturation, 0.5)
            fpc.r = rgb['r']
            fpc.g = rgb['g']
            fpc.b = rgb['b']

    Examples:
        # Solid orange (all floats, evaluated at import time)
        fpm.hsl(0.08, 1.0, 0.5)

        # Color wheel on the XY plane (expression-based)
        import math
        import fidgetpy as fp
        import fidgetpy.math as fpm

        angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
        dist  = fpm.sqrt(fp.x()**2 + fp.y()**2)
        rgb   = fpm.hsl(angle, dist, 0.5)

        # Sine rings (hue varies with distance from surface)
        sphere_sdf = fps.sphere(0.5)
        t = fpm.sin(sphere_sdf * 15.0) * 0.5 + 0.5   # 0–1 rings
        rgb = fpm.hsl(t, 1.0, 0.5)

    Notes:
        Uses the abs-formula HSL → RGB conversion (no conditionals needed):
            r_hue = clamp(|h*6 - 3| - 1, 0, 1)
            g_hue = clamp(2 - |h*6 - 2|, 0, 1)
            b_hue = clamp(2 - |h*6 - 4|, 0, 1)
            chroma = (1 - |2L - 1|) * S
            m = L - chroma / 2
            R = r_hue * chroma + m
        This is exact (not an approximation) for H ∈ [0, 1].
    """
    from .basic_math import clamp, abs as _abs

    h6     = h * 6.0
    r_hue  = clamp(_abs(h6 - 3.0) - 1.0, 0.0, 1.0)
    g_hue  = clamp(2.0 - _abs(h6 - 2.0), 0.0, 1.0)
    b_hue  = clamp(2.0 - _abs(h6 - 4.0), 0.0, 1.0)
    chroma = (1.0 - _abs(2.0 * l - 1.0)) * s
    m      = l - chroma * 0.5

    return {
        'r': r_hue * chroma + m,
        'g': g_hue * chroma + m,
        'b': b_hue * chroma + m,
    }
