"""
fidgetpy.color — DEPRECATED.

This module has been removed.  Use the Container API instead:

    import fidgetpy as fp
    import fidgetpy.shape as fps
    import fidgetpy.math as fpm

    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9
    fpc.g = 0.1
    fpc.b = 0.1

    # HSL colors (all args can be floats or fidgetpy expressions)
    import math
    angle = fpm.atan2(fp.y(), fp.x()) / (2 * math.pi) + 0.5
    rgb = fpm.hsl(angle, 1.0, 0.5)
    fpc.r = rgb['r']
    fpc.g = rgb['g']
    fpc.b = rgb['b']

    # Proximity paint
    dot = fps.sphere(0.15).translate(0.5, 0.5, 0)
    fpc.paint(dot, r=0.1, g=0.9, b=0.1)

    # Export / render
    fp.to_vm(fpc)       # dict {name: vm_string}
    fp.splat(fpc)       # Gaussian splatting .ply
    fpc.mesh()          # triangle mesh
"""

import warnings as _warnings

_warnings.warn(
    "fidgetpy.color is deprecated and will be removed in a future release. "
    "Use fp.container() instead. See fidgetpy.color module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)
