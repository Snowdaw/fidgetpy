"""
Miscellaneous operations for Fidget.

This module provides various utility operations for SDF expressions that
don't fit neatly into other categories, including:
- Special union types (smooth_step_union, chamfer_union)
- Special intersection types (chamfer_intersection)
- Shape manipulation (engrave, extrusion, revolution)
- Repetition (repeat, repeat_limited)
- Advanced blending (weight_blend)

All operations in this module are for direct function calls only.
"""

import fidgetpy as fp
import fidgetpy.math as fpm

def smooth_step_union(a, b, r):
    """
    Union of two SDFs with a smooth step transition of radius r.
    
    This operation creates a union of two shapes with a smooth transition
    between them using the smoothstep function. The transition has a different
    character than the polynomial smooth union.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        r: Transition radius (must be positive)
        
    Returns:
        An SDF expression representing the smooth step union
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If r is negative
        
    Examples:
        # Smooth step union of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.smooth_step_union(sphere, box, 0.3)
        
        # Create a smooth connection between shapes
        shape1 = fp.shape.cylinder(0.5, 1.0).translate(-1.0, 0.0, 0.0)
        shape2 = fp.shape.cylinder(0.5, 1.0).translate(1.0, 0.0, 0.0)
        connection = fpo.smooth_step_union(shape1, shape2, 0.5)
        
    IMPORTANT: Only for direct function calls: fpo.smooth_step_union(a, b, r)
    Method calls are not supported for operations.
    """
    e = fpm.smoothstep(0, r, a - b)
    return fpm.mix(a, b, e) - r * e * (1 - e)

def chamfer_union(a, b, r):
    """
    Union of two SDFs with a chamfered edge of radius r.
    
    This operation creates a union of two shapes with a flat chamfered
    edge where they meet, rather than a smooth blend. The chamfer creates
    a 45-degree angle cut at the intersection.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        r: Chamfer radius (must be positive)
        
    Returns:
        An SDF expression representing the chamfered union
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If r is negative
        
    Examples:
        # Chamfered union of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.chamfer_union(sphere, box, 0.2)
        
        # Create a mechanical-looking joint
        cyl1 = fp.shape.cylinder(0.5, 2.0).rotate_x(90)
        cyl2 = fp.shape.cylinder(0.5, 2.0).rotate_z(90)
        joint = fpo.chamfer_union(cyl1, cyl2, 0.3)
        
    IMPORTANT: Only for direct function calls: fpo.chamfer_union(a, b, r)
    Method calls are not supported for operations.
    """
    return fpm.min(fpm.min(a, b), (a + b) * 0.7071067811865475 - r)

def chamfer_intersection(a, b, r):
    """
    Intersection of two SDFs with a chamfered edge of radius r.
    
    This operation creates an intersection of two shapes with a flat chamfered
    edge where they meet, rather than a smooth blend. The chamfer creates
    a 45-degree angle cut at the intersection.
    
    Args:
        a: First SDF expression
        b: Second SDF expression
        r: Chamfer radius (must be positive)
        
    Returns:
        An SDF expression representing the chamfered intersection
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If r is negative
        
    Examples:
        # Chamfered intersection of a sphere and a box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.chamfer_intersection(sphere, box, 0.2)
        
        # Create a mechanical-looking cutout
        base = fp.shape.box(2.0, 2.0, 2.0)
        cutter = fp.shape.cylinder(1.0, 3.0)
        cutout = fpo.chamfer_intersection(base, cutter, 0.3)
        
    IMPORTANT: Only for direct function calls: fpo.chamfer_intersection(a, b, r)
    Method calls are not supported for operations.
    """
    return fpm.max(fpm.max(a, b), (a + b) * 0.7071067811865475 + r)

def engrave(base, engraving, depth):
    """
    Engraves one SDF into another with a given depth.
    
    This operation creates an engraving or embossing effect by cutting one
    shape into another to a specified depth. The engraving shape is used
    as a negative space in the base shape.
    
    Args:
        base: The base SDF expression (the object to engrave into)
        engraving: The SDF expression to engrave (the pattern or shape to cut)
        depth: The engraving depth (must be positive)
        
    Returns:
        An SDF expression representing the engraved shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If depth is negative
        
    Examples:
        # Engrave text into a box
        box = fp.shape.box(2.0, 1.0, 0.5)
        text = fp.shape.text("HELLO", 0.5)  # Assuming a text shape function
        result = fpo.engrave(box, text, 0.1)  # Engraves "HELLO" 0.1 units deep
        
        # Create a coin with an embossed design
        coin = fp.shape.cylinder(1.0, 0.2)
        design = fp.shape.sphere(0.7).translate(0.0, 0.0, 0.1)
        embossed_coin = fpo.engrave(coin, design, 0.05)
        
    IMPORTANT: Only for direct function calls: fpo.engrave(base, engraving, depth)
    Method calls are not supported for operations.
    """
    return fpm.max(base, -engraving + depth)

def extrusion(sdf_2d, height):
    """
    Extrudes a 2D SDF along the Z axis.
    
    This operation takes a 2D shape defined in the XY plane and extends it
    along the Z axis to create a 3D shape with a constant cross-section.
    The extrusion is centered at z=0.
    
    Args:
        sdf_2d: A 2D SDF expression (XY plane)
        height: The total height to extrude (centered at z=0, must be positive)
        
    Returns:
        An SDF expression representing the extruded 3D shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If height is not positive
        
    Examples:
        # Extrude a 2D circle to create a cylinder
        circle = fp.shape.circle(1.0)  # 2D circle in XY plane
        cylinder = fpo.extrusion(circle, 2.0)  # Extrude to height 2.0
        
        # Create a 3D text shape
        text = fp.shape.text_2d("FIDGET", 0.5)  # Assuming a 2D text shape function
        text_3d = fpo.extrusion(text, 0.2)  # Extrude to create 3D text
        
    IMPORTANT: Only for direct function calls: fpo.extrusion(sdf_2d, height)
    Method calls are not supported for operations.
    """
    # Convert height which is total height to half-height for the range [-h/2, h/2]
    half_height = height * 0.5
    
    # Compute the distance in the Z direction
    z_dist = fpm.abs(fp.z()) - half_height
    
    # Combine the 2D SDF with the Z distance
    # If inside the extrusion height range, use the 2D SDF
    # If outside, calculate the distance to the closest end cap
    return fpm.max(sdf_2d, z_dist)

def revolution(sdf_2d, axis_distance=0.0):
    """
    Creates a solid of revolution by rotating a 2D SDF around the Y axis.
    
    This operation takes a 2D shape defined in the XY plane and rotates it
    around the Y axis to create a 3D shape with rotational symmetry. The
    axis_distance parameter allows creating shapes like tori by offsetting
    the rotation axis.
    
    Args:
        sdf_2d: A 2D SDF expression in the XY plane to be revolved
        axis_distance: Distance from the rotation axis (default 0 = revolve around Y axis)
                      Must be non-negative
        
    Returns:
        An SDF expression representing the revolved 3D shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If axis_distance is negative
        
    Examples:
        # Create a sphere by revolving a semicircle
        semicircle = fp.shape.circle(1.0)  # 2D circle in XY plane
        sphere = fpo.revolution(semicircle)  # Revolve around Y axis
        
        # Create a torus by revolving a circle offset from the axis
        circle = fp.shape.circle(0.25).translate(1.0, 0.0, 0.0)  # Circle offset in X
        torus = fpo.revolution(circle)  # Creates a torus with major radius 1.0, minor radius 0.25
        
    IMPORTANT: Only for direct function calls: fpo.revolution(sdf_2d, axis_distance)
    Method calls are not supported for operations.
    """
    # Calculate the distance from the Y axis in the XZ plane
    x, y, z = fp.x(), fp.y(), fp.z()
    radius = fpm.length([x, z]) - axis_distance
    
    # Create a 2D point (radius, y) to use with the 2D SDF
    zero_sdf = 0.0
    return fpm.remap_xyz(sdf_2d, radius, y, zero_sdf)

def repeat(sdf, period):
    """
    Creates a periodic repetition of an SDF along all axes.
    
    This operation creates an infinite grid of repeated instances of the
    input shape. The period parameter controls the spacing between repetitions
    along each axis.
    
    Args:
        sdf: The SDF expression to repeat
        period: The repetition period as (x, y, z) tuple/vector
                All components must be positive
        
    Returns:
        An SDF expression representing the repeated shape
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If any period component is not positive
        
    Examples:
        # Create an infinite grid of spheres
        sphere = fp.shape.sphere(0.5)
        grid = fpo.repeat(sphere, (2.0, 2.0, 2.0))  # Spheres every 2 units in each direction
        
        # Create a row of cylinders along the X axis
        cylinder = fp.shape.cylinder(0.5, 1.0)
        row = fpo.repeat(cylinder, (3.0, 1000.0, 1000.0))  # Effectively only repeats in X
        
    IMPORTANT: Only for direct function calls: fpo.repeat(sdf, period)
    Method calls are not supported for operations.
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate the modulo for each coordinate to create repetition
    x_mod = fpm.mod(x, period[0]) - period[0] * 0.5
    y_mod = fpm.mod(y, period[1]) - period[1] * 0.5
    z_mod = fpm.mod(z, period[2]) - period[2] * 0.5
    
    # Remap the SDF with the modulated coordinates
    return fpm.remap_xyz(sdf, x_mod, y_mod, z_mod)

def repeat_limited(sdf, period, repetitions):
    """
    Creates a limited periodic repetition of an SDF.
    
    This operation creates a finite grid of repeated instances of the
    input shape. The period parameter controls the spacing between repetitions,
    and the repetitions parameter controls how many copies to create along each axis.
    
    Args:
        sdf: The SDF expression to repeat
        period: The repetition period as (x, y, z) tuple/vector
                All components must be positive
        repetitions: Number of repetitions as (x, y, z) tuple/vector
                    All components must be positive integers
        
    Returns:
        An SDF expression representing the repeated shape with limits
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If any period or repetition component is not positive
        
    Examples:
        # Create a 3x3x3 grid of spheres
        sphere = fp.shape.sphere(0.5)
        grid = fpo.repeat_limited(sphere, (2.0, 2.0, 2.0), (3, 3, 3))
        
        # Create a 5x1x1 row of boxes along the X axis
        box = fp.shape.box(0.5, 0.5, 0.5)
        row = fpo.repeat_limited(box, (1.5, 0.0, 0.0), (5, 1, 1))
        
    IMPORTANT: Only for direct function calls: fpo.repeat_limited(sdf, period, repetitions)
    Method calls are not supported for operations.
    """
    x, y, z = fp.x(), fp.y(), fp.z()
    
    # Calculate half-counts for centering
    half_rep_x = repetitions[0] * 0.5
    half_rep_y = repetitions[1] * 0.5
    half_rep_z = repetitions[2] * 0.5
    
    # Calculate the clamped coordinates for limited repetition
    x_cell = fpm.floor(fpm.clamp((x / period[0]) + half_rep_x, 0, repetitions[0] - 1) - half_rep_x)
    y_cell = fpm.floor(fpm.clamp((y / period[1]) + half_rep_y, 0, repetitions[1] - 1) - half_rep_y)
    z_cell = fpm.floor(fpm.clamp((z / period[2]) + half_rep_z, 0, repetitions[2] - 1) - half_rep_z)
    
    # Calculate the modulated coordinates
    x_mod = x - x_cell * period[0]
    y_mod = y - y_cell * period[1]
    z_mod = z - z_cell * period[2]
    
    # Remap the SDF with the modulated coordinates
    return fpm.remap_xyz(sdf, x_mod, y_mod, z_mod)

def weight_blend(sdfs, weights):
    """
    Blends multiple SDFs based on weights.
    
    This operation creates a weighted average of multiple shapes. Each shape
    is multiplied by its corresponding weight, and the result is normalized
    by the sum of all weights.
    
    Args:
        sdfs: List of SDF expressions to blend
        weights: List of weights for each SDF (must be same length as sdfs)
                All weights should be non-negative
        
    Returns:
        An SDF expression representing the weighted blend
        
    Raises:
        TypeError: If inputs are not SDF expressions or numeric types
        ValueError: If lengths of sdfs and weights don't match
        ValueError: If any weight is negative
        
    Examples:
        # Blend three shapes with equal weights
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        cylinder = fp.shape.cylinder(0.5, 2.0)
        result = fpo.weight_blend([sphere, box, cylinder], [1.0, 1.0, 1.0])
        
        # Create a shape that's 70% sphere and 30% box
        sphere = fp.shape.sphere(1.0)
        box = fp.shape.box(1.0, 1.0, 1.0)
        result = fpo.weight_blend([sphere, box], [0.7, 0.3])
        
    IMPORTANT: Only for direct function calls: fpo.weight_blend(sdfs, weights)
    Method calls are not supported for operations.
    """
    # Start with the first SDF * its weight
    result = sdfs[0] * weights[0]
    
    # Add each weighted SDF
    weight_sum = weights[0]
    for i in range(1, len(sdfs)):
        result = result + sdfs[i] * weights[i]
        weight_sum = weight_sum + weights[i]
    
    # Normalize by the sum of weights
    return result / weight_sum