"""
Tests for the fidgetpy.shape module.

This package contains test files for all shape categories in fidgetpy.shape:
- test_primitives.py: Tests for basic shape primitives
- test_rounded_shapes.py: Tests for shapes with rounded features
- test_cylinders.py: Tests for cylinder-based shapes
- test_curves.py: Tests for curve-based shapes
- test_specialized.py: Tests for more specialized shapes

All tests follow a similar pattern - they create shapes using fidgetpy,
mesh them in two different ways (using fidgetpy's internal meshing and using
fidget-cli with VM exports), and compare the resulting STL files.
"""