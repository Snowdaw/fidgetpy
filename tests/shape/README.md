# Fidget Python Shape Module Tests

This directory contains tests for the Fidget Python shape module. The tests create various shapes using fidgetpy, mesh them using two different methods, and compare the results:

1. Using fidgetpy's built-in meshing functionality with `fp.mesh()` and saving as STL with `fp.save_stl()`
2. Exporting the shape as a VM file, then meshing it using the fidget-cli tool and saving as STL

## Test Files

The tests are organized by shape category:

- `test_primitives.py`: Tests for basic shape primitives (sphere, box, plane, etc.)
- `test_rounded_shapes.py`: Tests for shapes with rounded features (rounded_box, rounded_cylinder, etc.)
- `test_cylinders.py`: Tests for cylinder-based shapes (cylinder, capsule, cone, etc.)
- `test_curves.py`: Tests for curve-based shapes (line_segment, bezier curves, etc.)
- `test_specialized.py`: Tests for more specialized shapes (ellipsoid, box_frame, death_star, etc.)
- `test_utils.py`: Utility functions used by the tests

## Requirements

To run these tests, you need:

1. A working fidgetpy installation
2. The fidget-cli binary (either in fidget/target/debug or fidget/target/release)
3. pytest

## Running the Tests

To run all shape tests:

```bash
cd fidget-py
python -m pytest -v tests/shape
```

To run tests for a specific shape category:

```bash
# For primitives
python -m pytest -v tests/shape/test_primitives.py

# For rounded shapes
python -m pytest -v tests/shape/test_rounded_shapes.py

# For cylinder shapes
python -m pytest -v tests/shape/test_cylinders.py

# For curve shapes
python -m pytest -v tests/shape/test_curves.py

# For specialized shapes
python -m pytest -v tests/shape/test_specialized.py
```

To run a specific test:

```bash
python -m pytest -v tests/shape/test_primitives.py::test_sphere
```

## Test Results

Each test creates two STL files - one meshed with fidgetpy and one meshed with fidget-cli. These files are initially created in a temporary directory and then, if the test passes, copied to the current directory with names like:

- `sphere_py.stl` (for the fidgetpy-meshed version)
- `sphere_cli.stl` (for the fidget-cli-meshed version)

If a test fails, the STL files will remain in the temporary directory (path will be shown in the error message).

## Visualizing the Results

You can open these STL files in Blender or another 3D viewer to manually inspect and compare them.

## Cleaning Up

The test script automatically cleans up the STL files when tests pass. If you want to keep them for inspection, modify the test files to remove the `unlink()` calls.