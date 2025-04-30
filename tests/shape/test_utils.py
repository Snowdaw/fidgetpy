"""
Utility functions for shape module tests.

This module provides common functionality for testing shapes,
including meshing, STL file comparison, and running fidget-cli.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import fidgetpy as fp
import shutil
import pytest
import struct
import numpy as np

# Global flag to control cleanup. Set to False to keep output files after tests.
CLEANUP_TEST_OUTPUT = True
# Directory for persistent test output when cleanup is disabled
PERSISTENT_OUTPUT_DIR = Path("test_output")


def parse_stl_file(stl_path):
    """
    Parse an STL file and return the number of vertices and triangles.
    
    Args:
        stl_path: Path to the STL file
        
    Returns:
        A tuple (num_vertices, num_triangles) or (0, 0) if parsing fails
    """
    try:
        with open(stl_path, 'rb') as f:
            # Skip header
            f.seek(80)
            
            # Read number of triangles (4 bytes)
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            # Each triangle is 50 bytes (12 floats + 1 short)
            # We'll read a sample to verify it's a valid STL file
            if num_triangles > 0:
                # Read the first triangle to verify format
                try:
                    # Normal vector (3 floats)
                    nx, ny, nz = struct.unpack('<fff', f.read(12))
                    
                    # Vertex 1 (3 floats)
                    v1x, v1y, v1z = struct.unpack('<fff', f.read(12))
                    
                    # Vertex 2 (3 floats)
                    v2x, v2y, v2z = struct.unpack('<fff', f.read(12))
                    
                    # Vertex 3 (3 floats)
                    v3x, v3y, v3z = struct.unpack('<fff', f.read(12))
                    
                    # Attribute byte count (1 short)
                    _ = struct.unpack('<H', f.read(2))[0]
                    
                    # If we got here, it's likely a valid STL file
                    # In a real STL parser, we'd collect unique vertices
                    # For simplicity, we'll estimate vertices as 3 per triangle
                    # This is an overestimate since vertices are shared
                    num_vertices = num_triangles * 3
                    
                    return num_vertices, num_triangles
                except:
                    print(f"  ERROR: Failed to parse triangle data in {stl_path}")
                    return 0, 0
            else:
                print(f"  WARNING: STL file {stl_path} has 0 triangles")
                return 0, 0
    except Exception as e:
        print(f"  ERROR: Failed to parse STL file {stl_path}: {str(e)}")
        return 0, 0


def find_fidget_cli():
    """
    Find the fidget-cli executable.
    
    Looks in common locations for the fidget-cli executable (debug or release).
    Returns the path to the executable.
    Raises FileNotFoundError if the executable is not found.
    """
    possible_paths = [
        # Debug build
        Path("../fidget/target/debug/fidget-cli"),
        Path("../../fidget/target/debug/fidget-cli"),
        # Release build
        Path("../fidget/target/release/fidget-cli"),
        Path("../../fidget/target/release/fidget-cli"),
        # Relative to current directory
        Path("fidget/target/debug/fidget-cli"),
        Path("fidget/target/release/fidget-cli"),
        # Check if it's in PATH
        shutil.which("fidget-cli")
    ]
    
    # Print current directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print("Searching for fidget-cli in:")
    
    for path in possible_paths:
        if path is not None:
            abs_path = Path(path).resolve() if isinstance(path, Path) else Path(path)
            print(f"  - {abs_path} {'(FOUND)' if os.path.exists(abs_path) else '(NOT FOUND)'}")
            if os.path.exists(abs_path):
                print(f"Found fidget-cli at: {abs_path}")
                return abs_path
    
    # If no executable was found, raise an error
    raise FileNotFoundError(
        "fidget-cli executable not found. Make sure it's built and available in "
        "fidget/target/debug, fidget/target/release, or in your PATH."
    )


def mesh_with_fidget_cli(vm_path, stl_path, depth=5, center="0,0,0", scale=1.0):
    """
    Mesh a VM file using fidget-cli.
    
    Args:
        vm_path: Path to the VM file
        stl_path: Path where the STL file will be saved
        depth: Octree depth for meshing
        center: Center point for meshing (comma-separated string)
        scale: Scale factor for meshing
    
    Returns:
        True if meshing was successful, False otherwise
    
    Raises:
        FileNotFoundError: If fidget-cli executable is not found
        subprocess.CalledProcessError: If the meshing process fails
    """
    # Find fidget-cli executable (will raise FileNotFoundError if not found)
    fidget_cli_path = find_fidget_cli()
    
    cmd = [
        str(fidget_cli_path),
        "mesh",
        "--input", str(vm_path),
        "--depth", str(depth),
        "--center", center,
        "--scale", str(scale),
        "--out", str(stl_path)
    ]
    
    print(f"Running fidget-cli command: {' '.join(cmd)}")
    
    try:
        # Run the command with output captured
        result = subprocess.run(cmd, check=True, capture_output=True)
        print(f"fidget-cli mesh command succeeded")
        
        # Check if the output file was created properly
        if os.path.exists(str(stl_path)):
            size = os.path.getsize(str(stl_path))
            print(f"Generated STL file size: {size} bytes")
            if size < 100:  # Too small for a proper mesh
                print(f"WARNING: Generated STL file is suspiciously small ({size} bytes)")
        else:
            print(f"WARNING: STL file was not created at {stl_path}")
            
        return True
    except subprocess.CalledProcessError as e:
        # Print error details for debugging
        print(f"fidget-cli mesh command failed with exit code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        if e.stdout:
            print(f"Output: {e.stdout.decode('utf-8')}")
        if e.stderr:
            print(f"Error: {e.stderr.decode('utf-8')}")
        return False


def compare_stl_files(stl_path1, stl_path2, tolerance=0.0):
    """
    Compare two STL files to check if they represent the same mesh.
    
    This comparison checks:
    1. If both files exist and are valid STL files
    2. If both meshes have vertices and triangles
    3. If the file sizes are roughly similar
    
    Args:
        stl_path1: Path to the first STL file
        stl_path2: Path to the second STL file
        tolerance: Relative tolerance for file size comparison
    
    Returns:
        True if the files are similar, False otherwise
    """
    print(f"\nComparing STL files:")
    print(f"  - {stl_path1}")
    print(f"  - {stl_path2}")
    
    # Check if files exist
    stl_path1_str = str(stl_path1)
    stl_path2_str = str(stl_path2)
    if not os.path.exists(stl_path1_str):
        print(f"ERROR: First STL file does not exist: {stl_path1_str}")
        return False
    
    if not os.path.exists(stl_path2_str):
        print(f"ERROR: Second STL file does not exist: {stl_path2_str}")
        return False
    
    # Compare file sizes as a basic check
    size1 = os.path.getsize(stl_path1_str)
    size2 = os.path.getsize(stl_path2_str)
    
    print(f"  Python STL size: {size1} bytes")
    print(f"  CLI STL size: {size2} bytes")
    
    # If files are very small, they might be empty or corrupt
    if size1 < 100:
        print(f"ERROR: First STL file is too small ({size1} bytes), likely empty or corrupt")
        return False
    
    if size2 < 100:
        print(f"ERROR: Second STL file is too small ({size2} bytes), likely empty or corrupt")
        return False
    
    # Parse STL files to get vertex and triangle counts
    vertices1, triangles1 = parse_stl_file(stl_path1_str)
    vertices2, triangles2 = parse_stl_file(stl_path2_str)
    
    print(f"  Python mesh: {vertices1} vertices, {triangles1} triangles")
    print(f"  CLI mesh: {vertices2} vertices, {triangles2} triangles")
    
    # Check if both meshes have vertices and triangles
    if vertices1 == 0 or triangles1 == 0:
        print(f"ERROR: Python mesh has no geometry (vertices: {vertices1}, triangles: {triangles1})")
        return False
    
    if vertices2 == 0 or triangles2 == 0:
        print(f"ERROR: CLI mesh has no geometry (vertices: {vertices2}, triangles: {triangles2})")
        return False
    
    # Calculate relative difference in file size
    relative_diff = abs(size1 - size2) / max(size1, size2)
    print(f"  Relative difference in size: {relative_diff:.2%}")
    
    # Calculate relative difference in triangle count
    triangle_diff = abs(triangles1 - triangles2) / max(triangles1, triangles2)
    print(f"  Relative difference in triangle count: {triangle_diff:.2%}")
    
    # Files are considered similar if their sizes and triangle counts are within the tolerance
    size_ok = relative_diff <= tolerance
    triangles_ok = triangle_diff <= tolerance * 2  # Allow more variance in triangle count
    
    result = size_ok and triangles_ok
    
    if result:
        print(f"  Result: Files are similar (within tolerance)")
    else:
        if not size_ok:
            print(f"  Result: Files differ significantly in size (exceeds {tolerance:.0%} tolerance)")
        if not triangles_ok:
            print(f"  Result: Files differ significantly in triangle count (exceeds {tolerance*2:.0%} tolerance)")
    
    return result


def shape_dual_meshing(shape, name, depth=5, scale=1.0, center=(0, 0, 0), save_output_if_no_cleanup=True):
    """
    Test a shape by meshing it with both fidgetpy and fidget-cli.

    Handles temporary file creation and optional persistent saving based on
    the global CLEANUP_TEST_OUTPUT flag.
    
    Args:
        shape: The fidgetpy shape to test
        name: Name to use for temp files
        depth: Octree depth for meshing
        scale: Scale factor for meshing
        center: Center point for meshing
        save_output_if_no_cleanup: If True and CLEANUP_TEST_OUTPUT is False,
                                   save output files to PERSISTENT_OUTPUT_DIR.

    Returns:
        A tuple (success, filepath_py, filepath_cli) where paths refer to
        temporary files if cleanup is enabled, or persistent files otherwise.
        - success is True if both meshes were created and are similar
        - filepath_py is the path to the STL created by fidgetpy
        - filepath_cli is the path to the STL created by fidget-cli
    """
    # Determine if persistent output should be saved
    should_save_persistently = not CLEANUP_TEST_OUTPUT and save_output_if_no_cleanup

    # Create persistent output directory if needed
    if should_save_persistently and not PERSISTENT_OUTPUT_DIR.exists():
        os.makedirs(PERSISTENT_OUTPUT_DIR)

    # Define persistent paths only if saving persistently
    persistent_py_path = PERSISTENT_OUTPUT_DIR / f"{name}_py.stl" if should_save_persistently else None
    persistent_cli_path = PERSISTENT_OUTPUT_DIR / f"{name}_cli.stl" if should_save_persistently else None
    persistent_vm_path = PERSISTENT_OUTPUT_DIR / f"{name}.vm" if should_save_persistently else None

    # Create temporary directory for all intermediate work
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Define paths within the temporary directory
        vm_path_tmp = tmp_dir_path / f"{name}.vm"
        stl_py_path_tmp = tmp_dir_path / f"{name}_py.stl"
        stl_cli_path_tmp = tmp_dir_path / f"{name}_cli.stl"

        # --- Step 1: Export shape to VM format ---
        vm_content = fp.to_vm(shape)
        with open(vm_path_tmp, "w") as f:
            f.write(vm_content)

        # Save VM file persistently if requested
        if should_save_persistently:
            shutil.copy(str(vm_path_tmp), str(persistent_vm_path))

        # --- Step 2: Mesh using fidgetpy ---
        center_list = list(center) # Convert tuple to list for center parameter
        mesh_py = fp.mesh(shape, scale=scale, depth=depth, center=center_list)

        # Check if Python mesh has geometry
        if len(mesh_py.vertices) == 0 or len(mesh_py.triangles) == 0:
            print(f"ERROR: Python mesh for {name} has no geometry (vertices: {len(mesh_py.vertices)}, triangles: {len(mesh_py.triangles)})")
            # Return temporary path if cleaning up, persistent otherwise
            py_path_to_return = str(persistent_py_path) if should_save_persistently else str(stl_py_path_tmp)
            return False, py_path_to_return, None

        fp.save_stl(mesh_py, str(stl_py_path_tmp))

        # Save Python-meshed STL persistently if requested
        if should_save_persistently:
            shutil.copy(str(stl_py_path_tmp), str(persistent_py_path))

        # --- Step 3: Mesh using fidget-cli ---
        center_str = f"{center[0]},{center[1]},{center[2]}"
        success_cli = mesh_with_fidget_cli(
            vm_path_tmp, stl_cli_path_tmp, depth=depth, center=center_str, scale=scale
        )

        if not success_cli:
            print(f"Failed to mesh {name} with fidget-cli")
            py_path_to_return = str(persistent_py_path) if should_save_persistently else str(stl_py_path_tmp)
            return False, py_path_to_return, None

        # Save CLI-meshed STL persistently if requested
        if should_save_persistently:
            shutil.copy(str(stl_cli_path_tmp), str(persistent_cli_path))

        # --- Step 4: Compare the STL files (using temporary paths) ---
        are_similar = compare_stl_files(stl_py_path_tmp, stl_cli_path_tmp)

        # Determine which paths to return based on cleanup flag
        py_path_final = str(persistent_py_path) if should_save_persistently else str(stl_py_path_tmp)
        cli_path_final = str(persistent_cli_path) if should_save_persistently else str(stl_cli_path_tmp)

        if are_similar:
            return True, py_path_final, cli_path_final
        else:
            print(f"STL files for {name} differ significantly")
            return False, py_path_final, cli_path_final

@pytest.fixture(scope="session", autouse=True)
def cleanup_persistent_output_folder(request):
    """
    Pytest fixture to clean up the persistent output folder after all tests
    in the session have run, based on the CLEANUP_TEST_OUTPUT flag.
    """
    # Let the tests run
    yield

    # Cleanup phase after tests
    if CLEANUP_TEST_OUTPUT:
        if PERSISTENT_OUTPUT_DIR.exists():
            print(f"\nCleaning up persistent output directory: {PERSISTENT_OUTPUT_DIR}")
            try:
                shutil.rmtree(PERSISTENT_OUTPUT_DIR)
                print(f"Successfully removed {PERSISTENT_OUTPUT_DIR}")
            except Exception as e:
                print(f"Error removing {PERSISTENT_OUTPUT_DIR}: {e}")
        else:
            print(f"\nPersistent output directory {PERSISTENT_OUTPUT_DIR} not found, no cleanup needed.")
    else:
        print(f"\nCleanup disabled (CLEANUP_TEST_OUTPUT=False), keeping {PERSISTENT_OUTPUT_DIR}")