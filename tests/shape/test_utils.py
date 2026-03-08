"""
Utility functions for shape module tests.

This module provides common functionality for testing shapes,
including meshing and PLY file output.
"""

import os
import tempfile
from pathlib import Path
import fidgetpy as fp
import shutil
import pytest


# Global flag to control cleanup. Set to False to keep output files after tests.
CLEANUP_TEST_OUTPUT = True
# Directory for persistent test output when cleanup is disabled
PERSISTENT_OUTPUT_DIR = Path("test_output")


def shape_dual_meshing(shape, name, depth=5, scale=1.0, center=(0, 0, 0), save_output_if_no_cleanup=True):
    """
    Test a shape by meshing it with fidgetpy and writing a PLY file.

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
        A tuple (success, filepath_py, None) where:
        - success is True if the mesh was created with geometry
        - filepath_py is the path to the PLY created by fidgetpy
        - None is a placeholder for the removed CLI comparison
    """
    should_save_persistently = not CLEANUP_TEST_OUTPUT and save_output_if_no_cleanup

    if should_save_persistently and not PERSISTENT_OUTPUT_DIR.exists():
        os.makedirs(PERSISTENT_OUTPUT_DIR)

    persistent_py_path = PERSISTENT_OUTPUT_DIR / f"{name}_py.ply" if should_save_persistently else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        ply_path_tmp = tmp_dir_path / f"{name}.ply"

        # Mesh using fidgetpy and write PLY
        center_list = list(center)
        fp.mesh(shape, output_file=str(ply_path_tmp), scale=scale, depth=depth,
                center=center_list, verbose=False)

        if not ply_path_tmp.exists() or ply_path_tmp.stat().st_size < 100:
            return False, str(ply_path_tmp), None

        if should_save_persistently:
            shutil.copy(str(ply_path_tmp), str(persistent_py_path))

        py_path_final = str(persistent_py_path) if should_save_persistently else str(ply_path_tmp)
        return True, py_path_final, None


@pytest.fixture(scope="session", autouse=True)
def cleanup_persistent_output_folder(request):
    """
    Pytest fixture to clean up the persistent output folder after all tests
    in the session have run, based on the CLEANUP_TEST_OUTPUT flag.
    """
    yield

    if CLEANUP_TEST_OUTPUT:
        if PERSISTENT_OUTPUT_DIR.exists():
            try:
                shutil.rmtree(PERSISTENT_OUTPUT_DIR)
            except Exception:
                pass
