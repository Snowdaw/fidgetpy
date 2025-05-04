"""
Tests for the ops module operations.

This module contains unit tests for the various operations in the fidgetpy.ops module.
"""

import unittest
import numpy as np
import fidgetpy as fp
import fidgetpy.math as fpm
import fidgetpy.ops as fpo

class TestBooleanOps(unittest.TestCase):
    """Test boolean operations"""
    
    def setUp(self):
        """Set up common test shapes"""
        # Create two basic shapes for testing
        x, y, z = fp.x(), fp.y(), fp.z()
        p = [x, y, z]
        self.p = p
        
        # Define a sphere and a cube
        self.sphere = fpm.length(p) - 1.0
        self.cube = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
        
        # Define test points
        self.points = np.array([
            [0.0, 0.0, 0.0],    # Inside both
            [0.9, 0.0, 0.0],    # Inside sphere, inside cube
            [1.1, 0.0, 0.0],    # Outside sphere, inside cube
            [0.0, 0.9, 0.0],    # Inside sphere, inside cube
            [0.0, 0.0, 0.9],    # Inside sphere, inside cube
            [0.9, 0.9, 0.0],    # Inside sphere, inside cube
            [1.1, 1.1, 0.0],    # Outside sphere, outside cube
        ], dtype=np.float32)
        
    def test_union(self):
        """Test union operation"""
        union_sdf = fpo.union(self.sphere, self.cube)
        result = fp.eval(union_sdf, self.points, self.p)
        
        # Union should be the minimum of the two SDFs
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        cube_values = fp.eval(self.cube, self.points, self.p)
        expected = np.minimum(sphere_values, cube_values)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_intersection(self):
        """Test intersection operation"""
        intersection_sdf = fpo.intersection(self.sphere, self.cube)
        result = fp.eval(intersection_sdf, self.points, self.p)
        
        # Intersection should be the maximum of the two SDFs
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        cube_values = fp.eval(self.cube, self.points, self.p)
        expected = np.maximum(sphere_values, cube_values)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_difference(self):
        """Test difference operation"""
        difference_sdf = fpo.difference(self.sphere, self.cube)
        result = fp.eval(difference_sdf, self.points, self.p)
        
        # Difference should be max(a, -b)
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        cube_values = fp.eval(self.cube, self.points, self.p)
        expected = np.maximum(sphere_values, -cube_values)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_smooth_union(self):
        """Test smooth union operation"""
        # Just verify it runs without errors; exact values depend on implementation
        smooth_union_sdf = fpo.smooth_union(self.sphere, self.cube, 0.2)
        result = fp.eval(smooth_union_sdf, self.points, self.p)
        self.assertEqual(len(result), len(self.points))
        
    def test_complement(self):
        """Test complement operation"""
        complement_sdf = fpo.complement(self.sphere)
        result = fp.eval(complement_sdf, self.points, self.p)
        
        # Complement should negate the SDF
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        expected = -sphere_values
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestBlendingOps(unittest.TestCase):
    """Test blending operations"""
    
    def setUp(self):
        """Set up common test shapes"""
        # Create two basic shapes for testing
        x, y, z = fp.x(), fp.y(), fp.z()
        p = [x, y, z]
        self.p = p
        
        # Define a sphere and a cube
        self.sphere = fpm.length([x, y, z]) - 1.0
        self.cube = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
        
        # Define test points
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ], dtype=np.float32)
    
    def test_blend(self):
        """Test blend operation"""
        blend_sdf = fpo.blend(self.sphere, self.cube, 0.5)
        result = fp.eval(blend_sdf, self.points, self.p)
        
        # Blend should be a linear interpolation
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        cube_values = fp.eval(self.cube, self.points, self.p)
        expected = sphere_values * 0.5 + cube_values * 0.5
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_smooth_min(self):
        """Test smooth min operation"""
        # Just verify it runs without errors
        smooth_min_sdf = fpo.smooth_min(self.sphere, self.cube, 0.2)
        result = fp.eval(smooth_min_sdf, self.points, self.p)
        self.assertEqual(len(result), len(self.points))
        
    def test_exponential_smooth_min(self):
        """Test exponential smooth min operation"""
        # Just verify it runs without errors
        exp_smooth_min_sdf = fpo.exponential_smooth_min(self.sphere, self.cube, 0.2)
        result = fp.eval(exp_smooth_min_sdf, self.points, self.p)
        self.assertEqual(len(result), len(self.points))
        
    def test_power_smooth_min(self):
        """Test power smooth min operation"""
        # Just verify it runs without errors
        power_smooth_min_sdf = fpo.power_smooth_min(self.sphere, self.cube, 8.0)
        result = fp.eval(power_smooth_min_sdf, self.points, self.p)
        self.assertEqual(len(result), len(self.points))


class TestDomainOps(unittest.TestCase):
    """Test domain operations"""
    
    def setUp(self):
        """Set up common test shapes"""
        # Create a basic shape for testing
        x, y, z = fp.x(), fp.y(), fp.z()
        p = [x, y, z]
        self.p = p

        # Define a sphere
        self.sphere = fpm.length([x, y, z]) - 1.0
        
        # Define test points
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    
    def test_onion(self):
        """Test onion operation"""
        thickness = 0.1
        onion_sdf = fpo.onion(self.sphere, thickness)
        result = fp.eval(onion_sdf, self.points, self.p)
        
        # Onion should be the absolute value minus thickness
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        expected = np.abs(sphere_values) - thickness
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_mirror_x(self):
        """Test mirror_x operation"""
        mirrored_sdf = fpo.mirror_x(self.sphere)
        
        # Create points with negative x to test mirroring
        mirror_test_points = np.array([
            [-0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2]  # Same but with positive x
        ], dtype=np.float32)
        
        result = fp.eval(mirrored_sdf, mirror_test_points, self.p)
        
        # Both points should give the same result after mirroring
        self.assertAlmostEqual(result[0], result[1], places=5)


class TestMiscOps(unittest.TestCase):
    """Test miscellaneous operations"""
    
    def setUp(self):
        """Set up common test shapes"""
        # Create two basic shapes for testing
        x, y, z = fp.x(), fp.y(), fp.z()
        p = [x, y, z]
        self.p = p
        
        # Define a sphere and a cube
        self.sphere = fpm.length([x, y, z]) - 1.0
        self.cube = fpm.max(fpm.max(fpm.abs(x), fpm.abs(y)), fpm.abs(z)) - 0.8
        
        # Define test points
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ], dtype=np.float32)
        
    def test_chamfer_union(self):
        """Test chamfer union operation"""
        # Just verify it runs without errors
        chamfer_union_sdf = fpo.chamfer_union(self.sphere, self.cube, 0.2)
        result = fp.eval(chamfer_union_sdf, self.points, self.p)
        self.assertEqual(len(result), len(self.points))
        
    def test_engrave(self):
        """Test engrave operation"""
        depth = 0.1
        engrave_sdf = fpo.engrave(self.sphere, self.cube, depth)
        result = fp.eval(engrave_sdf, self.points, self.p)
        
        # Engrave should be max(base, -engraving + depth)
        sphere_values = fp.eval(self.sphere, self.points, self.p)
        cube_values = fp.eval(self.cube, self.points, self.p)
        expected = np.maximum(sphere_values, -cube_values + depth)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_extrusion(self):
        """Test extrusion operation"""
        # Define a 2D circle SDF
        x, y = fp.x(), fp.y()
        circle = fpm.length([x, y]) - 0.8
        
        # Create a cylinder by extruding the circle
        height = 2.0
        extrusion_sdf = fpo.extrusion(circle, height)
        
        # Test points along the z-axis
        test_points = np.array([
            [0.0, 0.0, 0.0],   # Center of cylinder
            [0.0, 0.0, 0.9],   # Near the top cap, inside
            [0.0, 0.0, 1.1],   # Beyond the top cap, outside
            [0.7, 0.0, 0.0],   # Inside the circle, midway
            [0.9, 0.0, 0.0],   # Outside the circle, midway
        ], dtype=np.float32)
        
        result = fp.eval(extrusion_sdf, test_points, self.p)
        
        # Verify points inside and outside are correctly identified
        self.assertTrue(result[0] < 0)  # Center should be inside
        self.assertTrue(result[1] < 0)  # Near top but still inside
        self.assertTrue(result[2] > 0)  # Above top, should be outside
        self.assertTrue(result[3] < 0)  # Inside circle radius
        self.assertTrue(result[4] > 0)  # Outside circle radius


if __name__ == '__main__':
    unittest.main()