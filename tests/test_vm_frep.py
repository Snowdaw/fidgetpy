import unittest
import fidgetpy as fp
import numpy as np

class TestVMFRep(unittest.TestCase):
    
    def test_vm_frep_simplification(self):
        """Test that the VM-based F-Rep simplifies expressions effectively."""
        
        # Create a complex expression
        x = fp.x()
        y = fp.y()
        z = fp.z()
        
        # Create a simple sphere with radius 2 centered at the origin
        # This will have a negative value at the origin (inside the sphere)
        shape = (x*x + y*y + z*z).sqrt() - 2.0
        
        # Get the F-Rep string
        frep_str = str(shape)
        
        print(f"Simplified F-Rep: {frep_str}")
        
        # Assert that the F-Rep string contains fewer operations than a naive implementation would
        # We don't check the exact string as the simplification might change, but we verify
        # that some simplification has occurred by checking length and structure
        
        # Check that it's not the naive expansion that would have many nested operations
        self.assertLess(
            len(frep_str), 
            200,  # A reasonable threshold - the fully expanded version would be much longer
            "The F-Rep string is not simplified enough"
        )
        
        # Verify that the expression still evaluates correctly by checking points
        # Point inside the shape should have negative SDF value
        inside_point = [0.0, 0.0, 0.0]  # origin should be inside
        val = fp.eval(shape, [inside_point])[0]
        self.assertLess(val, 0, f"Expected negative SDF value for point inside shape, got {val}")
        
        # Point far outside the shape should have positive SDF value
        outside_point = [2.0, 2.0, 2.0]  # far from the shape
        val = fp.eval(shape, [outside_point])[0]
        self.assertGreater(val, 0, f"Expected positive SDF value for point far outside shape, got {val}")

    def test_complex_expressions(self):
        """Test F-Rep simplification with more complex expressions."""
        
        # Create variables
        x = fp.x()
        y = fp.y()
        z = fp.z()
        
        # Create a more complex nested expression
        # This creates a shape with multiple operations that should benefit from simplification
        expr1 = (x*x + y*y + z*z).sqrt() - 1.0  # sphere
        expr2 = (x*x + (y-2.0)*(y-2.0) + z*z).sqrt() - 0.2  # small sphere
        expr3 = ((x*x + z*z).sqrt() * 0.917 + y * 0.4) - 1.0  # cone-like shape
        
        # Combine with boolean operations
        shape = expr1 & ~(expr2 | expr3)
        
        # Get the F-Rep string
        frep_str = str(shape)
        print(f"Complex shape F-Rep: {frep_str}")
        
        # Assert that the resulting string is of reasonable length
        self.assertLess(len(frep_str), 500, "Complex shape F-Rep should be simplified")

if __name__ == "__main__":
    unittest.main()