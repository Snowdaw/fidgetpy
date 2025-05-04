import os
import unittest
import fidgetpy as fp

class TestBearVMFRep(unittest.TestCase):
    
    def setUp(self):
        # Get path to the bear.vm file
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.bear_vm_path = os.path.join(repo_root, "fidget", "models", "bear.vm")
        
        # Skip tests if the bear.vm file doesn't exist
        if not os.path.exists(self.bear_vm_path):
            self.skipTest(f"bear.vm not found at expected location: {self.bear_vm_path}")
        
        # Load the bear.vm content
        with open(self.bear_vm_path, "r") as f:
            self.bear_vm_content = f.read()
    
    def test_bear_vm_to_frep(self):
        """Test converting the imported bear model to F-Rep format."""
        # Import the bear model
        bear_expr = fp.from_vm(self.bear_vm_content)
        
        # Convert to F-Rep
        frep = fp.to_frep(bear_expr)
        
        # Assert it's a valid string with content
        self.assertIsInstance(frep, str)
        self.assertGreater(len(frep), 0)
        
        # Check that there are no register references in the output
        self.assertNotIn("_", frep, "F-Rep output still contains register references")
        
        # Print the F-Rep representation for inspection
        print(f"Bear model F-Rep (first 200 chars): {frep[:200]}...")
    
    def test_vm_roundtrip_and_frep(self):
        """Test import VM, export VM, then convert to F-Rep."""
        # Import the bear model
        bear_expr = fp.from_vm(self.bear_vm_content)
        
        # Re-export to VM
        exported_vm = fp.to_vm(bear_expr)
        
        # Re-import the exported VM
        reimported_expr = fp.from_vm(exported_vm)
        
        # Convert to F-Rep
        frep = fp.to_frep(reimported_expr)
        
        # Assert it's a valid string with content
        self.assertIsInstance(frep, str)
        self.assertGreater(len(frep), 0)
        
        # Check that there are no register references in the output
        self.assertNotIn("_", frep, "F-Rep output still contains register references")
        
        # Print the F-Rep representation for inspection
        print(f"Re-imported bear model F-Rep (first 200 chars): {frep[:200]}...")

if __name__ == "__main__":
    unittest.main()