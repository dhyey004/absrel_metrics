import unittest
import torch
from absrel_metrics import compute_abs_rel

class TestAbsRelMetric(unittest.TestCase):
    
    def test_compute_abs_rel(self):
        # 1. Create fake predictions (e.g., 2.0 meters, 4.0 meters)
        pred = torch.tensor([[2.0, 4.0],
                             [5.0, 0.0]]) 
        
        # 2. Create fake ground truth targets 
        # The 0.0 represents missing LiDAR data (the mask)
        target = torch.tensor([[2.0, 5.0],
                               [4.0, 0.0]]) 

        # 3. Calculate expected math manually:
        # Pixel 1: pred=2.0, target=2.0 -> error = abs(2-2)/2 = 0.0
        # Pixel 2: pred=4.0, target=5.0 -> error = abs(4-5)/5 = 0.2
        # Pixel 3: pred=5.0, target=4.0 -> error = abs(5-4)/4 = 0.25
        # Pixel 4: target=0.0 -> ignored because it is masked out!
        # Expected Mean = (0.0 + 0.2 + 0.25) / 3 valid pixels = 0.15

        
        result = compute_abs_rel(pred, target, mask=0.0)

        # 5. Check if the function's result matches our manual math
        # We use AlmostEqual to account for tiny GPU floating-point variations
        self.assertAlmostEqual(result.item(), 0.15, places=4)

if __name__ == "__main__":
    unittest.main()