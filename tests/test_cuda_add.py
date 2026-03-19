"""
uv run python tests/test_cuda_add.py
"""

import torch  # noqa: I001
import openpi_cuda_lib as cu

# Test the renamed module
print("Testing openpi_cuda module...")

# Create test tensor
input_tensor = torch.ones(10, device="cuda")
value_to_add = 5.0

# Call the CUDA extension
output = cu.add(input_tensor, value_to_add)

# Verify result
expected = torch.ones(10, device="cuda") + 5.0
success = torch.allclose(output, expected)

print(f"Input: {input_tensor.cpu().numpy()}")
print(f"Output: {output.cpu().numpy()}")
print(f"Expected: {expected.cpu().numpy()}")
print("✓ Test passed!" if success else "✗ Test failed!")

assert success, "Module test failed!"
