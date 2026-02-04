#!/bin/bash
set -e

echo "=== GPU Availability Test ==="

# Test CUDA with PyTorch
uv run python -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
    
    # Simple GPU test
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print('✓ GPU computation test passed')
    except Exception as e:
        print(f'✗ GPU computation test failed: {e}')
        sys.exit(1)
else:
    print('✗ No GPU available')
    sys.exit(1)

print('=== Test Complete ===')
"

echo "Exit code: $?"