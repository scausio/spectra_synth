import os
import time

print("=== ENVIRONMENT ===")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print()

try:
    import torch
except ImportError:
    print("PyTorch not installed")
    exit(1)

print("=== PYTORCH INFO ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("GPU NOT USED: running on CPU only")
    exit(0)

device = torch.device("cuda:0")
print("Using device:", torch.cuda.get_device_name(device))
print()

print("=== GPU LOAD TEST ===")
# allocate something non-trivial
x = torch.randn(8000, 8000, device=device)
y = x @ x

# keep GPU busy a bit
torch.cuda.synchronize()
time.sleep(5)

print("GPU IS BEING USED")
print("Tensor device:", x.device)
print("Allocated memory (MB):",
      torch.cuda.memory_allocated(device) / 1024**2)
