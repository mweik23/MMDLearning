# tests/test_global_var_single_cpu.py
import torch
from pathlib import Path
import sys
SRC_path = Path(__file__).parents[1].resolve() / 'src' / 'MMDLearning'
sys.path.append(str(SRC_path))

from utils.distributed import (
    dist_global_variance_autograd,
    dist_global_variance_nograd,
)

def main():
    torch.manual_seed(0)
    x = torch.randn(17, 5, dtype=torch.float32)  # CPU
    #print(x)

    # Our functions should work when dist is not initialized
    v_auto = dist_global_variance_autograd(x, unbiased=True)
    v_eval = dist_global_variance_nograd(x, unbiased=True)

    # Reference: per-feature unbiased var, then sum over features (scalar)
    ref = x.var(dim=0, unbiased=True).sum()

    print("autograd:", float(v_auto), "nograd:", float(v_eval), "ref:", float(ref))
    assert torch.allclose(v_auto, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_eval, ref, atol=1e-6, rtol=1e-6)
    print("Test passed.")
    
if __name__ == "__main__":
    main()
