import os
import sys
import subprocess
import time
import numpy as np

def build_extension():
    print("Building C++ extension...")
    try:
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
    except subprocess.CalledProcessError:
        print("Build failed!")
        sys.exit(1)

# Check if the module exists, if not, build it
if not os.path.exists("jepa_core.cpython-{}-x86_64-linux-gnu.so".format(sys.version_info.major * 10 + sys.version_info.minor)) and \
   not any(f.startswith("jepa_core") and f.endswith(".so") for f in os.listdir(".")):
    build_extension()

# Try importing
try:
    import jepa_core
except ImportError:
    # If import fails, try building again (maybe filename mismatch)
    build_extension()
    import jepa_core

def benchmark():
    # Parameters
    N = 4096   # Batch Size
    D = 2048   # Latent Dimension
    
    print(f"Initializing Tensors: Batch={N}, Dim={D}")
    
    # Ensure 64-byte alignment for best AVX performance (though our code handles unaligned)
    # NumPy usually allocates 16-byte aligned.
    # To be perfectly safe and mimic HPC environments:
    pred = np.random.rand(N, D).astype(np.float32)
    target = np.random.rand(N, D).astype(np.float32)
    
    # Pre-allocate result buffer
    result_cpp = np.zeros(N, dtype=np.float32)
    
    # Warmup
    jepa_core.compute_batch_l2_sq(pred, target, result_cpp)
    
    # --- NumPy Benchmark ---
    start = time.time()
    result_numpy = np.sum((pred - target)**2, axis=1)
    end = time.time()
    t_numpy = end - start
    print(f"NumPy Time: {t_numpy:.6f}s")
    
    # --- C++ Benchmark ---
    start = time.time()
    jepa_core.compute_batch_l2_sq(pred, target, result_cpp)
    end = time.time()
    t_cpp = end - start
    print(f"C++ Time:   {t_cpp:.6f}s")
    
    # --- Validation ---
    # Floating point arithmetic order differences lead to small epsilons
    diff = np.abs(result_numpy - result_cpp)
    max_diff = np.max(diff)
    
    print(f"Speedup: {t_numpy / t_cpp:.2f}x")
    print(f"Max Diff: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

if __name__ == "__main__":
    benchmark()
