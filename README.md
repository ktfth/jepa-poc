# JEPA (Joint-Embedding Predictive Architecture) - HPC Proof of Concept

This repository contains a Proof of Concept (POC) implementation of JEPA, evolving from basic Python prototypes to a high-performance hybrid C++/Python architecture utilizing AVX SIMD instructions and OpenMP multithreading.

## Project Structure

*   **`jepa_core.cpp`**: (New) The high-performance C++ core. Implements optimized tensor operations (like Batched L2 Distance) using AVX intrinsics, OpenMP, and Zero-Copy PyBind11 integration.
*   **`setup.py`**: Build script to compile the C++ core into a Python extension.
*   **`benchmark.py`**: Script to verify correctness and benchmark the performance gap between NumPy and the optimized C++ core.
*   **`v0.py`**: Initial Python prototype.
*   **`v1-sementic-search.py`**: Python implementation focusing on semantic search capabilities.
*   **`v2-temporal-patterns.cpp`**: Standalone C++ implementation exploring temporal patterns.

## Prerequisites

*   **Python 3.8+**
*   **C++ Compiler**: GCC (Linux) or MSVC (Windows) with C++14 support.
*   **Dependencies**:
    *   `numpy`
    *   `pybind11`
    *   `setuptools`

## Installation & Compilation

To utilize the optimized `jepa_core` in Python, you must compile the extension.

### Linux / macOS

Ensure you have `g++` (or `clang++`) and OpenMP installed (`libomp-dev` on some systems).

1.  Install Python dependencies:
    ```bash
    pip install numpy pybind11
    ```

2.  Compile the extension in-place:
    ```bash
    python3 setup.py build_ext --inplace
    ```
    *Flags used:* `-O3`, `-march=native`, `-fopenmp`.

### Windows

Ensure you have Visual Studio (MSVC) installed.

```cmd
pip install numpy pybind11
python setup.py build_ext --inplace
```

## Usage

### Running the Benchmark
Verify the performance improvements (typically 5x-20x faster than NumPy depending on CPU):

```bash
python3 benchmark.py
```

### Using the C++ Core
You can import the optimized module directly in your Python scripts:

```python
import jepa_core
import numpy as np

# Create tensors (Batch Size x Dimension)
# Memory should ideally be contiguous for best performance
pred = np.random.rand(4096, 1024).astype(np.float32)
target = np.random.rand(4096, 1024).astype(np.float32)
result = np.zeros(4096, dtype=np.float32)

# Compute L2 Squared Distance (Parallelized & Vectorized)
jepa_core.compute_batch_l2_sq(pred, target, result)
```

### Legacy Versions

**v2 Standalone C++:**
```bash
# Linux
g++ -O3 v2-temporal-patterns.cpp -o jepa_temporal
./jepa_temporal

# Windows
cl /EHsc /O2 v2-temporal-patterns.cpp /Fe:jepa_temporal.exe
```

**v1 Python:**
```bash
python3 v1-sementic-search.py
```

## Performance Optimizations

The `jepa_core` module implements several HPC techniques:
*   **SIMD (AVX/AVX2)**: Hand-written intrinsics processing 8 floats per cycle.
*   **Multithreading**: OpenMP parallelism over the batch dimension.
*   **Zero-Copy**: Direct memory access between Python and C++ buffers to avoid overhead.
*   **Cache Locality**: Linear memory access patterns to maximize hardware prefetching.
