#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <immintrin.h> // AVX2
#include <cmath>
#include <iostream>

namespace py = pybind11;

// Function to compute Batched L2 Distance Squared
// Input: Two matrices (Batch Size N x Dimension D)
// Output: One vector (Batch Size N) containing squared Euclidean distance per sample
// Optimizations: AVX2 SIMD, OpenMP Multithreading, Memory Alignment assumptions
void compute_batch_l2_sq(
    py::array_t<float> pred_array,
    py::array_t<float> target_array,
    py::array_t<float> result_array
) {
    // 1. Request buffer info (Zero-Copy access)
    // We request raw pointers. Using unchecked proxies would also work but raw pointers allow cleaner SIMD.
    py::buffer_info pred_info = pred_array.request();
    py::buffer_info target_info = target_array.request();
    py::buffer_info result_info = result_array.request();

    // 2. Validation
    if (pred_info.ndim != 2 || target_info.ndim != 2) {
        throw std::runtime_error("Inputs must be 2D tensors (Batch x Dimension)");
    }
    if (pred_info.shape[0] != target_info.shape[0] || pred_info.shape[1] != target_info.shape[1]) {
        throw std::runtime_error("Input shapes must match");
    }

    size_t N = pred_info.shape[0]; // Batch Size
    size_t D = pred_info.shape[1]; // Latent Dimension

    // Pointers to data
    float* pred_ptr = static_cast<float*>(pred_info.ptr);
    float* target_ptr = static_cast<float*>(target_info.ptr);
    float* result_ptr = static_cast<float*>(result_info.ptr);

    // 3. Parallel Execution (Multithreading)
    // We parallelize over the Batch dimension (N).
    // This scales linearly with cores and keeps threads independent (no race conditions).
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        
        float sum_sq_diff = 0.0f;
        size_t j = 0;

        // Pointers to the start of the current row
        // Optimization: Linear access pattern ensures hardware prefetching works effectively.
        const float* p_row = pred_ptr + i * D;
        const float* t_row = target_ptr + i * D;

        // 4. Vectorization (SIMD with AVX2)
        // Process 8 floats at a time (256 bits).
        // Using explicit intrinsics ensures the compiler generates vector instructions.
        __m256 sum_vec = _mm256_setzero_ps(); // Accumulator register

        for (; j <= D - 8; j += 8) {
            // Unaligned loads allow Python to send non-aligned buffers (safe but slightly slower than aligned_load)
            // If we enforced alignment in Python, we could use _mm256_load_ps
            __m256 v_pred = _mm256_loadu_ps(p_row + j);
            __m256 v_target = _mm256_loadu_ps(t_row + j);

            // diff = pred - target
            __m256 v_diff = _mm256_sub_ps(v_pred, v_target);

            // sum += diff * diff
            // We use mul + add instead of fmadd for broader compatibility (AVX vs AVX2/FMA)
            __m256 v_sq = _mm256_mul_ps(v_diff, v_diff);
            sum_vec = _mm256_add_ps(sum_vec, v_sq);
        }

        // Horizontal sum of the SIMD register
        // We need to sum the 8 floats inside sum_vec into a single scalar.
        // A common trick is to permute and add.
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int k = 0; k < 8; ++k) {
            sum_sq_diff += temp[k];
        }

        // 5. Cleanup Loop (Scalar)
        // Handle remaining elements if D is not a multiple of 8
        for (; j < D; ++j) {
            float diff = p_row[j] - t_row[j];
            sum_sq_diff += diff * diff;
        }

        result_ptr[i] = sum_sq_diff;
    }
}

PYBIND11_MODULE(jepa_core, m) {
    m.doc() = "Optimized C++ JEPA Core Components";
    m.def("compute_batch_l2_sq", &compute_batch_l2_sq, 
        "Compute squared L2 distance for a batch",
        py::arg("pred"), py::arg("target"), py::arg("result"));
}
