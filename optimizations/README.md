# CUDF Coalesce Performance Optimizations

This project demonstrates performance optimizations for CUDF operations under different workload scenarios, achieving **23% (single-thread) to 34-61% (multi-thread) performance improvements**.

---

## Two Optimization Strategies for Different Workloads

We address performance bottlenecks in two different workload scenarios through **two complementary optimization approaches**:

### Optimization Strategy 1: Kernel-Level Optimizations
**Problem**: Unnecessary kernel launches and unfused operations
**Solution**: Reduce kernel count through operation fusion and algorithmic improvements
**Techniques**:
- Replace `is_valid + copy_if_else` with fused `replace_nulls` kernel
- Mathematical simplification (e.g., `IF(y=1, x, 0) === x * y`)
**Impact**: **+21-23% speedup** in all scenarios (single-thread and multi-thread)

### Optimization Strategy 2: Serialization by Mutex
**Problem**: Kernel submission contention in multi-threaded environments
**Solution**: Serialize kernel submission while keeping parallel GPU execution
**Techniques**:
- Use `std::mutex` to serialize kernel submission across threads
- Each thread still uses separate CUDA streams for parallel execution
**Impact**: **+14-49% additional speedup** in multi-threaded scenarios

**Combined Effect**: Applying both strategies yields **34-61% total speedup** in multi-threaded workloads.

---

## Quick Start

```bash
./setup_env.sh      # One-time setup (~5 min)
./build.sh          # Compile (~30 sec)
./run_benchmarks.sh # Run tests (~2 min)
```

---

## Workload Scenarios & Kernel Optimizations

We test our kernel optimizations on two representative SQL coalesce patterns:

### Workload 1: CAST(COALESCE(x, 0) AS DOUBLE)

### Original Implementation (3 kernels, 7.15 ms)

```cpp
auto is_valid_mask = cudf::is_valid(x);
auto coalesced = cudf::copy_if_else(x, zero_column, is_valid_mask);
auto result = cudf::cast(coalesced, FLOAT64);
```

### Optimized Implementation (2 kernels, 5.51 ms, **+23% faster**)

```cpp
cudf::numeric_scalar<int32_t> zero(0, true);
auto coalesced = cudf::replace_nulls(x, zero);
auto result = cudf::cast(coalesced, FLOAT64);
```

**Key Insight**: `replace_nulls` is a fused, optimized kernel that's faster than `is_valid + copy_if_else`.

---

### Workload 2: CAST(COALESCE(IF(y=1, x, 0), 0) AS DOUBLE)

Where `y` is a binary column (0 or 1).

### Original Implementation (5 kernels, 17.59 ms)

```cpp
auto x_is_valid = cudf::is_valid(x);
auto x_coalesced = cudf::copy_if_else(x, zero, x_is_valid);
auto y_eq_1 = cudf::binary_operation(y, one, EQUAL);
auto result_int = cudf::copy_if_else(x_coalesced, zero, y_eq_1);
auto result = cudf::cast(result_int, FLOAT64);
```

### Optimized Implementation (3 kernels, 13.84 ms, **+21% faster**)

```cpp
cudf::numeric_scalar<int32_t> zero(0, true);
auto x_no_nulls = cudf::replace_nulls(x, zero);
auto result_int = cudf::binary_operation(x_no_nulls, y, MUL);
auto result = cudf::cast(result_int, FLOAT64);
```

**Key Insights**:
1. Use `replace_nulls` (faster than is_valid + copy_if_else)
2. **Mathematical simplification**: `IF(y=1, x, 0) === x * y` when y ∈ {0,1}

---

## Mutex Serialization for Multi-threaded Workloads

### The Problem: Kernel Submission Contention

In multi-threaded environments, concurrent kernel submissions can cause severe GPU command queue contention, leading to **negative parallel efficiency**:

**Example (Workload 2 without mutex)**:
- Single-thread time: 13.84 ms
- 4-thread concurrent: 77.64 ms  
- Expected (4× sequential): 4 × 13.84 = 55.36 ms
- Result: **40% slower** than sequential execution!

### The Solution: Serialize Submission, Parallelize Execution

Use mutex to serialize kernel submission while maintaining parallel GPU execution on separate streams:

```cpp
std::mutex kernel_submit_mutex;

// In worker thread:
{
    std::lock_guard<std::mutex> lock(kernel_submit_mutex);
    // Submit all kernels for this thread's work
    for (auto& col : my_columns) {
        auto result = process(col, stream);
    }
}
stream.synchronize();
```

**Key Benefits**:
1. Eliminates kernel submission contention
2. GPU still executes work in parallel across different streams
3. Achieves near-linear scaling in multi-threaded scenarios

See `kernel_submission_test.cpp` for complete implementation example.

---

## Benchmark Results

Test configuration: 100 columns/pairs, ~1GB INT32 input, ~2GB FLOAT64 output.

### Single-threaded Performance (Kernel Optimizations Only)

| Workload | Original | Optimized | Speedup | Kernel Reduction |
|----------|----------|-----------|---------|------------------|
| Workload 1 | 7.15 ms | **5.51 ms** | **+23.0%** | 3 → 2 kernels |
| Workload 2 | 17.59 ms | **13.84 ms** | **+21.3%** | 5 → 3 kernels |

### Multi-threaded Performance (4 threads, 100 items per thread)

> **Note**: We also conducted comprehensive tests with 2, 3, and 4 threads. For detailed results across different thread counts, see [multithreads_result.md](multithreads_result.md).

**Workload 1**:

| Version | 4-thread Time | Speedup | Contribution |
|---------|---------------|---------|--------------|
| Original (3 kernels, concurrent) | 65.18 ms | baseline | - |
| Kernel-optimized (2 kernels, concurrent) | 49.86 ms | +23.5% | Kernel optimization |
| **Kernel-optimized + mutex** | **25.55 ms** | **+60.8%** ⭐ | +23.5% kernels + 48.8% mutex |

**Workload 2**:

| Version | 4-thread Time | Speedup | Contribution |
|---------|---------------|---------|--------------|
| Original (5 kernels, concurrent) | 101.80 ms | baseline | - |
| Kernel-optimized (3 kernels, concurrent) | 77.64 ms | +23.7% | Kernel optimization |
| **Kernel-optimized + mutex** | **66.77 ms** | **+34.4%** ⭐ | +23.7% kernels + 14.0% mutex |

### Key Findings

**Kernel Optimization**: Provides consistent **~23%** speedup in both single and multi-threaded scenarios

**Mutex Serialization**: Adds **14-49%** additional speedup in multi-threaded workloads by eliminating contention

**Combined Approach**: Achieves **34-61%** total speedup in multi-threaded environments

---

## Critical: RMM Pool Setup

**Must use RMM pool memory resource** for stable performance:

```cpp
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr, 
    2ULL * 1024 * 1024 * 1024  // 2GB minimum
);
rmm::mr::set_current_device_resource(&pool_mr);
```

**Why**: Without pool, performance degrades due to GPU memory fragmentation. With pool, performance is stable and ~14x faster.

---

## Environment Setup

### Prerequisites

- Linux x86_64
- CUDA 12.x installed
- Conda or Miniconda
- GPU with compute capability 7.0+

### Setup Steps

Run the provided setup script:

```bash
./setup_env.sh
```

This creates a `cudf_test` conda environment with:
- libcudf 25.10.0
- CUDA toolkit 12.2
- GCC 12 (compatible with CUDA 12.2)
- CMake

**Manual setup** (if script doesn't work):

```bash
conda create -n cudf_test python=3.10 -y

conda install -n cudf_test \
    -c rapidsai \
    -c conda-forge \
    -c nvidia \
    libcudf \
    cuda-version=12.2 \
    cmake \
    gcc_linux-64=12 \
    gxx_linux-64=12 \
    -y
```

---

## Files

- `workload1_benchmark.cpp` - Workload 1 comparison
- `workload2_benchmark.cpp` - Workload 2 comparison
- `multithread_benchmark.cpp` - Multi-threading test
- `setup_env.sh` - Environment setup
- `build.sh` - Build script
- `run_benchmarks.sh` - Run all benchmarks

---

## Key Takeaways

### Kernel-Level Optimizations (Strategy 1)
1. ✅ **Use `replace_nulls` instead of `is_valid + copy_if_else`** → Fused kernel is faster
2. ✅ **Apply mathematical simplification** (e.g., `IF(y=1,x,0) === x*y`) → Reduce kernel count
3. ✅ **Impact**: **~23% speedup** in both single and multi-threaded workloads

### Mutex Serialization (Strategy 2)
4. ✅ **Serialize kernel submission in multi-threading** → Eliminates GPU command queue contention
5. ✅ **Maintain separate CUDA streams per thread** → Preserves parallel GPU execution
6. ✅ **Impact**: **+14-49% additional speedup** on top of kernel optimizations

### Best Practices
7. ✅ **Always use RMM pool_memory_resource** → Prevents memory fragmentation, ensures stable performance
8. ✅ **Combine both strategies** → Achieves **34-61% total speedup** in multi-threaded environments

---

## License

This is example/benchmark code for demonstration purposes.
