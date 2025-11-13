# CUDF Coalesce Performance Optimizations

Optimized implementations for two common SQL coalesce expressions achieving **~23% performance improvement**.

---

## Quick Start

```bash
./setup_env.sh      # One-time setup (~5 min)
./build.sh          # Compile (~30 sec)
./run_benchmarks.sh # Run tests (~2 min)
```

---

## Optimization 1: CAST(COALESCE(x, 0) AS DOUBLE)

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

## Optimization 2: CAST(COALESCE(IF(y=1, x, 0), 0) AS DOUBLE)

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

## Benchmark Results

Test configuration: 100 columns/pairs, ~1GB INT32 input, ~2GB FLOAT64 output.

### Single-threaded Performance

| Workload | Original | Optimized | Speedup | Kernel Reduction |
|----------|----------|-----------|---------|------------------|
| Workload 1 | 7.15 ms | **5.51 ms** | **+23.0%** | 3 → 2 kernels |
| Workload 2 | 17.59 ms | **13.84 ms** | **+21.3%** | 5 → 3 kernels |

### Multi-threading Performance (4 threads, 100 items per thread)

**Workload 1**:

| Version | 4-thread Time | Speedup vs Original |
|---------|---------------|---------------------|
| Original (3 kernels) | 65.18 ms | baseline |
| Optimized (2 kernels) | **49.86 ms** | **+23.5% faster** |

**Workload 2**:

| Version | 4-thread Time | Speedup vs Original |
|---------|---------------|---------------------|
| Original (5 kernels) | 101.80 ms | baseline |
| Optimized (3 kernels) | **77.64 ms** | **+23.7% faster** |

**Key Finding**: Optimizations provide **~23% speedup even in multi-threading**, where kernel submission bottleneck exists. Reducing kernel count helps mitigate the bottleneck.

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

1. ✅ **Use `replace_nulls` instead of `is_valid + copy_if_else`** → ~23% faster
2. ✅ **Simplify logic mathematically** (e.g., IF(y=1,x,0) = x*y) → Fewer kernels
3. ✅ **Always use RMM pool_memory_resource** → Stable performance
4. ✅ **Reduce kernel count** → Helps both single and multi-threading

---

## License

This is example/benchmark code for demonstration purposes.
