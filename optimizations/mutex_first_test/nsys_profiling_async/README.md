# Nsys Profiling with ASYNC Memory Resource

This directory contains executables for profiling concurrent and mutex 
kernel submission modes using **CUDA Async Memory Resource** (driver-managed).

## Difference from nsys_profiling

The only difference from `nsys_profiling/` is the memory allocator:

- **nsys_profiling**: Uses `pool_memory_resource` (pre-allocated pool)
- **nsys_profiling_async**: Uses `cuda_async_memory_resource` (driver-managed)

All other aspects (workload, threading, etc.) are identical.

## Directory Structure

```
nsys_profiling_async/
├── concurrent_mode.cpp     # Concurrent submission with async memory
├── mutex_mode.cpp          # Mutex-protected submission with async memory
├── CMakeLists.txt          # Build configuration
├── build.sh                # Build script
├── profile.sh              # Automated profiling script
└── README.md               # This file
```

## Memory Resource Comparison

### Pool Memory Resource (default)
```cpp
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr, 10ULL * 1024 * 1024 * 1024
);
rmm::mr::set_current_device_resource(&pool_mr);
```

### ASYNC Memory Resource (this directory)
```cpp
rmm::mr::cuda_async_memory_resource async_mr;
rmm::mr::set_current_device_resource(&async_mr);
```

## Building

```bash
./build.sh
```

This will create:
- `build/concurrent_mode` - Runs workloads with concurrent kernel submission
- `build/mutex_mode` - Runs workloads with mutex-protected submission

## Profiling

### Automated

```bash
./profile.sh
```

### Manual

```bash
# Profile concurrent mode
nsys profile \
    --output=concurrent_mode_async \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    ./build/concurrent_mode

# Profile mutex mode
nsys profile \
    --output=mutex_mode_async \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    ./build/mutex_mode
```

## Viewing Results

### Open in Nsight Systems GUI

```bash
nsys-ui nsys_reports/concurrent_mode.nsys-rep
nsys-ui nsys_reports/mutex_mode.nsys-rep
```

### Export Statistics

```bash
nsys stats --report cuda_api_sum nsys_reports/concurrent_mode.nsys-rep
nsys stats --report cuda_api_sum nsys_reports/mutex_mode.nsys-rep
```

## Configuration

Both executables use the same configuration:
- **Memory:** CUDA Async (driver-managed)
- **Threads:** 4
- **Items per thread:** 100
- **Rows per column:** 2,684,354
- **Runs:** 3

## Workload

### Workload 1 (only workload tested)
`CAST(COALESCE(x, 0) AS DOUBLE)`
- 3 kernels: is_valid, copy_if_else, cast

## Expected Observations

With async memory resource, you should observe:

1. **Memory allocation patterns**: No pre-allocated pool, dynamic allocation
2. **Driver interaction**: More frequent interaction with CUDA driver for 
   memory management
3. **Potential differences**: Memory allocation overhead may affect the 
   concurrent vs mutex comparison differently than with pool memory

## Comparison with Pool Memory

To compare results with pool memory resource, see:
- `../nsys_profiling/` for pool-based profiling
- `../nsys_profiling/WORKLOAD1_RESULTS.md` for pool-based results

Key questions to investigate:
1. Does async memory change the concurrent vs mutex performance gap?
2. How does dynamic allocation affect kernel submission contention?
3. Are there different memory allocation patterns between the two modes?

## Notes

- CUDA Async Memory Resource requires CUDA 11.2+ with appropriate driver
- Async allocator is generally faster for dynamic workloads
- Pool allocator has more predictable performance but requires pre-allocation
- Both use the same GPU stream pool for kernel execution

