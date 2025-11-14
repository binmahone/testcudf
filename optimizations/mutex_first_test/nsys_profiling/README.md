# Nsys Profiling for Concurrent vs Mutex Modes

This directory contains separate executables for profiling concurrent and mutex 
kernel submission modes using NVIDIA Nsight Systems (nsys).

## Directory Structure

```
nsys_profiling/
├── concurrent_mode.cpp     # Concurrent submission mode only
├── mutex_mode.cpp          # Mutex-protected submission mode only
├── CMakeLists.txt          # Build configuration
├── build.sh                # Build script
├── profile.sh              # Automated profiling script
└── README.md               # This file
```

## Purpose

The original `original_with_mutex_test.cpp` runs both concurrent and mutex 
modes in the same execution. For detailed nsys profiling, we need separate 
executables to:

1. Clearly isolate the behavior of each submission mode
2. Compare kernel submission patterns side-by-side
3. Analyze contention and serialization effects independently

## Building

```bash
./build.sh
```

This will create:
- `build/concurrent_mode` - Runs workloads with concurrent kernel submission
- `build/mutex_mode` - Runs workloads with mutex-protected submission

## Manual Profiling

### Profile Concurrent Mode

```bash
nsys profile \
    --output=concurrent_mode \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    ./build/concurrent_mode
```

### Profile Mutex Mode

```bash
nsys profile \
    --output=mutex_mode \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    ./build/mutex_mode
```

## Automated Profiling

Use the provided script to profile both modes:

```bash
./profile.sh
```

This will generate:
- `nsys_reports/concurrent_mode.nsys-rep`
- `nsys_reports/mutex_mode.nsys-rep`

## Viewing Results

### Open in Nsight Systems GUI

```bash
nsys-ui nsys_reports/concurrent_mode.nsys-rep
nsys-ui nsys_reports/mutex_mode.nsys-rep
```

### Export Statistics

```bash
nsys stats nsys_reports/concurrent_mode.nsys-rep
nsys stats nsys_reports/mutex_mode.nsys-rep
```

### Compare Reports

```bash
nsys stats --report cuda_api_sum nsys_reports/concurrent_mode.nsys-rep
nsys stats --report cuda_api_sum nsys_reports/mutex_mode.nsys-rep
```

## What to Look For

### In Concurrent Mode
- Multiple CUDA API calls overlapping in timeline
- Potential contention in kernel launch APIs
- Parallel kernel submissions from different streams

### In Mutex Mode
- Serialized CUDA API calls within mutex-protected regions
- Reduced contention in kernel launch
- Sequential pattern in kernel submissions per thread

### Key Metrics to Compare
1. **CUDA API call duration**: Compare `cudaLaunchKernel` times
2. **Kernel execution overlap**: Check concurrent kernel execution
3. **Stream synchronization**: Analyze sync points
4. **CPU thread activity**: Compare thread scheduling patterns

## Configuration

Both executables use the same configuration:
- Threads: 4
- Items per thread: 100
- Rows per column: 2,684,354
- Runs: 3 (reduced from 10 for faster profiling)

## Workloads

### Workload 1
`CAST(COALESCE(x, 0) AS DOUBLE)`
- 3 kernels: is_valid, copy_if_else, cast

### Workload 2
`CAST(COALESCE(IF(y=1, x, 0), 0) AS DOUBLE)`
- 5+ kernels: is_valid, copy_if_else, binary_operation, copy_if_else, cast

## Notes

- Profiling adds overhead; execution times will be longer than without nsys
- Use Release build for representative performance characteristics
- The reduced run count (3 vs 10) balances profile detail with collection time
- Each executable runs both workloads sequentially for comprehensive analysis

