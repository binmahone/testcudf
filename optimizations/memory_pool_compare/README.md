# Memory Resource Performance Comparison

This directory contains comprehensive tests comparing different RMM memory resources 
for various workload patterns.

## Overview

Compares three RMM memory resource types:
- **Pool Memory Resource** (`pool_memory_resource`)
- **CUDA Async Memory Resource** (`cuda_async_memory_resource`)
- **Arena Memory Resource** (`arena_memory_resource`)

## Files

### Test Programs
- `memory_resource_test.cpp` - Main test: Predictable workload (with/without mutex)
- `async_scenarios/` - Subdirectory with async-specific tests

### Build
- `CMakeLists.txt` - Build configuration
- `build.sh` - Build script

## Quick Start

### Build Main Test

```bash
./build.sh
```

### Run Main Test

```bash
# Test: Predictable workload with/without mutex
./build/memory_resource_test
```

### Async-Specific Tests

```bash
# Build and run async advantage tests
cd async_scenarios
./build.sh
./build/async_advantage_test
```

## Test Configuration

- **Threads**: 4
- **Items per thread**: 100
- **Rows per column**: 2,684,354
- **Test runs**: 10 (averaged)

## Key Findings

### Workload 1: CAST(COALESCE(x,0) AS DOUBLE)

| Memory Resource | Concurrent | With Mutex | Speedup |
|----------------|------------|------------|---------|
| Pool           | ~53 ms     | ~26 ms     | **50%** |
| CUDA Async     | ~47 ms     | ~31 ms     | 35%     |
| Arena          | **~43 ms** | ~34 ms     | 20%     |

### Workload 2: CAST(COALESCE(IF(y=1,x,0),0) AS DOUBLE)

| Memory Resource | Concurrent | With Mutex | Speedup |
|----------------|------------|------------|---------|
| Pool           | ~78 ms     | ~67 ms     | **14%** |
| CUDA Async     | ~78 ms     | ~73 ms     | 5%      |
| Arena          | ~78 ms     | ~88 ms     | **-10%** |

## Recommendation

**Use Pool Memory Resource** for production environments:
- Best absolute performance with mutex (~26ms)
- Highest and most consistent speedup (50%)
- No risk of performance degradation
- Stable behavior across workload types

## Detailed Analysis

See `MEMORY_RESOURCE_COMPARISON.md` for comprehensive analysis, technical insights, and recommendations.

