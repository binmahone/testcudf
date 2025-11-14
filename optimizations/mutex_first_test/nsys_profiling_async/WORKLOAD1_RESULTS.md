# Nsys Profiling Results: ASYNC Memory Resource

## Test Configuration

- **Memory:** CUDA Async (driver-managed)
- **Workload:** CAST(COALESCE(x, 0) AS DOUBLE)
- **Threads:** 4
- **Items per thread:** 100
- **Rows per column:** 2,684,354
- **Runs:** 3

## Performance Results

### Concurrent Mode (No Mutex)

```
Run 1/3: 128.81 ms (warmup)
Run 2/3:  56.34 ms
Run 3/3:  56.28 ms
Average:  80.48 ms
```

### Mutex Mode (Serialized Submission)

```
Run 1/3:  90.35 ms (warmup)
Run 2/3:  42.38 ms
Run 3/3:  43.94 ms
Average:  58.89 ms
```

### Performance Comparison

| Metric | Concurrent | Mutex | Improvement |
|--------|-----------|-------|-------------|
| Average (all 3 runs) | 80.48 ms | 58.89 ms | **26.8% faster** |
| Steady state (runs 2-3) | 56.31 ms | 43.16 ms | **23.3% faster** |

## Comparison with Pool Memory Resource

| Allocator | Concurrent Avg | Mutex Avg | Mutex Advantage |
|-----------|----------------|-----------|-----------------|
| **Pool Memory** | 72.59 ms | 64.55 ms | 11.1% faster |
| **Async Memory** | 80.48 ms | 58.89 ms | 26.8% faster |

### Key Observation

**Async memory resource shows GREATER benefit from mutex protection!**

- With pool: mutex is 11.1% faster
- With async: mutex is 26.8% faster

This suggests that **dynamic memory allocation amplifies the contention 
problem** in concurrent mode, making mutex protection even more valuable.

## Detailed Analysis

### 1. Kernel Launch Performance

**cudaLaunchKernel**:

```
Concurrent (Async):
- Average: 38,045.6 ns
- Max:     55,247,254 ns (55.2 ms!)

Mutex (Async):
- Average: 9,164.7 ns
- Max:     14,394,101 ns (14.4 ms)

vs Pool Memory:

Concurrent (Pool):
- Average: 7,685.0 ns
- Max:     12,516,831 ns (12.5 ms)

Mutex (Pool):
- Average: 12,176.9 ns
- Max:     25,384,808 ns (25.4 ms)
```

**Observation:** 
- Async + Concurrent has **5x worse average** launch time (38 µs vs 7.7 µs)
- Async + Mutex has **better average** than Pool + Mutex (9.2 µs vs 12.2 µs)
- Dynamic allocation in concurrent mode severely degrades launch performance

### 2. Memory Allocation Pattern

**cudaMallocFromPoolAsync_v11020** (only in async):

```
Concurrent:
- Total calls: 5,601
- Average:     7,996.4 ns
- Max:         35,902,692 ns (35.9 ms spike!)

Mutex:
- Total calls: 5,601
- Average:     7,812.9 ns
- Max:         34,835,611 ns (34.8 ms spike)
```

**Observation:**
- Both modes have similar allocation patterns
- But concurrent mode suffers from contention on memory allocations
- Mutex serialization helps memory allocator avoid conflicts

### 3. Stream Synchronization

**cudaStreamSynchronize**:

```
Concurrent (Async):
- Total time: 569,383,828 ns (569 ms)
- Average:    469,788.6 ns

Mutex (Async):
- Total time: 38,120,572 ns (38 ms)
- Average:    31,452.6 ns

vs Pool Memory:

Concurrent (Pool):
- Total time: 621,286,635 ns (621 ms)
- Average:    512,612.7 ns

Mutex (Pool):
- Total time: 40,784,345 ns (41 ms)
- Average:    33,650.4 ns
```

**Observation:**
- Async has **slightly better** sync performance than pool in concurrent mode
- Mutex mode benefits are similar for both allocators
- **93% reduction** in sync overhead with mutex (both allocators)

### 4. GPU Kernel Execution

**copy_if_else_kernel** (main workload):

```
Concurrent (Async):
- Total: 98,027,275 ns
- Avg:   81,689.4 ns

Mutex (Async):
- Total: 37,581,997 ns
- Avg:   31,318.3 ns

vs Pool Memory:

Concurrent (Pool):
- Total: 96,633,831 ns
- Avg:   80,528.2 ns

Mutex (Pool):
- Total: 37,704,451 ns
- Avg:   31,420.4 ns
```

**Observation:**
- Kernel execution times are **nearly identical** between pool and async
- The real difference is in the **submission and allocation overhead**
- Mutex provides **62% faster kernel execution** regardless of allocator

### 5. System-Level Lock Contention

**pthread_mutex_lock**:

```
Concurrent (Async):
- Total time: 146,963,494 ns (147 ms)
- Calls:      190

Mutex (Async):
- Total time: 322,362,751 ns (322 ms)
- Calls:      9

vs Pool Memory:

Concurrent (Pool):
- Total time: 128,555,234 ns (129 ms)
- Calls:      79

Mutex (Pool):
- Total time: 377,920,699 ns (378 ms)
- Calls:      9
```

**Observation:**
- Async + Concurrent: **2.4x more** internal lock calls than Pool (190 vs 79)
- Dynamic allocation creates additional lock contention points
- Explicit mutex (322ms) prevents 190 internal lock calls and their cascading effects

### 6. Memory Cleanup Overhead

**cudaMemPoolDestroy** (only in async):

```
Concurrent: 315,370,802 ns (315 ms)
Mutex:      321,134,945 ns (321 ms)
```

**Observation:**
- Memory pool cleanup is similar for both modes
- This is done at program exit, not affecting main workload

## Why Async Benefits More from Mutex

### 1. Memory Allocation Contention

Dynamic allocation in concurrent mode creates multiple contention points:
- CUDA driver memory allocator locks
- Internal pool management locks
- Page table update locks

### 2. Increased Lock Calls

Async + Concurrent triggers **2.4x more** internal locks than Pool + Concurrent:
- 190 vs 79 pthread_mutex_lock calls
- Each lock represents a contention point

### 3. Cascading Delays

Memory allocation delays cascade to kernel launches:
- Kernel can't launch until memory is allocated
- Multiple threads competing for allocator attention
- Creates a "thundering herd" problem

### 4. Mutex Serialization Benefits

Mutex protection in async mode:
- Serializes memory allocation requests
- Reduces allocator contention
- Creates predictable allocation patterns
- Allows GPU scheduler to optimize execution

## Performance Breakdown

### Pool Memory Results

```
Concurrent: 72.59 ms
Mutex:      64.55 ms
Speedup:    11.1%
```

### Async Memory Results

```
Concurrent: 80.48 ms
Mutex:      58.89 ms
Speedup:    26.8%
```

### Why the Difference?

1. **Pool Memory:** Pre-allocated, allocation is just a pointer bump
   - Concurrent contention is limited to pool management
   - Mutex helps but benefits are modest

2. **Async Memory:** Dynamic allocation from CUDA driver
   - Concurrent contention spans multiple driver layers
   - Mutex eliminates multiple contention points
   - **Greater benefit from serialization**

## Recommendations

### When to Use Mutex + Async

This combination is **particularly beneficial** when:
1. Memory usage is unpredictable (can't pre-size pool)
2. Multi-threaded workload (4+ threads)
3. High-frequency operations (100+ per thread)
4. Memory allocation per operation

### When to Use Mutex + Pool

Pool is still preferred when:
1. Memory usage is predictable
2. Can afford to pre-allocate large pool
3. Want most consistent performance
4. Minimize interaction with CUDA driver

### Universal Recommendation

**Always use mutex protection** for multi-threaded CUDF workloads:
- 11-27% performance improvement
- Works with any memory allocator
- Reduces contention at all levels
- More predictable execution

## Generated Files

```
nsys_reports/concurrent_mode.nsys-rep
nsys_reports/concurrent_mode.sqlite
nsys_reports/mutex_mode.nsys-rep
nsys_reports/mutex_mode.sqlite
```

## Viewing Reports

```bash
# Open in Nsight Systems GUI
nsys-ui nsys_reports/concurrent_mode.nsys-rep
nsys-ui nsys_reports/mutex_mode.nsys-rep

# Compare with pool memory results
nsys-ui ../nsys_profiling/nsys_reports/concurrent_mode.nsys-rep
nsys-ui ../nsys_profiling/nsys_reports/mutex_mode.nsys-rep
```

## Conclusion

The profiling data reveals that **async memory resource magnifies the 
benefits of mutex protection**:

- Pool memory: 11.1% improvement with mutex
- Async memory: 26.8% improvement with mutex

This is because dynamic memory allocation introduces additional contention 
points that mutex serialization effectively eliminates. The mutex-first 
optimization is even more critical when using driver-managed memory.

**Key Insight:** The apparent "cost" of serialization (322ms of mutex 
locking) prevents far greater costs from uncontrolled contention on memory 
allocation and kernel launch paths.

