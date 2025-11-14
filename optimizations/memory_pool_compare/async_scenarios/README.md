# Async Memory Pool Advantage Testing

Tests to determine where CUDA Async Memory Pool should excel compared to 
traditional Pool and Arena allocators.

## The Question

CUDA Async Memory Allocator (CUDA 11.2+) provides stream-ordered allocation 
with no global locks. Should be faster, right?

**Spoiler:** In practice, differences are < 2% for most workloads.

## Test Scenarios

### 1. Extreme Size Variation
- Allocation sizes: 100K to 10M rows (100x range)
- Tests dynamic adaptation vs fixed pool

### 2. Very High Concurrency
- 32 concurrent streams
- Tests lock-free stream-ordered allocation

### 3. Rapid Alloc/Dealloc Churn
- Frequent create/destroy cycles
- Tests stream-ordered deallocation efficiency

### 4. Interleaved Multi-Stream Operations
- Operations interleaved across many streams
- Tests cross-stream memory reuse

### 5. Small Frequent Allocations
- Many small allocations (50K rows each)
- Tests allocation overhead

## Build and Run

```bash
./build.sh
./build/async_advantage_test
```

## Expected Results

**Performance differences: < 5% in most cases**

```
Typical results:
  Pool  : 1934ms
  Async : 1918ms  (+0.8% faster)
  Arena : 1944ms
```

## Why Such Small Differences?

1. **Pool pre-allocation is excellent** - Already very fast
2. **GPU memory subsystem is efficient** - CUDA 12.x is well-optimized
3. **Compute dominates** - Allocation is small % of total time
4. **Workloads are predictable** - Pool's strategy works well

## Key Findings

### Async Pool Advantages (Theoretical):
- ✅ Stream-ordered allocation (no global locks)
- ✅ Automatic size adaptation
- ✅ Cross-stream memory reuse

### Async Pool Advantages (Practical):
- ⚠️ < 2% performance difference
- ✅ Zero configuration required
- ✅ Works for any workload

### Real Winners:
1. **Pool** - Best for production (predictable + tuned)
2. **Async** - Best for development (zero-config)
3. **Arena** - Sometimes wins in pure concurrent scenarios

## Recommendation

**For Rapids/Spark Production:**
- Use **Pool** (proven, predictable, slightly faster)

**For Development/Research:**
- Use **Async** (zero-config, good enough)

**Focus optimization on:**
- Kernel fusion (+23%)
- Stream synchronization (+50%)
- Not allocator choice (< 2%)

## Documentation

- `EXTREME_FRAGMENTATION_FINDINGS.md` - **🔥 NEWEST!** Large allocs reverse the winner!
- `MULTITHREAD_FRAGMENTATION_FINDINGS.md` - Multi-threaded + medium allocs
- `FRAGMENTATION_FINDINGS.md` - Single-thread fragmentation analysis
- `ASYNC_ADVANTAGES.md` - Detailed analysis of why async doesn't dominate
  - Where it should excel (theory vs reality)
  - Practical recommendations
  - The hard truth about allocator performance

## Key Findings 🎯

### Medium Allocations (10-100MB)
**Multi-threaded allocation exposes fragmentation impact!**

At 8 threads with medium allocations:
- Pool: +10.7% degradation
- Async: +6.8% degradation ← **4% better!**
- This is where Async's stream-ordered advantage shows up

### Large Allocations (>100MB) 🔥 **NEW!**
**The tables turn completely!**

At 8 threads with LARGE allocations (20MB-1GB):
- Pool: -6.0% (faster!) ✅ **14.5% better!**
- Async: +8.5% (slower) ⚠️
- Allocation size determines the winner!

