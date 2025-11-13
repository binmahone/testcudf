# Multi-threading Performance Results

Complete benchmark data for 2, 3, and 4 threads comparing original implementation vs optimized (kernel + mutex).

---

## Test Configuration

- **Items per thread**: 100 columns/pairs
- **Data per thread**: ~1GB input
- **Thread counts tested**: 2, 3, 4
- **Runs per test**: 10 (averaged)

---

## Workload 1: CAST(COALESCE(x, 0) AS DOUBLE)

### Results

| Threads | Original+Concurrent | Optimized+Concurrent | Kernel Speedup | Optimized+Mutex | Total Speedup |
|---------|---------------------|----------------------|----------------|-----------------|---------------|
| 2 | 23.92 ms | 20.47 ms | +14.4% | **11.60 ms** | **+51.3%** |
| 3 | 41.51 ms | 33.15 ms | +20.1% | **18.10 ms** | **+56.4%** |
| 4 | 60.10 ms | 44.08 ms | +26.7% | **24.40 ms** | **+59.4%** |

### Optimization Breakdown (4 threads example)

```
Original (3 kernels, concurrent):  60.10 ms
  ↓ Kernel optimization (3→2)
Optimized (2 kernels, concurrent): 44.08 ms  (+26.7%)
  ↓ Mutex serialization
Optimized + mutex:                 24.40 ms  (+32.7% additional)
                                            ─────────────────
Total improvement:                          +59.4%
```

---

## Workload 2: CAST(COALESCE(IF(y=1, x, 0), 0) AS DOUBLE)

### Results

| Threads | Original+Concurrent | Optimized+Concurrent | Kernel Speedup | Optimized+Mutex | Total Speedup |
|---------|---------------------|----------------------|----------------|-----------------|---------------|
| 2 | 46.28 ms | 44.31 ms | +4.2% | **32.50 ms** | **+29.8%** |
| 3 | 68.97 ms | 57.98 ms | +15.9% | **49.50 ms** | **+28.3%** |
| 4 | 100.91 ms | 75.26 ms | +25.4% | **63.20 ms** | **+37.4%** |

### Optimization Breakdown (4 threads example)

```
Original (5 kernels, concurrent):  100.91 ms
  ↓ Kernel optimization (5→3)
Optimized (3 kernels, concurrent):  75.26 ms  (+25.4%)
  ↓ Mutex serialization
Optimized + mutex:                  63.20 ms  (+12.0% additional)
                                             ─────────────────
Total improvement:                           +37.4%
```

---

## Key Observations

### 1. Total Speedup Ranges

- **Workload 1**: 51-59% improvement (increases with thread count)
- **Workload 2**: 28-37% improvement (increases with thread count)

### 2. Mutex Impact

**Workload 1** (simpler, fewer kernels):
- Mutex provides **+33-49% additional** improvement
- Very sensitive to submission contention

**Workload 2** (complex, more kernels):
- Mutex provides **+12-25% additional** improvement  
- Less sensitive, but still significant

### 3. Scaling Pattern

Both workloads show **better optimization gains with more threads**:
- More threads → more contention → more benefit from mutex
- Kernel reduction also helps more with higher thread counts

---

## Recommendations

### For Multi-threaded CUDF Applications

1. **Always use optimized kernel implementations**
   - Workload 1: Use `replace_nulls` (not `is_valid + copy_if_else`)
   - Workload 2: Use mathematical simplification (x*y instead of if(y=1,x,0))

2. **Serialize kernel submission with mutex**
   ```cpp
   std::mutex kernel_submit_mutex;
   
   // In worker thread:
   {
       std::lock_guard<std::mutex> lock(kernel_submit_mutex);
       for (auto& item : my_work) {
           process(item, stream);  // Submit kernels
       }
   }
   stream.synchronize();  // Execute in parallel
   ```

3. **Expected improvements**:
   - 2 threads: 30-51% total
   - 3 threads: 28-56% total
   - 4 threads: 37-59% total

---

## Reproducing Results

```bash
cd optimizations
./build.sh
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/cudf_test/lib:$LD_LIBRARY_PATH
./build/thread_count_test
```

Expected variance: ±5%

---

## Test Date

Generated: November 2025

