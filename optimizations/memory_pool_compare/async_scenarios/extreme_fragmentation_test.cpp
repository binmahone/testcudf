#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <thread>
#include <string>
#include <barrier>

/**
 * Extreme fragmentation test with LARGE allocations (up to 2GB)
 * 8 threads only - the sweet spot where differences show up
 * 
 * Tests if large-scale fragmentation reveals bigger differences
 * between Pool and Async
 */

size_t get_free_memory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

std::string format_mb(size_t bytes) {
    return std::to_string(bytes / (1024.0 * 1024.0)) + " MB";
}

std::string format_gb(size_t bytes) {
    return std::to_string(bytes / (1024.0 * 1024.0 * 1024.0)) + " GB";
}

std::unique_ptr<cudf::column> create_column(
    size_t rows, rmm::cuda_stream_view stream) {
    std::vector<int32_t> data(rows, 42);
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t), stream);
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), rmm::device_buffer{}, 0
    );
}

struct ExtremeResults {
    double baseline_time;
    double after_fragmentation_time;
    double degradation_pct;
    size_t min_free_memory;
    
    void print(const std::string& name) const {
        std::cout << "\n  " << std::setw(15) << std::left << name << ":" 
                  << std::endl;
        std::cout << "    Baseline:        " << std::fixed 
                  << std::setprecision(2) << baseline_time << " ms" 
                  << std::endl;
        std::cout << "    After fragmentation: " << after_fragmentation_time 
                  << " ms";
        
        if (degradation_pct > 15.0) {
            std::cout << "  ⚠️  SEVERE DEGRADATION " << std::setprecision(1)
                      << degradation_pct << "%";
        } else if (degradation_pct > 5.0) {
            std::cout << "  ⚠️  Degraded " << std::setprecision(1)
                      << degradation_pct << "%";
        } else if (degradation_pct < -5.0) {
            std::cout << "  ✓ Improved " << std::setprecision(1)
                      << (-degradation_pct) << "%";
        } else {
            std::cout << "  ✓ Stable (" << std::showpos 
                      << std::setprecision(1) << degradation_pct 
                      << "%)" << std::noshowpos;
        }
        std::cout << std::endl;
        
        std::cout << "    Min free memory: " << format_gb(min_free_memory)
                  << std::endl;
    }
};

double measure_multithread_allocation(size_t num_threads, 
                                       size_t allocs_per_thread,
                                       size_t rows, 
                                       rmm::cuda_stream_pool& stream_pool) {
    std::barrier barrier(num_threads);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, t, stream]() {
            barrier.arrive_and_wait();
            
            std::vector<std::unique_ptr<cudf::column>> columns;
            for (size_t i = 0; i < allocs_per_thread; ++i) {
                columns.push_back(create_column(rows, stream));
            }
            stream.synchronize();
        });
    }
    
    for (auto& th : threads) th.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0;
}

void create_extreme_fragmentation(rmm::cuda_stream_pool& stream_pool,
                                   size_t& min_free) {
    constexpr size_t NUM_THREADS = 8;
    std::mt19937 gen(42);
    
    // Extreme size distribution: 20MB to 1GB (5M to 250M rows)
    std::uniform_int_distribution<size_t> size_dist(
        5000000,      // 20MB (5M rows × 4 bytes)
        250000000     // 1GB (250M rows × 4 bytes)
    );
    
    std::cout << "  Creating extreme fragmentation..." << std::endl;
    std::cout << "  Allocation sizes: 20MB to 1GB" << std::endl;
    
    for (size_t round = 0; round < 40; ++round) {
        if (round % 10 == 0) {
            std::cout << "    Round " << round << "/40..." << std::endl;
        }
        
        std::vector<std::thread> threads;
        std::vector<std::vector<std::unique_ptr<cudf::column>>> 
            per_thread_cols(NUM_THREADS);
        
        // Phase 1: Each thread allocates 5 varying-size columns
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                for (size_t i = 0; i < 5; ++i) {
                    size_t rows = size_dist(gen);
                    per_thread_cols[t].push_back(create_column(rows, stream));
                    
                    // Track min free memory
                    size_t current_free = get_free_memory();
                    if (current_free < min_free) {
                        min_free = current_free;
                    }
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();
        
        // Phase 2: Randomly free some to create holes (keep ~40%)
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&, t]() {
                auto& cols = per_thread_cols[t];
                std::shuffle(cols.begin(), cols.end(), gen);
                if (cols.size() > 2) {
                    cols.resize(2);  // Keep only 2 out of 5
                }
            });
        }
        for (auto& th : threads) th.join();
        
        // Force cleanup
        cudaDeviceSynchronize();
        
        // Check memory status
        size_t current_free = get_free_memory();
        if (current_free < min_free) {
            min_free = current_free;
        }
        
        // Safety check: if too low, clean everything
        if (current_free < 2ULL * 1024 * 1024 * 1024) {
            std::cout << "    ⚠️  Low memory (" << format_gb(current_free)
                      << "), cleaning up..." << std::endl;
            per_thread_cols.clear();
            cudaDeviceSynchronize();
        }
    }
    
    std::cout << "  Fragmentation creation complete" << std::endl;
}

ExtremeResults test_extreme_fragmentation(const std::string& name) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    constexpr size_t NUM_THREADS = 8;
    rmm::cuda_stream_pool stream_pool(NUM_THREADS);
    ExtremeResults results;
    
    constexpr size_t ALLOCS_PER_THREAD = 20;
    constexpr size_t TEST_ROWS = 5000000;  // 20MB each
    
    size_t initial_free = get_free_memory();
    std::cout << "Initial free memory: " << format_gb(initial_free) 
              << std::endl;
    
    // Phase 1: Baseline (clean state)
    std::cout << "\nPhase 1: Baseline measurement (8 threads × 20 allocs)..."
              << std::endl;
    results.baseline_time = measure_multithread_allocation(
        NUM_THREADS, ALLOCS_PER_THREAD, TEST_ROWS, stream_pool);
    std::cout << "  Baseline: " << std::fixed << std::setprecision(2)
              << results.baseline_time << " ms" << std::endl;
    
    // Phase 2: Create EXTREME fragmentation
    std::cout << "\nPhase 2: Creating EXTREME fragmentation (40 rounds)..."
              << std::endl;
    std::cout << "  8 threads × 5 allocs/round × 40 rounds" << std::endl;
    std::cout << "  Size range: 20MB - 1GB per allocation" << std::endl;
    
    results.min_free_memory = initial_free;
    create_extreme_fragmentation(stream_pool, results.min_free_memory);
    
    size_t after_frag_free = get_free_memory();
    std::cout << "  Memory after fragmentation: " << format_gb(after_frag_free)
              << std::endl;
    std::cout << "  Min free during fragmentation: " 
              << format_gb(results.min_free_memory) << std::endl;
    std::cout << "  Memory used: " 
              << format_gb(initial_free - results.min_free_memory) 
              << std::endl;
    
    // Phase 3: Measure after fragmentation
    std::cout << "\nPhase 3: Measuring after extreme fragmentation..."
              << std::endl;
    results.after_fragmentation_time = measure_multithread_allocation(
        NUM_THREADS, ALLOCS_PER_THREAD, TEST_ROWS, stream_pool);
    results.degradation_pct = 
        ((results.after_fragmentation_time - results.baseline_time) 
         / results.baseline_time) * 100.0;
    
    std::cout << "  After fragmentation: " << std::fixed 
              << std::setprecision(2) << results.after_fragmentation_time 
              << " ms" << std::endl;
    std::cout << "  Degradation: " << std::showpos << std::fixed 
              << std::setprecision(1) << results.degradation_pct 
              << "%" << std::noshowpos << std::endl;
    
    return results;
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "EXTREME Fragmentation Test (8 Threads, up to 2GB allocs)"
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    size_t total_free, total_mem;
    cudaMemGetInfo(&total_free, &total_mem);
    std::cout << "\nGPU Memory: " << format_gb(total_free) << " free / "
              << format_gb(total_mem) << " total" << std::endl;
    
    if (total_free < 10ULL * 1024 * 1024 * 1024) {
        std::cout << "\n⚠️  Warning: Less than 10GB free memory" << std::endl;
        std::cout << "This test may cause OOM. Recommend ≥16GB GPU" 
                  << std::endl;
    }
    
    ExtremeResults pool_res, async_res, arena_res;
    
    // Test Pool
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 1: POOL MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 16ULL * 1024 * 1024 * 1024  // 16GB pool
        );
        rmm::mr::set_current_device_resource(&pool_mr);
        
        pool_res = test_extreme_fragmentation("Pool");
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Test Async
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 2: CUDA ASYNC MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        async_res = test_extreme_fragmentation("Async");
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Test Arena
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 3: ARENA MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource> arena_mr(
            &cuda_mr, 16ULL * 1024 * 1024 * 1024  // 16GB
        );
        rmm::mr::set_current_device_resource(&arena_mr);
        
        arena_res = test_extreme_fragmentation("Arena");
    }
    
    // Summary
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "EXTREME FRAGMENTATION SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    pool_res.print("Pool");
    async_res.print("Async");
    arena_res.print("Arena");
    
    // Analysis
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nDegradation comparison:" << std::endl;
    std::cout << "  Pool:  " << std::showpos << std::fixed 
              << std::setprecision(1) << pool_res.degradation_pct << "%" 
              << std::noshowpos;
    if (std::abs(pool_res.degradation_pct) > 
        std::abs(async_res.degradation_pct) + 5.0) {
        std::cout << "  ⚠️  Worse than Async";
    }
    std::cout << std::endl;
    
    std::cout << "  Async: " << std::showpos 
              << async_res.degradation_pct << "%" << std::noshowpos;
    if (std::abs(async_res.degradation_pct) < 
        std::abs(pool_res.degradation_pct) - 5.0) {
        std::cout << "  ✓ Better than Pool!";
        double advantage = pool_res.degradation_pct - 
                          async_res.degradation_pct;
        std::cout << " (+" << std::fixed << std::setprecision(1)
                  << advantage << "% advantage)";
    }
    std::cout << std::endl;
    
    std::cout << "  Arena: " << std::showpos 
              << arena_res.degradation_pct << "%" << std::noshowpos 
              << std::endl;
    
    // Conclusion
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CONCLUSION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    if (async_res.degradation_pct < pool_res.degradation_pct - 10.0) {
        std::cout << "\n🔥 ASYNC SHOWS SIGNIFICANT ADVANTAGE!" << std::endl;
        std::cout << "   Large-scale fragmentation exposes Pool's weakness"
                  << std::endl;
        std::cout << "   Async's driver management handles it better"
                  << std::endl;
    } else if (async_res.degradation_pct < pool_res.degradation_pct - 5.0) {
        std::cout << "\n✓ Async shows modest advantage" << std::endl;
        std::cout << "  Better than medium-size fragmentation test"
                  << std::endl;
    } else if (std::abs(async_res.degradation_pct - 
                        pool_res.degradation_pct) < 5.0) {
        std::cout << "\n~ Similar behavior under extreme fragmentation"
                  << std::endl;
        std::cout << "  Both handle large allocations reasonably well"
                  << std::endl;
    } else {
        std::cout << "\n✗ Pool handles extreme fragmentation better"
                  << std::endl;
        std::cout << "  Even large allocations don't favor Async"
                  << std::endl;
    }
    
    std::cout << "\nKey insights:" << std::endl;
    std::cout << "1. Allocation sizes: 20MB - 1GB (vs 0.4-20MB in previous)"
              << std::endl;
    std::cout << "2. This creates more realistic large-scale fragmentation"
              << std::endl;
    std::cout << "3. Larger holes should make allocator differences visible"
              << std::endl;
    std::cout << "4. 8 threads = sweet spot for lock contention" << std::endl;
    std::cout << "5. 40 rounds with aggressive free pattern maximizes impact"
              << std::endl;
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

