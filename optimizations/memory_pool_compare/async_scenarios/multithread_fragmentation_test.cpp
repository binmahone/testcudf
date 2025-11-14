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

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <thread>
#include <string>
#include <barrier>
#include <atomic>

/**
 * Multi-threaded fragmentation test
 * 
 * Hypothesis: Multi-threaded concurrent allocation/deallocation creates 
 * more severe fragmentation, and Async's stream-ordered allocation might 
 * handle this better than Pool's global lock.
 */

std::unique_ptr<cudf::column> create_column(
    size_t rows, rmm::cuda_stream_view stream) {
    std::vector<int32_t> data(rows, 42);
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t), stream);
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), rmm::device_buffer{}, 0
    );
}

struct ThreadedPerfResults {
    double baseline_time;
    double after_fragmentation_time;
    double long_term_time;
    double degradation_pct;
    
    void print(const std::string& name) const {
        std::cout << "  " << std::setw(15) << std::left << name << ":" 
                  << std::endl;
        std::cout << "    Baseline (clean):   " << std::fixed 
                  << std::setprecision(2) << baseline_time << " ms" 
                  << std::endl;
        std::cout << "    After fragmentation: " << after_fragmentation_time 
                  << " ms";
        
        if (degradation_pct > 10.0) {
            std::cout << "  ⚠️  DEGRADED " << std::setprecision(1)
                      << degradation_pct << "%";
        } else if (degradation_pct < -5.0) {
            std::cout << "  ✓ IMPROVED " << std::setprecision(1)
                      << (-degradation_pct) << "%";
        } else {
            std::cout << "  ✓ Stable (" << std::showpos 
                      << std::setprecision(1) << degradation_pct 
                      << "%)" << std::noshowpos;
        }
        std::cout << std::endl;
        
        double long_deg = ((long_term_time - baseline_time) / baseline_time) 
                         * 100.0;
        std::cout << "    Long-term:          " << std::fixed 
                  << std::setprecision(2) << long_term_time << " ms";
        
        if (long_deg > 10.0) {
            std::cout << "  ⚠️  DEGRADED " << std::setprecision(1)
                      << long_deg << "%";
        } else {
            std::cout << "  ✓ Stable (" << std::showpos 
                      << std::setprecision(1) << long_deg 
                      << "%)" << std::noshowpos;
        }
        std::cout << std::endl;
    }
};

double measure_multithread_allocation(size_t num_threads, size_t allocs_per_thread,
                                       size_t rows, rmm::cuda_stream_pool& stream_pool) {
    std::barrier barrier(num_threads);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, t, stream]() {
            barrier.arrive_and_wait();  // Sync start
            
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

void create_multithread_fragmentation(size_t num_threads, size_t rounds,
                                       rmm::cuda_stream_pool& stream_pool) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> size_dist(100000, 5000000);
    
    for (size_t round = 0; round < rounds; ++round) {
        std::vector<std::thread> threads;
        std::vector<std::vector<std::unique_ptr<cudf::column>>> per_thread_cols(
            num_threads);
        
        // Phase 1: Each thread allocates varying sizes
        for (size_t t = 0; t < num_threads; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                for (size_t i = 0; i < 20; ++i) {
                    size_t rows = size_dist(gen);
                    per_thread_cols[t].push_back(create_column(rows, stream));
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();
        
        // Phase 2: Each thread randomly frees some (creates holes)
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                auto& cols = per_thread_cols[t];
                std::shuffle(cols.begin(), cols.end(), gen);
                if (cols.size() > 5) {
                    cols.resize(5);  // Keep only 5
                }
            });
        }
        for (auto& th : threads) th.join();
    }
}

ThreadedPerfResults test_multithread_fragmentation(
    const std::string& name, size_t num_threads) {
    
    std::cout << "\nTesting: " << name << " with " << num_threads 
              << " threads" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    rmm::cuda_stream_pool stream_pool(num_threads);
    ThreadedPerfResults results;
    
    constexpr size_t ALLOCS_PER_THREAD = 25;
    constexpr size_t ROWS = 1000000;
    
    // Phase 1: Baseline (clean state)
    std::cout << "Phase 1: Baseline measurement..." << std::endl;
    results.baseline_time = measure_multithread_allocation(
        num_threads, ALLOCS_PER_THREAD, ROWS, stream_pool);
    std::cout << "  Baseline: " << results.baseline_time 
              << " ms (" << num_threads << " threads × " 
              << ALLOCS_PER_THREAD << " allocs)" << std::endl;
    
    // Phase 2: Create multi-threaded fragmentation
    std::cout << "Phase 2: Creating fragmentation (50 rounds, " 
              << num_threads << " threads)..." << std::endl;
    create_multithread_fragmentation(num_threads, 50, stream_pool);
    std::cout << "  Fragmentation created" << std::endl;
    
    // Phase 3: Measure after fragmentation
    std::cout << "Phase 3: Measuring after fragmentation..." << std::endl;
    results.after_fragmentation_time = measure_multithread_allocation(
        num_threads, ALLOCS_PER_THREAD, ROWS, stream_pool);
    results.degradation_pct = 
        ((results.after_fragmentation_time - results.baseline_time) 
         / results.baseline_time) * 100.0;
    std::cout << "  After fragmentation: " << results.after_fragmentation_time 
              << " ms" << std::endl;
    std::cout << "  Change: " << std::showpos << std::fixed 
              << std::setprecision(1) << results.degradation_pct 
              << "%" << std::noshowpos << std::endl;
    
    // Phase 4: Long-term with continued fragmentation
    std::cout << "Phase 4: Long-term stress (100 more rounds)..." 
              << std::endl;
    create_multithread_fragmentation(num_threads, 100, stream_pool);
    
    results.long_term_time = measure_multithread_allocation(
        num_threads, ALLOCS_PER_THREAD, ROWS, stream_pool);
    double long_deg = ((results.long_term_time - results.baseline_time) 
                       / results.baseline_time) * 100.0;
    std::cout << "  Long-term: " << results.long_term_time << " ms" 
              << std::endl;
    std::cout << "  Total change: " << std::showpos << std::fixed 
              << std::setprecision(1) << long_deg << "%" << std::noshowpos 
              << std::endl;
    
    return results;
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Multi-Threaded Fragmentation Performance Test" << std::endl;
    std::cout << "Does concurrent allocation expose fragmentation issues?" 
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    struct AllResults {
        ThreadedPerfResults threads_4;
        ThreadedPerfResults threads_8;
        ThreadedPerfResults threads_16;
    };
    
    AllResults pool_res, async_res, arena_res;
    
    // Test Pool with varying thread counts
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 1: POOL MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024
        );
        rmm::mr::set_current_device_resource(&pool_mr);
        
        pool_res.threads_4 = test_multithread_fragmentation("Pool", 4);
        pool_res.threads_8 = test_multithread_fragmentation("Pool", 8);
        pool_res.threads_16 = test_multithread_fragmentation("Pool", 16);
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test Async
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 2: CUDA ASYNC MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        async_res.threads_4 = test_multithread_fragmentation("Async", 4);
        async_res.threads_8 = test_multithread_fragmentation("Async", 8);
        async_res.threads_16 = test_multithread_fragmentation("Async", 16);
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test Arena
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 3: ARENA MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource> arena_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024
        );
        rmm::mr::set_current_device_resource(&arena_mr);
        
        arena_res.threads_4 = test_multithread_fragmentation("Arena", 4);
        arena_res.threads_8 = test_multithread_fragmentation("Arena", 8);
        arena_res.threads_16 = test_multithread_fragmentation("Arena", 16);
    }
    
    // Summary
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "MULTI-THREADED FRAGMENTATION SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto print_comparison = [](const std::string& title,
                               const ThreadedPerfResults& pool,
                               const ThreadedPerfResults& async,
                               const ThreadedPerfResults& arena) {
        std::cout << "\n" << title << ":" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        pool.print("Pool");
        async.print("Async");
        arena.print("Arena");
        
        std::cout << "\n  Short-term degradation comparison:" << std::endl;
        std::cout << "    Pool:  " << std::showpos << std::fixed 
                  << std::setprecision(1) << pool.degradation_pct << "%" 
                  << std::noshowpos;
        if (std::abs(pool.degradation_pct) > 
            std::abs(async.degradation_pct) + 5.0) {
            std::cout << "  ⚠️  Worse than Async";
        }
        std::cout << std::endl;
        
        std::cout << "    Async: " << std::showpos 
                  << async.degradation_pct << "%" << std::noshowpos;
        if (std::abs(async.degradation_pct) < 
            std::abs(pool.degradation_pct) - 5.0) {
            std::cout << "  ✓ Better than Pool!";
        }
        std::cout << std::endl;
        
        std::cout << "    Arena: " << std::showpos 
                  << arena.degradation_pct << "%" << std::noshowpos 
                  << std::endl;
    };
    
    print_comparison("4 Threads", pool_res.threads_4, 
                     async_res.threads_4, arena_res.threads_4);
    print_comparison("8 Threads", pool_res.threads_8, 
                     async_res.threads_8, arena_res.threads_8);
    print_comparison("16 Threads", pool_res.threads_16, 
                     async_res.threads_16, arena_res.threads_16);
    
    // Final analysis
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CONCLUSION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    bool async_advantage_found = false;
    
    // Check if Async shows advantage at any thread count
    if ((pool_res.threads_4.degradation_pct > 
         async_res.threads_4.degradation_pct + 10.0) ||
        (pool_res.threads_8.degradation_pct > 
         async_res.threads_8.degradation_pct + 10.0) ||
        (pool_res.threads_16.degradation_pct > 
         async_res.threads_16.degradation_pct + 10.0)) {
        
        std::cout << "\n✓ ASYNC SHOWS ADVANTAGE in multi-threaded fragmentation!"
                  << std::endl;
        std::cout << "  Pool suffers more degradation under concurrent load"
                  << std::endl;
        std::cout << "  Async's stream-ordered allocation helps!"
                  << std::endl;
        async_advantage_found = true;
    } else if (std::abs(pool_res.threads_4.degradation_pct - 
                        async_res.threads_4.degradation_pct) < 10.0 &&
               std::abs(pool_res.threads_8.degradation_pct - 
                        async_res.threads_8.degradation_pct) < 10.0 &&
               std::abs(pool_res.threads_16.degradation_pct - 
                        async_res.threads_16.degradation_pct) < 10.0) {
        
        std::cout << "\n~ Both Pool and Async show SIMILAR behavior"
                  << std::endl;
        std::cout << "  Multi-threaded fragmentation affects both similarly"
                  << std::endl;
        std::cout << "  Difference < 10% at all thread counts"
                  << std::endl;
    } else {
        std::cout << "\n✗ Pool actually handles multi-threaded fragmentation "
                  << "BETTER" << std::endl;
        std::cout << "  Pool's global management may be more efficient"
                  << std::endl;
    }
    
    std::cout << "\nKey findings:" << std::endl;
    std::cout << "1. Thread scaling:" << std::endl;
    std::cout << "   - Check if degradation increases with more threads"
              << std::endl;
    std::cout << "2. Async advantage (if any):" << std::endl;
    std::cout << "   - Stream-ordered allocation reduces lock contention"
              << std::endl;
    std::cout << "3. Production recommendation:" << std::endl;
    if (async_advantage_found) {
        std::cout << "   - Consider Async for high-thread-count workloads"
                  << std::endl;
    } else {
        std::cout << "   - Pool remains the best choice even with many threads"
                  << std::endl;
    }
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

