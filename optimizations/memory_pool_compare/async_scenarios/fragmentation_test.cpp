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

/**
 * Test memory fragmentation behavior across different allocators
 * Focus: Memory efficiency, not just time performance
 */

size_t get_free_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_idx < 3) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
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

struct MemoryStats {
    size_t initial_free;
    size_t min_free;
    size_t final_free;
    size_t peak_used;
    
    void print(const std::string& name) const {
        std::cout << "  " << std::setw(15) << std::left << name << ":" 
                  << std::endl;
        std::cout << "    Initial free: " << format_bytes(initial_free) 
                  << std::endl;
        std::cout << "    Min free:     " << format_bytes(min_free)
                  << "  (peak used: " << format_bytes(peak_used) << ")"
                  << std::endl;
        std::cout << "    Final free:   " << format_bytes(final_free);
        
        size_t leaked = initial_free - final_free;
        if (leaked > 1024 * 1024) {  // > 1MB
            std::cout << "  ⚠️  LEAKED: " << format_bytes(leaked);
        } else {
            std::cout << "  ✓ No leak";
        }
        std::cout << std::endl;
        
        size_t fragmentation = peak_used - (initial_free - min_free);
        if (fragmentation > peak_used * 0.1) {  // > 10% overhead
            std::cout << "    Fragmentation: ~" 
                      << format_bytes(fragmentation)
                      << " (" << std::fixed << std::setprecision(1)
                      << (fragmentation * 100.0 / peak_used) << "% overhead)"
                      << std::endl;
        }
    }
};

// Test 1: Long-running with varying sizes (fragmentation accumulation)
MemoryStats test_long_running_fragmentation(const std::string& mr_name) {
    std::cout << "\n[1] Long-Running Varying Size Allocations" << std::endl;
    std::cout << "    (Tests fragmentation accumulation over time)" 
              << std::endl;
    
    MemoryStats stats;
    stats.initial_free = get_free_memory();
    stats.min_free = stats.initial_free;
    
    rmm::cuda_stream_pool stream_pool(4);
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> size_dist(100000, 5000000);
    
    // Simulate long-running workload with varying sizes
    for (int round = 0; round < 50; ++round) {
        std::vector<std::unique_ptr<cudf::column>> live_columns;
        
        // Allocate phase: varying sizes
        for (int i = 0; i < 20; ++i) {
            size_t rows = size_dist(gen);
            auto stream = stream_pool.get_stream();
            live_columns.push_back(create_column(rows, stream));
            
            size_t current_free = get_free_memory();
            if (current_free < stats.min_free) {
                stats.min_free = current_free;
            }
        }
        
        // Deallocate phase: random pattern (creates holes)
        std::shuffle(live_columns.begin(), live_columns.end(), gen);
        for (int i = 0; i < 10; ++i) {
            if (!live_columns.empty()) {
                live_columns.pop_back();
            }
        }
        
        // Keep some alive (memory pressure)
        if (live_columns.size() > 15) {
            live_columns.resize(5);
        }
    }
    
    cudaDeviceSynchronize();
    stats.final_free = get_free_memory();
    stats.peak_used = stats.initial_free - stats.min_free;
    
    stats.print(mr_name);
    return stats;
}

// Test 2: Allocation/deallocation with holes (fragmentation creation)
MemoryStats test_fragmentation_holes(const std::string& mr_name) {
    std::cout << "\n[2] Allocation/Deallocation Creating Holes" << std::endl;
    std::cout << "    (Tests allocator's ability to reuse freed blocks)" 
              << std::endl;
    
    MemoryStats stats;
    stats.initial_free = get_free_memory();
    stats.min_free = stats.initial_free;
    
    rmm::cuda_stream_pool stream_pool(4);
    constexpr size_t SMALL = 500000;
    constexpr size_t LARGE = 5000000;
    
    for (int round = 0; round < 30; ++round) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        
        // Allocate: small, large, small, large pattern
        auto stream = stream_pool.get_stream();
        for (int i = 0; i < 10; ++i) {
            columns.push_back(create_column(SMALL, stream));
            columns.push_back(create_column(LARGE, stream));
        }
        
        size_t current_free = get_free_memory();
        if (current_free < stats.min_free) {
            stats.min_free = current_free;
        }
        
        // Free all small ones (creates holes)
        for (size_t i = 0; i < columns.size(); ) {
            columns.erase(columns.begin() + i);
            i++;  // Skip next (large)
        }
        
        // Try to allocate medium size (tests if holes can be reused)
        for (int i = 0; i < 5; ++i) {
            columns.push_back(create_column(1000000, stream));
        }
        
        current_free = get_free_memory();
        if (current_free < stats.min_free) {
            stats.min_free = current_free;
        }
    }
    
    cudaDeviceSynchronize();
    stats.final_free = get_free_memory();
    stats.peak_used = stats.initial_free - stats.min_free;
    
    stats.print(mr_name);
    return stats;
}

// Test 3: Multi-stream allocation patterns (cross-stream fragmentation)
MemoryStats test_multistream_fragmentation(const std::string& mr_name) {
    std::cout << "\n[3] Multi-Stream Fragmentation" << std::endl;
    std::cout << "    (Tests cross-stream memory reuse efficiency)" 
              << std::endl;
    
    MemoryStats stats;
    stats.initial_free = get_free_memory();
    stats.min_free = stats.initial_free;
    
    rmm::cuda_stream_pool stream_pool(8);
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> size_dist(200000, 2000000);
    
    // Multiple rounds of multi-stream allocations
    for (int round = 0; round < 40; ++round) {
        std::vector<std::thread> threads;
        std::vector<std::vector<std::unique_ptr<cudf::column>>> per_thread;
        per_thread.resize(8);
        
        // Each stream allocates different sizes
        for (int t = 0; t < 8; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                for (int i = 0; i < 5; ++i) {
                    size_t rows = size_dist(gen);
                    per_thread[t].push_back(create_column(rows, stream));
                }
            });
        }
        
        for (auto& th : threads) th.join();
        
        size_t current_free = get_free_memory();
        if (current_free < stats.min_free) {
            stats.min_free = current_free;
        }
        
        // Free half randomly (creates fragmentation across streams)
        for (auto& vec : per_thread) {
            if (vec.size() > 2) {
                vec.resize(2);
            }
        }
    }
    
    cudaDeviceSynchronize();
    stats.final_free = get_free_memory();
    stats.peak_used = stats.initial_free - stats.min_free;
    
    stats.print(mr_name);
    return stats;
}

// Test 4: Steady state memory efficiency
MemoryStats test_steady_state_efficiency(const std::string& mr_name) {
    std::cout << "\n[4] Steady State Memory Efficiency" << std::endl;
    std::cout << "    (Tests stable workload memory usage)" << std::endl;
    
    MemoryStats stats;
    stats.initial_free = get_free_memory();
    stats.min_free = stats.initial_free;
    
    rmm::cuda_stream_pool stream_pool(4);
    constexpr size_t ROWS = 1000000;
    
    // Steady state: same pattern repeated
    for (int round = 0; round < 100; ++round) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        auto stream = stream_pool.get_stream();
        
        for (int i = 0; i < 10; ++i) {
            columns.push_back(create_column(ROWS, stream));
        }
        
        size_t current_free = get_free_memory();
        if (current_free < stats.min_free) {
            stats.min_free = current_free;
        }
        
        // Process and release
        columns.clear();
    }
    
    cudaDeviceSynchronize();
    stats.final_free = get_free_memory();
    stats.peak_used = stats.initial_free - stats.min_free;
    
    stats.print(mr_name);
    return stats;
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Memory Fragmentation Analysis" << std::endl;
    std::cout << "Testing: Pool vs Async vs Arena" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    struct TestResults {
        MemoryStats long_running;
        MemoryStats holes;
        MemoryStats multistream;
        MemoryStats steady_state;
    };
    
    TestResults pool_res, async_res, arena_res;
    
    // Test Pool
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: Pool Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<
            rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024);
        rmm::mr::set_current_device_resource(&pool_mr);
        
        pool_res.long_running = test_long_running_fragmentation("Pool");
        pool_res.holes = test_fragmentation_holes("Pool");
        pool_res.multistream = test_multistream_fragmentation("Pool");
        pool_res.steady_state = test_steady_state_efficiency("Pool");
    }
    
    // Test Async
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: CUDA Async Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        async_res.long_running = test_long_running_fragmentation("Async");
        async_res.holes = test_fragmentation_holes("Async");
        async_res.multistream = test_multistream_fragmentation("Async");
        async_res.steady_state = test_steady_state_efficiency("Async");
    }
    
    // Test Arena
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: Arena Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::arena_memory_resource<
            rmm::mr::cuda_memory_resource> arena_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024);
        rmm::mr::set_current_device_resource(&arena_mr);
        
        arena_res.long_running = test_long_running_fragmentation("Arena");
        arena_res.holes = test_fragmentation_holes("Arena");
        arena_res.multistream = test_multistream_fragmentation("Arena");
        arena_res.steady_state = test_steady_state_efficiency("Arena");
    }
    
    // Summary
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "FRAGMENTATION ANALYSIS SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto compare_fragmentation = [](const std::string& test,
                                     const MemoryStats& pool,
                                     const MemoryStats& async,
                                     const MemoryStats& arena) {
        std::cout << "\n" << test << ":" << std::endl;
        
        auto calc_overhead = [](const MemoryStats& s) {
            return s.peak_used > 0 ? 
                   (s.initial_free - s.min_free - s.peak_used) * 100.0 
                   / s.peak_used : 0.0;
        };
        
        double pool_overhead = calc_overhead(pool);
        double async_overhead = calc_overhead(async);
        double arena_overhead = calc_overhead(arena);
        
        std::cout << "  Peak memory used:" << std::endl;
        std::cout << "    Pool:  " << format_bytes(pool.peak_used);
        if (pool.peak_used <= async.peak_used && 
            pool.peak_used <= arena.peak_used) {
            std::cout << "  ← Most efficient";
        }
        std::cout << std::endl;
        
        std::cout << "    Async: " << format_bytes(async.peak_used);
        if (async.peak_used < pool.peak_used && 
            async.peak_used <= arena.peak_used) {
            std::cout << "  ← Most efficient";
            double saving = (pool.peak_used - async.peak_used) * 100.0 
                           / pool.peak_used;
            std::cout << " (saves " << std::fixed << std::setprecision(1)
                      << saving << "%)";
        }
        std::cout << std::endl;
        
        std::cout << "    Arena: " << format_bytes(arena.peak_used);
        if (arena.peak_used < pool.peak_used && 
            arena.peak_used < async.peak_used) {
            std::cout << "  ← Most efficient";
        }
        std::cout << std::endl;
        
        size_t pool_leak = pool.initial_free - pool.final_free;
        size_t async_leak = async.initial_free - async.final_free;
        size_t arena_leak = arena.initial_free - arena.final_free;
        
        if (pool_leak > 1024*1024 || async_leak > 1024*1024 || 
            arena_leak > 1024*1024) {
            std::cout << "  Memory leak detected:" << std::endl;
            if (pool_leak > 1024*1024)
                std::cout << "    Pool:  " << format_bytes(pool_leak) 
                          << std::endl;
            if (async_leak > 1024*1024)
                std::cout << "    Async: " << format_bytes(async_leak) 
                          << std::endl;
            if (arena_leak > 1024*1024)
                std::cout << "    Arena: " << format_bytes(arena_leak) 
                          << std::endl;
        }
    };
    
    compare_fragmentation("[1] Long-running fragmentation",
                          pool_res.long_running,
                          async_res.long_running,
                          arena_res.long_running);
    
    compare_fragmentation("[2] Holes fragmentation",
                          pool_res.holes,
                          async_res.holes,
                          arena_res.holes);
    
    compare_fragmentation("[3] Multi-stream fragmentation",
                          pool_res.multistream,
                          async_res.multistream,
                          arena_res.multistream);
    
    compare_fragmentation("[4] Steady state efficiency",
                          pool_res.steady_state,
                          async_res.steady_state,
                          arena_res.steady_state);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "KEY FINDINGS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\n1. Memory Efficiency:" << std::endl;
    std::cout << "   - Check if Async uses less peak memory" << std::endl;
    std::cout << "   - Pool pre-allocates, may have overhead" << std::endl;
    std::cout << "   - Async allocates on-demand" << std::endl;
    
    std::cout << "\n2. Fragmentation:" << std::endl;
    std::cout << "   - Driver-managed (Async) may handle better" << std::endl;
    std::cout << "   - Pool may accumulate holes over time" << std::endl;
    std::cout << "   - Arena behavior depends on usage pattern" << std::endl;
    
    std::cout << "\n3. Cross-Stream Reuse:" << std::endl;
    std::cout << "   - Async stream-ordered may reuse better" << std::endl;
    std::cout << "   - Pool/Arena have global free lists" << std::endl;
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

