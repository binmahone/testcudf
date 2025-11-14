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

/**
 * Test if memory fragmentation degrades allocation performance over time
 * 
 * Hypothesis: After creating fragmentation, Pool might slow down as it 
 * searches through free lists with many holes. Async might handle this 
 * better with driver-level management.
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

double measure_allocation_speed(size_t num_allocs, size_t rows,
                                 rmm::cuda_stream_view stream) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (size_t i = 0; i < num_allocs; ++i) {
        columns.push_back(create_column(rows, stream));
    }
    stream.synchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count() / 1000.0;
}

struct PerformanceResults {
    double initial_speed;
    double after_fragmentation;
    double final_speed;
    double degradation_percent;
    
    void print(const std::string& name) const {
        std::cout << "  " << std::setw(15) << std::left << name << ": "
                  << std::endl;
        std::cout << "    Initial:            " << std::fixed 
                  << std::setprecision(2) << initial_speed << " ms" 
                  << std::endl;
        std::cout << "    After fragmentation: " << after_fragmentation 
                  << " ms";
        
        if (degradation_percent > 5.0) {
            std::cout << "  ⚠️  SLOWER by " << std::setprecision(1)
                      << degradation_percent << "%";
        } else if (degradation_percent < -5.0) {
            std::cout << "  ✓ FASTER by " << std::setprecision(1)
                      << (-degradation_percent) << "%";
        } else {
            std::cout << "  ✓ Stable";
        }
        std::cout << std::endl;
        
        std::cout << "    Final (long-term):  " << std::fixed 
                  << std::setprecision(2) << final_speed << " ms";
        
        double final_deg = ((final_speed - initial_speed) / initial_speed) 
                          * 100.0;
        if (final_deg > 5.0) {
            std::cout << "  ⚠️  Degraded by " << std::setprecision(1)
                      << final_deg << "%";
        } else {
            std::cout << "  ✓ Stable";
        }
        std::cout << std::endl;
    }
};

PerformanceResults test_fragmentation_performance(const std::string& name) {
    std::cout << "\nTesting: " << name << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    rmm::cuda_stream_pool stream_pool(4);
    auto stream = stream_pool.get_stream();
    PerformanceResults results;
    
    // Phase 1: Baseline - measure initial allocation speed
    std::cout << "Phase 1: Measuring baseline allocation speed..." 
              << std::endl;
    results.initial_speed = measure_allocation_speed(100, 1000000, stream);
    std::cout << "  Baseline: " << results.initial_speed << " ms for 100 allocs"
              << std::endl;
    
    // Phase 2: Create fragmentation
    std::cout << "Phase 2: Creating fragmentation..." << std::endl;
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> size_dist(100000, 5000000);
    
    for (int round = 0; round < 50; ++round) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        
        // Allocate varying sizes
        for (int i = 0; i < 30; ++i) {
            size_t rows = size_dist(gen);
            columns.push_back(create_column(rows, stream));
        }
        
        // Free in random order (creates holes)
        std::shuffle(columns.begin(), columns.end(), gen);
        columns.resize(columns.size() / 3);  // Keep 1/3, free 2/3
    }
    stream.synchronize();
    std::cout << "  Completed 50 rounds of fragmentation" << std::endl;
    
    // Phase 3: Measure allocation speed after fragmentation
    std::cout << "Phase 3: Measuring speed after fragmentation..." 
              << std::endl;
    results.after_fragmentation = measure_allocation_speed(
        100, 1000000, stream);
    results.degradation_percent = 
        ((results.after_fragmentation - results.initial_speed) 
         / results.initial_speed) * 100.0;
    std::cout << "  After fragmentation: " << results.after_fragmentation 
              << " ms for 100 allocs" << std::endl;
    std::cout << "  Change: " << std::showpos << std::fixed 
              << std::setprecision(1) << results.degradation_percent 
              << "%" << std::noshowpos << std::endl;
    
    // Phase 4: Long-term stability test
    std::cout << "Phase 4: Long-term stability (100 more rounds)..." 
              << std::endl;
    
    for (int round = 0; round < 100; ++round) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        
        for (int i = 0; i < 20; ++i) {
            size_t rows = size_dist(gen);
            columns.push_back(create_column(rows, stream));
        }
        
        std::shuffle(columns.begin(), columns.end(), gen);
        columns.resize(5);
    }
    stream.synchronize();
    
    results.final_speed = measure_allocation_speed(100, 1000000, stream);
    std::cout << "  Long-term: " << results.final_speed 
              << " ms for 100 allocs" << std::endl;
    
    double final_change = ((results.final_speed - results.initial_speed) 
                           / results.initial_speed) * 100.0;
    std::cout << "  Total change from baseline: " << std::showpos 
              << std::fixed << std::setprecision(1) << final_change 
              << "%" << std::noshowpos << std::endl;
    
    return results;
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Fragmentation Performance Impact Test" << std::endl;
    std::cout << "Does fragmentation slow down future allocations?" 
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    PerformanceResults pool_res, async_res, arena_res;
    
    // Test Pool
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 1: POOL MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024
        );
        rmm::mr::set_current_device_resource(&pool_mr);
        
        pool_res = test_fragmentation_performance("Pool");
    }
    
    // Small pause to let GPU settle
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test Async
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 2: CUDA ASYNC MEMORY RESOURCE" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        async_res = test_fragmentation_performance("Async");
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
        
        arena_res = test_fragmentation_performance("Arena");
    }
    
    // Summary
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "PERFORMANCE DEGRADATION SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    pool_res.print("Pool");
    std::cout << std::endl;
    async_res.print("Async");
    std::cout << std::endl;
    arena_res.print("Arena");
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nShort-term degradation (after initial fragmentation):" 
              << std::endl;
    std::cout << "  Pool:  " << std::showpos << std::fixed 
              << std::setprecision(1) << pool_res.degradation_percent 
              << "%" << std::noshowpos;
    if (std::abs(pool_res.degradation_percent) > 
        std::abs(async_res.degradation_percent) &&
        std::abs(pool_res.degradation_percent) > 
        std::abs(arena_res.degradation_percent)) {
        std::cout << "  ← Most affected";
    }
    std::cout << std::endl;
    
    std::cout << "  Async: " << std::showpos << async_res.degradation_percent 
              << "%" << std::noshowpos;
    if (std::abs(async_res.degradation_percent) < 
        std::abs(pool_res.degradation_percent) &&
        std::abs(async_res.degradation_percent) < 
        std::abs(arena_res.degradation_percent)) {
        std::cout << "  ← Most stable";
    }
    std::cout << std::endl;
    
    std::cout << "  Arena: " << std::showpos 
              << arena_res.degradation_percent << "%" << std::noshowpos;
    if (std::abs(arena_res.degradation_percent) < 
        std::abs(pool_res.degradation_percent) &&
        std::abs(arena_res.degradation_percent) < 
        std::abs(async_res.degradation_percent)) {
        std::cout << "  ← Most stable";
    }
    std::cout << std::endl;
    
    double pool_final_deg = ((pool_res.final_speed - pool_res.initial_speed) 
                             / pool_res.initial_speed) * 100.0;
    double async_final_deg = ((async_res.final_speed - async_res.initial_speed) 
                              / async_res.initial_speed) * 100.0;
    double arena_final_deg = ((arena_res.final_speed - arena_res.initial_speed) 
                              / arena_res.initial_speed) * 100.0;
    
    std::cout << "\nLong-term degradation (after 150 fragmentation rounds):" 
              << std::endl;
    std::cout << "  Pool:  " << std::showpos << pool_final_deg << "%" 
              << std::noshowpos << std::endl;
    std::cout << "  Async: " << std::showpos << async_final_deg << "%" 
              << std::noshowpos << std::endl;
    std::cout << "  Arena: " << std::showpos << arena_final_deg << "%" 
              << std::noshowpos << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CONCLUSION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    bool async_advantage = false;
    
    if (async_res.degradation_percent < pool_res.degradation_percent - 5.0) {
        std::cout << "\n✓ Async shows ADVANTAGE in fragmentation resilience!"
                  << std::endl;
        std::cout << "  Async degrades " << std::fixed << std::setprecision(1)
                  << (pool_res.degradation_percent - 
                      async_res.degradation_percent)
                  << "% less than Pool" << std::endl;
        async_advantage = true;
    } else if (std::abs(async_res.degradation_percent - 
                        pool_res.degradation_percent) < 5.0) {
        std::cout << "\n~ Async and Pool show SIMILAR fragmentation behavior"
                  << std::endl;
        std::cout << "  Both degrade within 5% of each other" << std::endl;
    } else {
        std::cout << "\n✗ Pool actually handles fragmentation BETTER than Async"
                  << std::endl;
        std::cout << "  Pool degrades " << std::fixed << std::setprecision(1)
                  << (async_res.degradation_percent - 
                      pool_res.degradation_percent)
                  << "% less than Async" << std::endl;
    }
    
    if (!async_advantage && 
        arena_res.degradation_percent < async_res.degradation_percent - 5.0) {
        std::cout << "\n  Note: Arena shows best fragmentation resilience"
                  << std::endl;
    }
    
    std::cout << "\nKey insight:" << std::endl;
    std::cout << "  - If degradation < 5%: Fragmentation is NOT a real issue"
              << std::endl;
    std::cout << "  - If degradation > 10%: Fragmentation DOES impact performance"
              << std::endl;
    std::cout << "  - Async advantage (if any) is in maintaining stable performance"
              << std::endl;
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

