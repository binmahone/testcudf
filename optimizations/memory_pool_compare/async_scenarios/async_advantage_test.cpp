#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/replace.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
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
 * Test scenarios where cuda_async_memory_resource should excel:
 * - Stream-ordered allocation/deallocation
 * - Dynamic varying sizes
 * - High concurrency without global synchronization
 * - Cross-stream memory reuse
 */

std::unique_ptr<cudf::column> create_column(
    size_t rows, rmm::cuda_stream_view stream) {
    std::vector<int32_t> data(rows);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(1, 1000);
    for (size_t i = 0; i < rows; ++i) data[i] = dist(gen);
    
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t), stream);
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), rmm::device_buffer{}, 0
    );
}

// Scenario 1: Extreme size variation (10x-1000x range)
double test_extreme_size_variation(const std::string& mr_name) {
    std::cout << "\n[1] Extreme Size Variation (100K to 10M rows)" 
              << std::endl;
    
    rmm::cuda_stream_pool stream_pool(8);
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 8; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, t, stream]() {
            std::mt19937 gen(t * 1000);
            std::uniform_int_distribution<size_t> size_dist(
                100000, 10000000);  // 100K to 10M - extreme variation
            
            for (size_t i = 0; i < 50; ++i) {
                size_t rows = size_dist(gen);
                auto col = create_column(rows, stream);
                auto doubled = cudf::cast(col->view(),
                    cudf::data_type{cudf::type_id::FLOAT64}, stream);
            }
            stream.synchronize();
        });
    }
    
    for (auto& th : threads) th.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    std::cout << "  " << std::setw(15) << std::left << mr_name 
              << ": " << std::fixed << std::setprecision(1) 
              << elapsed << " ms" << std::endl;
    
    return elapsed;
}

// Scenario 2: Very high stream concurrency (32 streams)
double test_very_high_concurrency(const std::string& mr_name) {
    std::cout << "\n[2] Very High Concurrency (32 concurrent streams)" 
              << std::endl;
    
    constexpr size_t NUM_STREAMS = 32;
    rmm::cuda_stream_pool stream_pool(NUM_STREAMS);
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t t = 0; t < NUM_STREAMS; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, stream]() {
            constexpr size_t ROWS = 500000;
            
            for (int i = 0; i < 30; ++i) {
                auto col = create_column(ROWS, stream);
                auto result = cudf::cast(col->view(),
                    cudf::data_type{cudf::type_id::FLOAT64}, stream);
            }
            stream.synchronize();
        });
    }
    
    for (auto& th : threads) th.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    std::cout << "  " << std::setw(15) << std::left << mr_name 
              << ": " << elapsed << " ms" << std::endl;
    
    return elapsed;
}

// Scenario 3: Rapid allocation/deallocation churn
double test_rapid_churn(const std::string& mr_name) {
    std::cout << "\n[3] Rapid Alloc/Dealloc Churn" << std::endl;
    
    rmm::cuda_stream_pool stream_pool(8);
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 8; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, stream]() {
            constexpr size_t ROWS = 500000;
            
            // Rapid creation and destruction
            for (size_t i = 0; i < 100; ++i) {
                auto col1 = create_column(ROWS, stream);
                auto col2 = create_column(ROWS, stream);
                auto result = cudf::binary_operation(
                    col1->view(), col2->view(),
                    cudf::binary_operator::ADD,
                    cudf::data_type{cudf::type_id::INT32}, stream
                );
                // All immediately go out of scope
            }
            stream.synchronize();
        });
    }
    
    for (auto& th : threads) th.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    std::cout << "  " << std::setw(15) << std::left << mr_name 
              << ": " << elapsed << " ms" << std::endl;
    
    return elapsed;
}

// Scenario 4: Interleaved operations on different streams
double test_interleaved_streams(const std::string& mr_name) {
    std::cout << "\n[4] Interleaved Multi-Stream Operations" << std::endl;
    
    rmm::cuda_stream_pool stream_pool(16);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int round = 0; round < 5; ++round) {
        for (int t = 0; t < 16; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, stream]() {
                std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<size_t> size_dist(
                    200000, 2000000);
                
                for (int i = 0; i < 10; ++i) {
                    size_t rows = size_dist(gen);
                    auto col = create_column(rows, stream);
                    auto result = cudf::cast(col->view(),
                        cudf::data_type{cudf::type_id::FLOAT64}, stream);
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    std::cout << "  " << std::setw(15) << std::left << mr_name 
              << ": " << elapsed << " ms" << std::endl;
    
    return elapsed;
}

// Scenario 5: Small frequent allocations
double test_small_frequent_allocs(const std::string& mr_name) {
    std::cout << "\n[5] Small Frequent Allocations" << std::endl;
    
    rmm::cuda_stream_pool stream_pool(8);
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 8; ++t) {
        auto stream = stream_pool.get_stream();
        threads.emplace_back([&, stream]() {
            // Many small allocations
            for (size_t i = 0; i < 200; ++i) {
                auto col = create_column(50000, stream);  // Small: 50K rows
                auto result = cudf::cast(col->view(),
                    cudf::data_type{cudf::type_id::FLOAT64}, stream);
            }
            stream.synchronize();
        });
    }
    
    for (auto& th : threads) th.join();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    std::cout << "  " << std::setw(15) << std::left << mr_name 
              << ": " << elapsed << " ms" << std::endl;
    
    return elapsed;
}

struct TestResults {
    double extreme_size_var;
    double high_concurrency;
    double rapid_churn;
    double interleaved;
    double small_frequent;
};

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CUDA Async Pool - Where Should It Excel?" << std::endl;
    std::cout << "Testing stream-ordered allocation advantages" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    TestResults pool_res, async_res, arena_res;
    
    // Test Pool
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: Pool Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<
            rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 12ULL * 1024 * 1024 * 1024);  // 12GB for large allocs
        rmm::mr::set_current_device_resource(&pool_mr);
        
        pool_res.extreme_size_var = test_extreme_size_variation("Pool");
        pool_res.high_concurrency = test_very_high_concurrency("Pool");
        pool_res.rapid_churn = test_rapid_churn("Pool");
        pool_res.interleaved = test_interleaved_streams("Pool");
        pool_res.small_frequent = test_small_frequent_allocs("Pool");
    }
    
    // Test Async
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: CUDA Async Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        async_res.extreme_size_var = test_extreme_size_variation("Async");
        async_res.high_concurrency = test_very_high_concurrency("Async");
        async_res.rapid_churn = test_rapid_churn("Async");
        async_res.interleaved = test_interleaved_streams("Async");
        async_res.small_frequent = test_small_frequent_allocs("Async");
    }
    
    // Test Arena
    {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Testing: Arena Memory Resource" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::arena_memory_resource<
            rmm::mr::cuda_memory_resource> arena_mr(
            &cuda_mr, 12ULL * 1024 * 1024 * 1024);
        rmm::mr::set_current_device_resource(&arena_mr);
        
        arena_res.extreme_size_var = test_extreme_size_variation("Arena");
        arena_res.high_concurrency = test_very_high_concurrency("Arena");
        arena_res.rapid_churn = test_rapid_churn("Arena");
        arena_res.interleaved = test_interleaved_streams("Arena");
        arena_res.small_frequent = test_small_frequent_allocs("Arena");
    }
    
    // Summary
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "RESULTS SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    auto print_result = [](const std::string& test, 
                           double pool, double async, double arena) {
        std::cout << "\n" << test << ":" << std::endl;
        std::cout << "  Pool  : " << std::setw(8) << std::fixed 
                  << std::setprecision(1) << pool << " ms";
        
        double pool_vs_async = ((pool - async) / pool * 100.0);
        double pool_vs_arena = ((pool - arena) / pool * 100.0);
        
        if (pool < async && pool < arena) {
            std::cout << "  <- FASTEST";
        }
        std::cout << std::endl;
        
        std::cout << "  Async : " << std::setw(8) << async << " ms";
        if (async < pool && async < arena) {
            std::cout << "  <- FASTEST";
        }
        if (pool_vs_async > 1.0) {
            std::cout << "  (+" << std::setprecision(1) 
                      << pool_vs_async << "% vs Pool)";
        } else if (pool_vs_async < -1.0) {
            std::cout << "  (" << std::setprecision(1) 
                      << pool_vs_async << "% vs Pool)";
        }
        std::cout << std::endl;
        
        std::cout << "  Arena : " << std::setw(8) << arena << " ms";
        if (arena < pool && arena < async) {
            std::cout << "  <- FASTEST";
        }
        std::cout << std::endl;
    };
    
    print_result("[1] Extreme size variation (100K-10M)",
                 pool_res.extreme_size_var, 
                 async_res.extreme_size_var, 
                 arena_res.extreme_size_var);
    
    print_result("[2] Very high concurrency (32 streams)",
                 pool_res.high_concurrency, 
                 async_res.high_concurrency, 
                 arena_res.high_concurrency);
    
    print_result("[3] Rapid alloc/dealloc churn",
                 pool_res.rapid_churn, 
                 async_res.rapid_churn, 
                 arena_res.rapid_churn);
    
    print_result("[4] Interleaved multi-stream ops",
                 pool_res.interleaved, 
                 async_res.interleaved, 
                 arena_res.interleaved);
    
    print_result("[5] Small frequent allocations",
                 pool_res.small_frequent, 
                 async_res.small_frequent, 
                 arena_res.small_frequent);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nAsync Pool advantages come from:" << std::endl;
    std::cout << "  1. Stream-ordered allocation (no global locks)" 
              << std::endl;
    std::cout << "  2. Driver-managed pooling across streams" 
              << std::endl;
    std::cout << "  3. Automatic size adaptation" << std::endl;
    std::cout << "\nIn practice:" << std::endl;
    std::cout << "  - Differences are typically < 5% for most workloads" 
              << std::endl;
    std::cout << "  - Pool pre-allocation is very effective" << std::endl;
    std::cout << "  - Modern GPU memory subsystem is already efficient" 
              << std::endl;
    std::cout << "\nRecommendation:" << std::endl;
    std::cout << "  - Use Pool for production (predictable, well-tested)" 
              << std::endl;
    std::cout << "  - Use Async for research (zero-config, adaptive)" 
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
