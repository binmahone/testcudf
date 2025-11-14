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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <thread>
#include <barrier>

/**
 * CONCURRENT mode only - for nsys profiling
 */

std::unique_ptr<cudf::column> generate_x(size_t rows, double null_prob) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1, 1000);
    std::uniform_real_distribution<double> ndist(0, 1);
    
    std::vector<int32_t> data(rows);
    for (size_t i = 0; i < rows; ++i) data[i] = dist(gen);
    
    size_t ms = cudf::bitmask_allocation_size_bytes(rows) / 
                sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> mask(ms, 0);
    for (size_t i = 0; i < rows; ++i) {
        if (ndist(gen) > null_prob) cudf::set_bit_unsafe(mask.data(), i);
    }
    
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t),
                            rmm::cuda_stream_default);
    rmm::device_buffer mbuf(mask.data(), 
                            cudf::bitmask_allocation_size_bytes(rows),
                            rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), std::move(mbuf), 0
    );
}


std::unique_ptr<cudf::column> w1_original(
    const cudf::column_view& x, rmm::cuda_stream_view stream) {
    
    auto zero = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, stream
    );
    cudaMemsetAsync(zero->mutable_view().data<int32_t>(), 0, 
                    x.size() * sizeof(int32_t), stream.value());
    
    auto mask = cudf::is_valid(x, stream);
    auto coalesced = cudf::copy_if_else(x, zero->view(), mask->view(), 
                                        stream);
    return cudf::cast(coalesced->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64}, stream);
}


int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 10ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t NUM_ITEMS = 100;
    constexpr size_t ROWS = 2684354;
    constexpr size_t NUM_THREADS = 4;
    constexpr size_t NUM_RUNS = 3;
    
    std::cout << "=== CONCURRENT Mode (for nsys profiling) ===" 
              << std::endl;
    std::cout << "Configuration: " << NUM_THREADS << " threads, " 
              << NUM_ITEMS << " items per thread" << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::vector<std::vector<std::unique_ptr<cudf::column>>> all_x(
        NUM_THREADS);
    
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            all_x[t].push_back(generate_x(ROWS, 0.2));
        }
    }
    cudaDeviceSynchronize();
    
    rmm::cuda_stream_pool stream_pool(NUM_THREADS);
    
    // Workload 1 - Run 3 times
    std::cout << "Running Workload 1 (CONCURRENT) - 3 runs..." << std::endl;
    std::vector<double> w1_times;
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::cout << "  Run " << (run + 1) << "/" << NUM_RUNS << "..." 
                  << std::endl;
        std::barrier barrier(NUM_THREADS);
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                barrier.arrive_and_wait();
                for (size_t i = 0; i < NUM_ITEMS; ++i) {
                    w1_original(all_x[t][i]->view(), stream);
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count() / 1000.0;
        w1_times.push_back(elapsed);
        std::cout << "    Time: " << std::fixed << std::setprecision(2) 
                  << elapsed << " ms" << std::endl;
    }
    
    double w1_avg = 0;
    for (double t : w1_times) w1_avg += t;
    w1_avg /= w1_times.size();
    std::cout << std::endl << "Average time: " << std::fixed 
              << std::setprecision(2) << w1_avg << " ms" << std::endl;
    
    std::cout << std::endl << "CONCURRENT mode completed." << std::endl;
    
    return 0;
}

