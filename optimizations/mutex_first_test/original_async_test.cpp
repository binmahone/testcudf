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
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
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
#include <mutex>

/**
 * Test mutex optimization on ORIGINAL workloads with ASYNC memory resource
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

std::unique_ptr<cudf::column> generate_y(size_t rows) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1);
    
    std::vector<int32_t> data(rows);
    for (size_t i = 0; i < rows; ++i) data[i] = dist(gen);
    
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t),
                            rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), rmm::device_buffer{}, 0
    );
}

// Workload 1 - ORIGINAL (3 kernels)
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

// Workload 2 - ORIGINAL (5+ kernels)
std::unique_ptr<cudf::column> w2_original(
    const cudf::column_view& x, const cudf::column_view& y,
    rmm::cuda_stream_view stream) {
    
    auto zero = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, stream
    );
    cudaMemsetAsync(zero->mutable_view().data<int32_t>(), 0, 
                    x.size() * sizeof(int32_t), stream.value());
    
    auto x_valid = cudf::is_valid(x, stream);
    auto x_coal = cudf::copy_if_else(x, zero->view(), x_valid->view(), 
                                     stream);
    
    auto one = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, stream
    );
    cudaMemsetAsync(one->mutable_view().data<int32_t>(), 1, 
                    x.size() * sizeof(int32_t), stream.value());
    
    auto y_eq_1 = cudf::binary_operation(
        y, one->view(), cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, stream
    );
    
    auto res_int = cudf::copy_if_else(x_coal->view(), zero->view(), 
                                      y_eq_1->view(), stream);
    
    return cudf::cast(res_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64}, stream);
}

int main() {
    rmm::mr::cuda_async_memory_resource async_mr;
    rmm::mr::set_current_device_resource(&async_mr);
    
    constexpr size_t NUM_ITEMS = 100;
    constexpr size_t ROWS = 2684354;
    constexpr size_t NUM_THREADS = 4;
    constexpr size_t NUM_RUNS = 10;
    
    std::cout << "=== ASYNC Memory Resource Test ===" << std::endl;
    std::cout << "Memory: CUDA Async (driver-managed)" << std::endl;
    std::cout << "Configuration: 4 threads, 100 items per thread" 
              << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::vector<std::vector<std::unique_ptr<cudf::column>>> all_x(
        NUM_THREADS);
    std::vector<std::vector<std::unique_ptr<cudf::column>>> all_y(
        NUM_THREADS);
    
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            all_x[t].push_back(generate_x(ROWS, 0.2));
            all_y[t].push_back(generate_y(ROWS));
        }
    }
    cudaDeviceSynchronize();
    
    rmm::cuda_stream_pool stream_pool(NUM_THREADS);
    
    // Test Workload 1
    std::cout << "Workload 1: CAST(COALESCE(x,0) AS DOUBLE)" << std::endl;
    std::cout << "Original implementation (3 kernels)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Concurrent
    std::vector<double> w1_concurrent;
    for (size_t run = 0; run < NUM_RUNS; ++run) {
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
        w1_concurrent.push_back(std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count() / 1000.0);
    }
    
    double w1_conc_avg = 0;
    for (double t : w1_concurrent) w1_conc_avg += t;
    w1_conc_avg /= w1_concurrent.size();
    
    // Mutex
    std::vector<double> w1_mutex;
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier barrier(NUM_THREADS);
        std::mutex submit_mutex;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                barrier.arrive_and_wait();
                {
                    std::lock_guard<std::mutex> lock(submit_mutex);
                    for (size_t i = 0; i < NUM_ITEMS; ++i) {
                        w1_original(all_x[t][i]->view(), stream);
                    }
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        
        auto end = std::chrono::high_resolution_clock::now();
        w1_mutex.push_back(std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count() / 1000.0);
    }
    
    double w1_mutex_avg = 0;
    for (double t : w1_mutex) w1_mutex_avg += t;
    w1_mutex_avg /= w1_mutex.size();
    
    std::cout << "  Concurrent: " << std::fixed << std::setprecision(2) 
              << w1_conc_avg << " ms" << std::endl;
    std::cout << "  With mutex: " << w1_mutex_avg << " ms ("
              << std::fixed << std::setprecision(1)
              << ((w1_conc_avg - w1_mutex_avg) / w1_conc_avg * 100)
              << "% faster)" << std::endl;
    std::cout << std::endl;
    
    // Test Workload 2
    std::cout << "Workload 2: CAST(COALESCE(IF(y=1,x,0),0) AS DOUBLE)" 
              << std::endl;
    std::cout << "Original implementation (5+ kernels)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Concurrent
    std::vector<double> w2_concurrent;
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier barrier(NUM_THREADS);
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                barrier.arrive_and_wait();
                for (size_t i = 0; i < NUM_ITEMS; ++i) {
                    w2_original(all_x[t][i]->view(), all_y[t][i]->view(), 
                               stream);
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        
        auto end = std::chrono::high_resolution_clock::now();
        w2_concurrent.push_back(std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count() / 1000.0);
    }
    
    double w2_conc_avg = 0;
    for (double t : w2_concurrent) w2_conc_avg += t;
    w2_conc_avg /= w2_concurrent.size();
    
    // Mutex
    std::vector<double> w2_mutex;
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier barrier(NUM_THREADS);
        std::mutex submit_mutex;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            auto stream = stream_pool.get_stream();
            threads.emplace_back([&, t, stream]() {
                barrier.arrive_and_wait();
                {
                    std::lock_guard<std::mutex> lock(submit_mutex);
                    for (size_t i = 0; i < NUM_ITEMS; ++i) {
                        w2_original(all_x[t][i]->view(), 
                                   all_y[t][i]->view(), stream);
                    }
                }
                stream.synchronize();
            });
        }
        for (auto& th : threads) th.join();
        
        auto end = std::chrono::high_resolution_clock::now();
        w2_mutex.push_back(std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count() / 1000.0);
    }
    
    double w2_mutex_avg = 0;
    for (double t : w2_mutex) w2_mutex_avg += t;
    w2_mutex_avg /= w2_mutex.size();
    
    std::cout << "  Concurrent: " << std::fixed << std::setprecision(2) 
              << w2_conc_avg << " ms" << std::endl;
    std::cout << "  With mutex: " << w2_mutex_avg << " ms ("
              << std::fixed << std::setprecision(1)
              << ((w2_conc_avg - w2_mutex_avg) / w2_conc_avg * 100)
              << "% faster)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Summary - ASYNC Memory Resource" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

