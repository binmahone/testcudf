#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/binaryop.hpp>
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
#include <atomic>
#include <barrier>

/**
 * Multi-threaded performance test
 * Test if Strategy 1's many kernel submissions become a bottleneck
 * when multiple threads are submitting work concurrently
 */

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " 
        << units[unit_index];
    return oss.str();
}

std::unique_ptr<cudf::column> generate_x_column(
    size_t num_rows,
    double null_probability,
    std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& data_dist,
    std::uniform_real_distribution<double>& null_dist) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = data_dist(gen);
    }
    
    size_t mask_size = cudf::bitmask_allocation_size_bytes(num_rows) / 
                       sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> null_mask(mask_size, 0);
    
    for (size_t i = 0; i < num_rows; ++i) {
        if (null_dist(gen) > null_probability) {
            cudf::set_bit_unsafe(null_mask.data(), i);
        }
    }
    
    rmm::device_buffer data_buffer(
        host_data.data(), 
        num_rows * sizeof(int32_t), 
        rmm::cuda_stream_default
    );
    rmm::device_buffer mask_buffer(
        null_mask.data(), 
        cudf::bitmask_allocation_size_bytes(num_rows),
        rmm::cuda_stream_default
    );
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        std::move(data_buffer),
        std::move(mask_buffer),
        0
    );
}

std::unique_ptr<cudf::column> generate_y_column(
    size_t num_rows,
    std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& binary_dist) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = binary_dist(gen);
    }
    
    rmm::device_buffer data_buffer(
        host_data.data(), 
        num_rows * sizeof(int32_t), 
        rmm::cuda_stream_default
    );
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        std::move(data_buffer),
        rmm::device_buffer{},
        0
    );
}

std::unique_ptr<cudf::column> conditional_coalesce_cast(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col,
    rmm::cuda_stream_view stream) {
    
    size_t num_rows = x_col.size();
    
    auto zero_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream
    );
    cudaMemsetAsync(zero_column->mutable_view().data<int32_t>(), 
                    0, num_rows * sizeof(int32_t), stream.value());
    
    auto x_is_valid = cudf::is_valid(x_col, stream);
    auto x_coalesced = cudf::copy_if_else(
        x_col,
        zero_column->view(),
        x_is_valid->view(),
        stream
    );
    
    auto one_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream
    );
    cudaMemsetAsync(one_column->mutable_view().data<int32_t>(), 
                    1, num_rows * sizeof(int32_t), stream.value());
    
    auto y_eq_1 = cudf::binary_operation(
        y_col,
        one_column->view(),
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream
    );
    
    auto result_int = cudf::copy_if_else(
        x_coalesced->view(),
        zero_column->view(),
        y_eq_1->view(),
        stream
    );
    
    auto result_double = cudf::cast(
        result_int->view(),
        cudf::data_type{cudf::type_id::FLOAT64},
        stream
    );
    
    return result_double;
}

// Worker function for Strategy 1 (individual)
void worker_strategy1(
    int thread_id,
    const std::vector<std::unique_ptr<cudf::column>>& x_columns,
    const std::vector<std::unique_ptr<cudf::column>>& y_columns,
    size_t num_pairs,
    rmm::cuda_stream_view stream,
    std::barrier<>& start_barrier,
    std::barrier<>& end_barrier,
    std::atomic<bool>& start_flag,
    std::chrono::high_resolution_clock::time_point& thread_start,
    std::chrono::high_resolution_clock::time_point& thread_end) {
    
    // Wait for all threads to be ready
    start_barrier.arrive_and_wait();
    
    // All threads start together
    if (thread_id == 0) {
        thread_start = std::chrono::high_resolution_clock::now();
    }
    
    // Process workload
    for (size_t i = 0; i < num_pairs; ++i) {
        auto result = conditional_coalesce_cast(
            x_columns[i]->view(),
            y_columns[i]->view(),
            stream
        );
    }
    
    // Synchronize this thread's stream
    stream.synchronize();
    
    // Wait for all threads to finish
    end_barrier.arrive_and_wait();
    
    if (thread_id == 0) {
        thread_end = std::chrono::high_resolution_clock::now();
    }
}

// Worker function for Strategy 2 (concat-split)
void worker_strategy2(
    int thread_id,
    const std::vector<cudf::column_view>& x_views,
    const std::vector<cudf::column_view>& y_views,
    const std::vector<cudf::size_type>& split_indices,
    rmm::cuda_stream_view stream,
    std::barrier<>& start_barrier,
    std::barrier<>& end_barrier,
    std::atomic<bool>& start_flag,
    std::chrono::high_resolution_clock::time_point& thread_start,
    std::chrono::high_resolution_clock::time_point& thread_end) {
    
    start_barrier.arrive_and_wait();
    
    if (thread_id == 0) {
        thread_start = std::chrono::high_resolution_clock::now();
    }
    
    // Concat, process, split
    auto concat_x = cudf::concatenate(x_views, stream);
    auto concat_y = cudf::concatenate(y_views, stream);
    auto result = conditional_coalesce_cast(
        concat_x->view(),
        concat_y->view(),
        stream
    );
    auto split_result = cudf::split(result->view(), split_indices);
    
    stream.synchronize();
    
    end_barrier.arrive_and_wait();
    
    if (thread_id == 0) {
        thread_end = std::chrono::high_resolution_clock::now();
    }
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 
        26ULL * 1024 * 1024 * 1024,  // 26GB pool
        29ULL * 1024 * 1024 * 1024   // 29GB max (留2GB给系统)
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_OUTPUT_COLUMNS = 100;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_THREADS = 2;
    constexpr size_t NUM_RUNS = 10;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_output = total_rows / NUM_OUTPUT_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Multi-threaded Performance Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << NUM_THREADS << std::endl;
    std::cout << "  Workload per thread: 100 pairs (x,y)" << std::endl;
    std::cout << "  Total workload: " << (NUM_THREADS * 100) << " pairs" 
              << std::endl;
    std::cout << "  Data per thread: " 
              << format_bytes(TOTAL_DATA_SIZE) << std::endl;
    std::cout << "  Total data: " 
              << format_bytes(TOTAL_DATA_SIZE * NUM_THREADS) << std::endl;
    std::cout << "  RMM Pool: 26GB (max 29GB)" << std::endl;
    std::cout << std::endl;
    
    // Generate data for each thread
    std::cout << "Generating data for " << NUM_THREADS << " threads..." 
              << std::endl;
    
    std::vector<std::vector<std::unique_ptr<cudf::column>>> all_x_columns;
    std::vector<std::vector<std::unique_ptr<cudf::column>>> all_y_columns;
    all_x_columns.resize(NUM_THREADS);
    all_y_columns.resize(NUM_THREADS);
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    std::uniform_int_distribution<int32_t> binary_dist(0, 1);
    
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        all_x_columns[t].reserve(NUM_OUTPUT_COLUMNS);
        all_y_columns[t].reserve(NUM_OUTPUT_COLUMNS);
        
        for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
            all_x_columns[t].push_back(
                generate_x_column(rows_per_output, NULL_PROBABILITY, 
                                 gen, data_dist, null_dist)
            );
            all_y_columns[t].push_back(
                generate_y_column(rows_per_output, gen, binary_dist)
            );
        }
    }
    cudaDeviceSynchronize();
    std::cout << "Data generation complete." << std::endl;
    std::cout << std::endl;
    
    // Create stream pool
    rmm::cuda_stream_pool stream_pool(NUM_THREADS);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Strategy 1: Individual (Multi-threaded)" << std::endl;
    std::cout << "  " << NUM_THREADS << " threads x 100 pairs = " 
              << (NUM_THREADS * 100) << " total operations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<double> s1_times;
    s1_times.reserve(NUM_RUNS);
    
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier start_barrier(NUM_THREADS);
        std::barrier end_barrier(NUM_THREADS);
        std::atomic<bool> start_flag{false};
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            rmm::cuda_stream_view stream = stream_pool.get_stream();
            
            threads.emplace_back(worker_strategy1,
                                t,
                                std::ref(all_x_columns[t]),
                                std::ref(all_y_columns[t]),
                                NUM_OUTPUT_COLUMNS,
                                stream,
                                std::ref(start_barrier),
                                std::ref(end_barrier),
                                std::ref(start_flag),
                                std::ref(start_time),
                                std::ref(end_time));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = std::chrono::duration_cast<
            std::chrono::microseconds>(end_time - start_time).count() 
            / 1000.0;
        s1_times.push_back(elapsed);
        
        std::cout << "  Run " << std::setw(2) << (run+1) << ": " 
                  << std::fixed << std::setprecision(3) 
                  << elapsed << " ms" << std::endl;
    }
    
    double avg_s1 = 0.0;
    for (double t : s1_times) avg_s1 += t;
    avg_s1 /= s1_times.size();
    
    double min_s1 = *std::min_element(s1_times.begin(), s1_times.end());
    double max_s1 = *std::max_element(s1_times.begin(), s1_times.end());
    
    std::cout << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(3) 
              << avg_s1 << " ms" << std::endl;
    std::cout << "  Min:     " << min_s1 << " ms" << std::endl;
    std::cout << "  Max:     " << max_s1 << " ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Strategy 2: Concat-Split (Multi-threaded)" << std::endl;
    std::cout << "  " << NUM_THREADS << " threads, each does concat-split" 
              << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Prepare views for each thread
    std::vector<std::vector<cudf::column_view>> all_x_views;
    std::vector<std::vector<cudf::column_view>> all_y_views;
    all_x_views.resize(NUM_THREADS);
    all_y_views.resize(NUM_THREADS);
    
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(NUM_OUTPUT_COLUMNS - 1);
    for (size_t i = 1; i < NUM_OUTPUT_COLUMNS; ++i) {
        split_indices.push_back(i * rows_per_output);
    }
    
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
            all_x_views[t].push_back(all_x_columns[t][i]->view());
            all_y_views[t].push_back(all_y_columns[t][i]->view());
        }
    }
    
    std::vector<double> s2_times;
    s2_times.reserve(NUM_RUNS);
    
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier start_barrier(NUM_THREADS);
        std::barrier end_barrier(NUM_THREADS);
        std::atomic<bool> start_flag{false};
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            rmm::cuda_stream_view stream = stream_pool.get_stream();
            
            threads.emplace_back(worker_strategy2,
                                t,
                                std::ref(all_x_views[t]),
                                std::ref(all_y_views[t]),
                                std::ref(split_indices),
                                stream,
                                std::ref(start_barrier),
                                std::ref(end_barrier),
                                std::ref(start_flag),
                                std::ref(start_time),
                                std::ref(end_time));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = std::chrono::duration_cast<
            std::chrono::microseconds>(end_time - start_time).count() 
            / 1000.0;
        s2_times.push_back(elapsed);
        
        std::cout << "  Run " << std::setw(2) << (run+1) << ": " 
                  << std::fixed << std::setprecision(3) 
                  << elapsed << " ms" << std::endl;
    }
    
    double avg_s2 = 0.0;
    for (double t : s2_times) avg_s2 += t;
    avg_s2 /= s2_times.size();
    
    double min_s2 = *std::min_element(s2_times.begin(), s2_times.end());
    double max_s2 = *std::max_element(s2_times.begin(), s2_times.end());
    
    std::cout << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(3) 
              << avg_s2 << " ms" << std::endl;
    std::cout << "  Min:     " << min_s2 << " ms" << std::endl;
    std::cout << "  Max:     " << max_s2 << " ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Single-threaded baseline (from previous test):" 
              << std::endl;
    std::cout << "  Strategy 1: ~17.6 ms (100 pairs)" << std::endl;
    std::cout << "  Strategy 2: ~34.7 ms (100 pairs)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Multi-threaded (" << NUM_THREADS << " threads):" 
              << std::endl;
    std::cout << "  Strategy 1: " << std::fixed << std::setprecision(3) 
              << avg_s1 << " ms" << std::endl;
    std::cout << "  Strategy 2: " << avg_s2 << " ms" << std::endl;
    std::cout << std::endl;
    
    double s1_parallel_efficiency = (17.6 * NUM_THREADS) / avg_s1;
    double s2_parallel_efficiency = (34.7 * NUM_THREADS) / avg_s2;
    
    std::cout << "Parallel efficiency (ideal = " << NUM_THREADS 
              << ".0x):" << std::endl;
    std::cout << "  Strategy 1: " << std::fixed << std::setprecision(2) 
              << s1_parallel_efficiency << "x";
    if (s1_parallel_efficiency < NUM_THREADS * 0.8) {
        std::cout << "  ⚠️  Kernel submission bottleneck detected!";
    } else {
        std::cout << "  ✓  Good parallelism";
    }
    std::cout << std::endl;
    
    std::cout << "  Strategy 2: " << s2_parallel_efficiency << "x";
    if (s2_parallel_efficiency < NUM_THREADS * 0.8) {
        std::cout << "  ⚠️  Bottleneck detected!";
    } else {
        std::cout << "  ✓  Good parallelism";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    if (avg_s1 < avg_s2) {
        std::cout << "Winner: Strategy 1 is " 
                  << std::fixed << std::setprecision(2) 
                  << (avg_s2 / avg_s1) << "x faster in multi-threaded!" 
                  << std::endl;
    } else {
        std::cout << "Winner: Strategy 2 is " 
                  << std::fixed << std::setprecision(2) 
                  << (avg_s1 / avg_s2) << "x faster in multi-threaded!" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

