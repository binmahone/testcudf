#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
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
#include <barrier>

/**
 * Test kernel submission bottleneck with multi-threading
 * Strategy 1: Many small kernels - will it bottleneck?
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

void worker_thread(
    int thread_id,
    const std::vector<std::unique_ptr<cudf::column>>& x_columns,
    const std::vector<std::unique_ptr<cudf::column>>& y_columns,
    size_t num_pairs,
    rmm::cuda_stream_view stream,
    std::barrier<>& barrier,
    double& thread_time) {
    
    // Wait for all threads
    barrier.arrive_and_wait();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process workload
    for (size_t i = 0; i < num_pairs; ++i) {
        auto result = conditional_coalesce_cast(
            x_columns[i]->view(),
            y_columns[i]->view(),
            stream
        );
    }
    
    stream.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    thread_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count() / 1000.0;
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 
        28ULL * 1024 * 1024 * 1024,
        30ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_OUTPUT_COLUMNS = 100;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_THREADS = 5;
    constexpr size_t NUM_RUNS = 10;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_output = total_rows / NUM_OUTPUT_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Multi-threaded Scalability Test" << std::endl;
    std::cout << "  Workload: cast(coalesce(if(y==1,x,0),0) as double)" 
              << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << NUM_THREADS << std::endl;
    std::cout << "  Pairs per thread: " << NUM_OUTPUT_COLUMNS << std::endl;
    std::cout << "  Total pairs: " << (NUM_THREADS * NUM_OUTPUT_COLUMNS) 
              << std::endl;
    std::cout << "  Data per thread: " 
              << format_bytes(TOTAL_DATA_SIZE) << std::endl;
    std::cout << "  RMM Pool: 28GB (max 30GB)" << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::cout << "Generating independent data for each thread..." 
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
    std::cout << "Data ready. Each thread has independent 1GB dataset." 
              << std::endl;
    std::cout << std::endl;
    
    // Create stream pool
    rmm::cuda_stream_pool stream_pool(NUM_THREADS);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Strategy 1 Multi-threaded" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<double> wall_times;
    std::vector<std::vector<double>> per_thread_times;
    per_thread_times.resize(NUM_THREADS);
    
    wall_times.reserve(NUM_RUNS);
    
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        std::barrier sync_barrier(NUM_THREADS);
        std::vector<double> thread_times(NUM_THREADS, 0.0);
        
        auto wall_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            rmm::cuda_stream_view stream = stream_pool.get_stream();
            
            threads.emplace_back(worker_thread,
                                t,
                                std::ref(all_x_columns[t]),
                                std::ref(all_y_columns[t]),
                                NUM_OUTPUT_COLUMNS,
                                stream,
                                std::ref(sync_barrier),
                                std::ref(thread_times[t]));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration_cast<
            std::chrono::microseconds>(wall_end - wall_start).count() 
            / 1000.0;
        wall_times.push_back(wall_time);
        
        std::cout << "  Run " << std::setw(2) << (run+1) << ": "
                  << "Wall=" << std::fixed << std::setprecision(3) 
                  << std::setw(8) << wall_time << "ms, Per-thread: [";
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            if (t > 0) std::cout << ", ";
            std::cout << std::setprecision(1) << thread_times[t];
            per_thread_times[t].push_back(thread_times[t]);
        }
        std::cout << "]ms" << std::endl;
    }
    
    auto calc_avg = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double val : v) sum += val;
        return sum / v.size();
    };
    
    double avg_wall = calc_avg(wall_times);
    double min_wall = *std::min_element(wall_times.begin(), 
                                         wall_times.end());
    double max_wall = *std::max_element(wall_times.begin(), 
                                         wall_times.end());
    
    std::cout << std::endl;
    std::cout << "Wall clock time (all threads):" << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(3) 
              << avg_wall << " ms" << std::endl;
    std::cout << "  Min:     " << min_wall << " ms" << std::endl;
    std::cout << "  Max:     " << max_wall << " ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Per-thread average times:" << std::endl;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        double avg_t = calc_avg(per_thread_times[t]);
        std::cout << "  Thread " << t << ": " << std::fixed 
                  << std::setprecision(3) << avg_t << " ms" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    double single_thread_baseline = 17.6;  // From previous test
    
    std::cout << "Single-threaded baseline (1 thread, 100 pairs):" 
              << std::endl;
    std::cout << "  Strategy 1: ~" << single_thread_baseline << " ms" 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "Multi-threaded (" << NUM_THREADS 
              << " threads, 100 pairs each):" << std::endl;
    std::cout << "  Wall clock: " << std::fixed << std::setprecision(3) 
              << avg_wall << " ms" << std::endl;
    std::cout << "  Expected (perfect parallelism): ~" 
              << single_thread_baseline << " ms" << std::endl;
    std::cout << std::endl;
    
    double parallel_efficiency = single_thread_baseline / avg_wall;
    double speedup = (single_thread_baseline * NUM_THREADS) / avg_wall;
    
    std::cout << "Scalability metrics:" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << speedup << "x (ideal: " << NUM_THREADS << ".0x)" 
              << std::endl;
    std::cout << "  Parallel efficiency: " << std::fixed 
              << std::setprecision(1) 
              << (speedup / NUM_THREADS * 100.0) << "%" << std::endl;
    std::cout << "  Overhead per thread: " << std::fixed 
              << std::setprecision(3)
              << (avg_wall - single_thread_baseline) << " ms" << std::endl;
    std::cout << std::endl;
    
    if (speedup >= NUM_THREADS * 0.85) {
        std::cout << "✓ Excellent parallelism!" << std::endl;
        std::cout << "  Kernel submission is NOT a bottleneck!" 
                  << std::endl;
    } else if (speedup >= NUM_THREADS * 0.6) {
        std::cout << "~ Good parallelism with some overhead" << std::endl;
        std::cout << "  Minor kernel submission contention" << std::endl;
    } else {
        std::cout << "⚠ Poor parallelism!" << std::endl;
        std::cout << "  Kernel submission bottleneck detected!" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Conclusion" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "For Strategy 1 (Individual processing):" << std::endl;
    std::cout << "  Single-thread: " << single_thread_baseline << " ms" 
              << std::endl;
    std::cout << "  Multi-thread (" << NUM_THREADS << " threads): " 
              << avg_wall << " ms wall clock" << std::endl;
    std::cout << "  Kernel submissions: " 
              << (NUM_THREADS * NUM_OUTPUT_COLUMNS * 5) 
              << " kernels (approx)" << std::endl;
    std::cout << "  Result: " << std::fixed << std::setprecision(1) 
              << (speedup / NUM_THREADS * 100.0) 
              << "% efficiency" << std::endl;
    
    std::cout << std::endl;
    std::cout << "CRITICAL FINDING:" << std::endl;
    std::cout << "  Each thread takes ~" << std::fixed << std::setprecision(1) 
              << calc_avg(per_thread_times[0]) << " ms" << std::endl;
    std::cout << "  vs " << single_thread_baseline 
              << " ms in single-threaded mode" << std::endl;
    std::cout << "  Slowdown: " << std::fixed << std::setprecision(2) 
              << (calc_avg(per_thread_times[0]) / single_thread_baseline) 
              << "x per thread!" << std::endl;
    std::cout << std::endl;
    
    std::cout << "This indicates:" << std::endl;
    if (speedup < NUM_THREADS * 0.7) {
        std::cout << "  ⚠️  KERNEL SUBMISSION BOTTLENECK CONFIRMED!" 
                  << std::endl;
        std::cout << "  - Many small kernels (2500 total) overwhelm GPU" 
                  << std::endl;
        std::cout << "  - Threads compete for GPU command queue" 
                  << std::endl;
        std::cout << "  - Each thread is serialized/slowed down" 
                  << std::endl;
    } else {
        std::cout << "  ✓ Good parallelism, no major bottleneck" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

