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
#include <barrier>

/**
 * Calculate exact memory requirement for Strategy 2 with 5 threads
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

size_t get_gpu_free_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
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

void worker_strategy2(
    int thread_id,
    const std::vector<cudf::column_view>& x_views,
    const std::vector<cudf::column_view>& y_views,
    const std::vector<cudf::size_type>& split_indices,
    rmm::cuda_stream_view stream,
    std::barrier<>& barrier,
    size_t& peak_mem_usage) {
    
    barrier.arrive_and_wait();
    
    size_t before_mem = get_gpu_free_memory();
    
    auto concat_x = cudf::concatenate(x_views, stream);
    size_t after_concat_x = get_gpu_free_memory();
    
    auto concat_y = cudf::concatenate(y_views, stream);
    size_t after_concat_y = get_gpu_free_memory();
    
    auto result = conditional_coalesce_cast(
        concat_x->view(),
        concat_y->view(),
        stream
    );
    size_t after_process = get_gpu_free_memory();
    
    auto split_result = cudf::split(result->view(), split_indices);
    size_t after_split = get_gpu_free_memory();
    
    stream.synchronize();
    
    peak_mem_usage = before_mem - after_process;
    
    if (thread_id == 0) {
        std::cout << "  Thread " << thread_id << " memory trace:" 
                  << std::endl;
        std::cout << "    Before:         " 
                  << format_bytes(before_mem) << " free" << std::endl;
        std::cout << "    After concat x: " 
                  << format_bytes(after_concat_x) << " free (-" 
                  << format_bytes(before_mem - after_concat_x) << ")" 
                  << std::endl;
        std::cout << "    After concat y: " 
                  << format_bytes(after_concat_y) << " free (-" 
                  << format_bytes(after_concat_x - after_concat_y) << ")" 
                  << std::endl;
        std::cout << "    After process:  " 
                  << format_bytes(after_process) << " free (-" 
                  << format_bytes(after_concat_y - after_process) << ")" 
                  << std::endl;
        std::cout << "    Peak usage:     " 
                  << format_bytes(peak_mem_usage) << std::endl;
    }
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_OUTPUT_COLUMNS = 100;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_THREADS = 5;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_output = total_rows / NUM_OUTPUT_COLUMNS;
    
    size_t total_gpu_mem, free_gpu_mem;
    cudaMemGetInfo(&free_gpu_mem, &total_gpu_mem);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Strategy 2 Memory Requirement Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "GPU Memory:" << std::endl;
    std::cout << "  Total: " << format_bytes(total_gpu_mem) << std::endl;
    std::cout << "  Free:  " << format_bytes(free_gpu_mem) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << NUM_THREADS << std::endl;
    std::cout << "  Pairs per thread: " << NUM_OUTPUT_COLUMNS << std::endl;
    std::cout << "  Data per thread: " 
              << format_bytes(TOTAL_DATA_SIZE) << " input" << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Theoretical Memory Calculation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    size_t input_x_mem = TOTAL_DATA_SIZE;
    size_t input_y_mem = TOTAL_DATA_SIZE;
    size_t concat_x_mem = TOTAL_DATA_SIZE;
    size_t concat_y_mem = TOTAL_DATA_SIZE;
    size_t intermediate_mem = TOTAL_DATA_SIZE * 4;  // Various temp buffers
    size_t output_mem = TOTAL_DATA_SIZE * 2;  // DOUBLE
    
    size_t per_thread_peak = input_x_mem + input_y_mem + concat_x_mem + 
                             concat_y_mem + intermediate_mem + output_mem;
    
    std::cout << "Per-thread memory (Strategy 2):" << std::endl;
    std::cout << "  Input x columns:      " << format_bytes(input_x_mem) 
              << std::endl;
    std::cout << "  Input y columns:      " << format_bytes(input_y_mem) 
              << std::endl;
    std::cout << "  Concat x buffer:      " << format_bytes(concat_x_mem) 
              << std::endl;
    std::cout << "  Concat y buffer:      " << format_bytes(concat_y_mem) 
              << std::endl;
    std::cout << "  Intermediate buffers: " << format_bytes(intermediate_mem) 
              << std::endl;
    std::cout << "  Output (DOUBLE):      " << format_bytes(output_mem) 
              << std::endl;
    std::cout << "  ─────────────────────────────" << std::endl;
    std::cout << "  Peak per thread:      " << format_bytes(per_thread_peak) 
              << std::endl;
    std::cout << std::endl;
    
    size_t total_for_5_threads = per_thread_peak * NUM_THREADS;
    std::cout << "Total for " << NUM_THREADS << " threads:" << std::endl;
    std::cout << "  Estimated peak: " << format_bytes(total_for_5_threads) 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Actual Memory Test - Single Thread" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test with single thread first
    size_t POOL_SIZE = 10ULL * 1024 * 1024 * 1024;  // 10GB
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 
        POOL_SIZE,
        POOL_SIZE + 2ULL * 1024 * 1024 * 1024  // +2GB max
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    std::cout << "Using " << format_bytes(POOL_SIZE) << " pool..." 
              << std::endl;
    
    // Generate data for 1 thread
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    std::uniform_int_distribution<int32_t> binary_dist(0, 1);
    
    std::vector<std::unique_ptr<cudf::column>> x_columns;
    std::vector<std::unique_ptr<cudf::column>> y_columns;
    x_columns.reserve(NUM_OUTPUT_COLUMNS);
    y_columns.reserve(NUM_OUTPUT_COLUMNS);
    
    size_t before_data = get_gpu_free_memory();
    
    for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
        x_columns.push_back(generate_x_column(
            rows_per_output, NULL_PROBABILITY, 
            gen, data_dist, null_dist
        ));
        y_columns.push_back(generate_y_column(
            rows_per_output, gen, binary_dist
        ));
    }
    cudaDeviceSynchronize();
    
    size_t after_data = get_gpu_free_memory();
    size_t data_mem = before_data - after_data;
    
    std::cout << "  Input data: " << format_bytes(data_mem) << std::endl;
    
    // Prepare views
    std::vector<cudf::column_view> x_views;
    std::vector<cudf::column_view> y_views;
    for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
        x_views.push_back(x_columns[i]->view());
        y_views.push_back(y_columns[i]->view());
    }
    
    std::vector<cudf::size_type> split_indices;
    for (size_t i = 1; i < NUM_OUTPUT_COLUMNS; ++i) {
        split_indices.push_back(i * rows_per_output);
    }
    
    // Track memory during execution
    size_t min_free = before_data;
    
    size_t before_concat_x = get_gpu_free_memory();
    auto concat_x = cudf::concatenate(x_views);
    cudaDeviceSynchronize();
    size_t after_concat_x = get_gpu_free_memory();
    min_free = std::min(min_free, after_concat_x);
    
    std::cout << "  After concat x: -" 
              << format_bytes(before_concat_x - after_concat_x) 
              << " (free: " << format_bytes(after_concat_x) << ")" 
              << std::endl;
    
    auto concat_y = cudf::concatenate(y_views);
    cudaDeviceSynchronize();
    size_t after_concat_y = get_gpu_free_memory();
    min_free = std::min(min_free, after_concat_y);
    
    std::cout << "  After concat y: -" 
              << format_bytes(after_concat_x - after_concat_y) 
              << " (free: " << format_bytes(after_concat_y) << ")" 
              << std::endl;
    
    auto result = conditional_coalesce_cast(
        concat_x->view(),
        concat_y->view(),
        rmm::cuda_stream_default
    );
    cudaDeviceSynchronize();
    size_t after_process = get_gpu_free_memory();
    min_free = std::min(min_free, after_process);
    
    std::cout << "  After process:  -" 
              << format_bytes(after_concat_y - after_process) 
              << " (free: " << format_bytes(after_process) << ")" 
              << std::endl;
    
    auto split_result = cudf::split(result->view(), split_indices);
    cudaDeviceSynchronize();
    size_t after_split = get_gpu_free_memory();
    min_free = std::min(min_free, after_split);
    
    std::cout << "  After split:    -" 
              << format_bytes(after_process - after_split) 
              << " (free: " << format_bytes(after_split) << ")" 
              << std::endl;
    
    size_t peak_usage = before_data - min_free;
    
    std::cout << std::endl;
    std::cout << "  Peak memory usage (single thread): " 
              << format_bytes(peak_usage) << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Memory Requirement for 5 Threads" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    size_t required_for_5 = peak_usage * NUM_THREADS;
    size_t recommended_pool = required_for_5 * 1.2;  // +20% safety margin
    
    std::cout << "Calculation:" << std::endl;
    std::cout << "  Per-thread peak: " << format_bytes(peak_usage) 
              << std::endl;
    std::cout << "  5 threads total: " << format_bytes(required_for_5) 
              << " (minimum)" << std::endl;
    std::cout << "  Recommended pool: " << format_bytes(recommended_pool) 
              << " (+20% margin)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Available GPU memory: " << format_bytes(total_gpu_mem) 
              << std::endl;
    
    if (recommended_pool <= total_gpu_mem) {
        std::cout << "  ✓ Sufficient GPU memory available" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommended RMM pool configuration:" << std::endl;
        std::cout << "  Initial: " << format_bytes(recommended_pool) 
                  << std::endl;
        std::cout << "  Maximum: " << format_bytes(
            std::min(recommended_pool + 2ULL*1024*1024*1024, 
                     total_gpu_mem - 1ULL*1024*1024*1024)) 
                  << std::endl;
    } else {
        std::cout << "  ⚠️  NOT enough GPU memory!" << std::endl;
        std::cout << "  Need: " << format_bytes(recommended_pool) 
                  << std::endl;
        std::cout << "  Have: " << format_bytes(total_gpu_mem) << std::endl;
        std::cout << "  Strategy 2 cannot run with 5 threads on this GPU!" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

