#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

/**
 * Test for memory leak / resource accumulation in Strategy 2
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

std::unique_ptr<cudf::column> generate_column(
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

std::unique_ptr<cudf::column> coalesce_column(
    const cudf::column_view& input_column) {
    
    size_t num_rows = input_column.size();
    
    auto zero_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        rmm::cuda_stream_default
    );
    
    cudaMemset(zero_column->mutable_view().data<int32_t>(), 
               0, 
               num_rows * sizeof(int32_t));
    
    auto is_valid_mask = cudf::is_valid(input_column);
    
    return cudf::copy_if_else(
        input_column,
        zero_column->view(),
        is_valid_mask->view()
    );
}

int main() {
    rmm::mr::cuda_memory_resource mr;
    rmm::mr::set_current_device_resource(&mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;  // 1GB
    constexpr size_t NUM_COLUMNS = 200;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_RUNS = 30;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_column = total_rows / NUM_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Memory Leak / Resource Accumulation Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Get initial memory
    size_t initial_free = get_gpu_free_memory();
    std::cout << "Initial GPU free memory: " 
              << format_bytes(initial_free) << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.reserve(NUM_COLUMNS);
    
    for (size_t i = 0; i < NUM_COLUMNS; ++i) {
        input_columns.push_back(
            generate_column(rows_per_column, NULL_PROBABILITY, 
                           gen, data_dist, null_dist)
        );
    }
    cudaDeviceSynchronize();
    
    size_t after_data_free = get_gpu_free_memory();
    std::cout << "After data generation: " 
              << format_bytes(after_data_free) << std::endl;
    std::cout << "Data allocated: " 
              << format_bytes(initial_free - after_data_free) << std::endl;
    std::cout << std::endl;
    
    // Prepare for concat-split
    std::vector<cudf::column_view> column_views;
    column_views.reserve(NUM_COLUMNS);
    for (const auto& col : input_columns) {
        column_views.push_back(col->view());
    }
    
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(NUM_COLUMNS - 1);
    for (size_t i = 1; i < NUM_COLUMNS; ++i) {
        split_indices.push_back(i * rows_per_column);
    }
    
    std::cout << "Running " << NUM_RUNS 
              << " iterations with memory tracking..." << std::endl;
    std::cout << std::endl;
    
    std::cout << std::left << std::setw(8) << "Run"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Free Memory"
              << std::setw(20) << "Memory Change"
              << std::endl;
    std::cout << std::string(63, '-') << std::endl;
    
    size_t prev_free = after_data_free;
    
    for (size_t i = 0; i < NUM_RUNS; ++i) {
        size_t before_free = get_gpu_free_memory();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Strategy 2 operations
        auto concat_col = cudf::concatenate(column_views);
        auto coalesced = coalesce_column(concat_col->view());
        auto split_result = cudf::split(coalesced->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Explicitly destroy temporary objects
        split_result.clear();
        coalesced.reset();
        concat_col.reset();
        
        cudaDeviceSynchronize();
        
        size_t after_free = get_gpu_free_memory();
        
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double time_ms = dur.count() / 1000.0;
        
        long long mem_change = static_cast<long long>(after_free) - 
                               static_cast<long long>(prev_free);
        
        std::cout << std::left << std::setw(8) << (i+1)
                  << std::right << std::setw(15) << std::fixed 
                  << std::setprecision(3) << time_ms
                  << std::setw(20) << format_bytes(after_free)
                  << std::setw(15);
        
        if (mem_change > 0) {
            std::cout << "+" << format_bytes(mem_change);
        } else if (mem_change < 0) {
            std::cout << "-" << format_bytes(-mem_change);
        } else {
            std::cout << "0";
        }
        
        // Highlight slow runs
        if (time_ms > 40.0) {
            std::cout << "  <-- SLOW!";
        }
        
        std::cout << std::endl;
        
        prev_free = after_free;
    }
    
    std::cout << std::endl;
    
    size_t final_free = get_gpu_free_memory();
    long long total_leak = static_cast<long long>(after_data_free) - 
                           static_cast<long long>(final_free);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Initial free (after data): " 
              << format_bytes(after_data_free) << std::endl;
    std::cout << "  Final free:                " 
              << format_bytes(final_free) << std::endl;
    
    if (total_leak > 1024*1024) {  // > 1MB
        std::cout << "  Net memory leaked: " 
                  << format_bytes(total_leak) << " ⚠️" << std::endl;
        std::cout << "  Conclusion: Possible memory leak detected!" 
                  << std::endl;
    } else if (total_leak < -1024*1024) {
        std::cout << "  Net memory freed: " 
                  << format_bytes(-total_leak) << std::endl;
    } else {
        std::cout << "  Net memory change: ~0 (within 1MB)" << std::endl;
        std::cout << "  Conclusion: No significant memory leak." 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

