#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

/**
 * Test with different RMM memory resource strategies
 * Compare: cuda_memory_resource vs pool_memory_resource
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

void run_test(const std::string& mr_name, size_t num_runs) {
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_COLUMNS = 200;
    constexpr double NULL_PROBABILITY = 0.2;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_column = total_rows / NUM_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Test with: " << mr_name << std::endl;
    std::cout << "========================================" << std::endl;
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
    
    std::cout << "Running " << num_runs << " iterations:" << std::endl;
    
    for (size_t i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto concat_col = cudf::concatenate(column_views);
        auto coalesced = coalesce_column(concat_col->view());
        auto split_result = cudf::split(coalesced->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double time_ms = dur.count() / 1000.0;
        
        std::cout << "  " << std::setw(2) << (i+1) << ": " 
                  << std::fixed << std::setprecision(3) 
                  << std::setw(8) << time_ms << " ms";
        
        if (time_ms > 40.0) {
            std::cout << "  <-- SLOW!";
        }
        
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    constexpr size_t NUM_RUNS = 30;
    
    // Test 1: cuda_memory_resource (direct allocation)
    {
        rmm::mr::cuda_memory_resource mr;
        rmm::mr::set_current_device_resource(&mr);
        run_test("cuda_memory_resource (no pool)", NUM_RUNS);
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 2: pool_memory_resource
    {
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 2ULL * 1024 * 1024 * 1024  // 2GB initial pool
        );
        rmm::mr::set_current_device_resource(&pool_mr);
        run_test("pool_memory_resource (2GB pool)", NUM_RUNS);
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

