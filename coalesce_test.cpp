#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>

/**
 * This program demonstrates coalesce(x, 0) operation using CUDF.
 * coalesce(x, 0) returns x if x is not NULL, otherwise returns 0.
 * Uses copy_if_else API and processes ~1GB of data.
 */

// Structure to hold test results
struct TestResult {
    double coalesce_time_ms;
    double data_transfer_time_ms;
    double throughput_gbps;
    size_t num_columns;
    size_t num_rows;
    size_t null_count;
    std::vector<double> individual_run_times;
};

// Helper function to print first N elements of column
template <typename T>
void print_column_sample(const cudf::column_view& col, 
                         size_t max_print = 20) {
    size_t print_count = std::min(static_cast<size_t>(col.size()), 
                                   max_print);
    std::vector<T> host_data(print_count);
    cudaMemcpy(host_data.data(), 
               col.data<T>(), 
               print_count * sizeof(T), 
               cudaMemcpyDeviceToHost);
    
    std::vector<bool> null_mask;
    if (col.nullable()) {
        null_mask.resize(print_count);
        
        std::vector<cudf::bitmask_type> mask_data(
            cudf::bitmask_allocation_size_bytes(print_count) / 
            sizeof(cudf::bitmask_type)
        );
        cudaMemcpy(mask_data.data(), 
                   col.null_mask(), 
                   cudf::bitmask_allocation_size_bytes(print_count),
                   cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < print_count; ++i) {
            null_mask[i] = cudf::bit_is_set(mask_data.data(), i);
        }
    }
    
    std::cout << "[";
    for (size_t i = 0; i < print_count; ++i) {
        if (i > 0) std::cout << ", ";
        if (col.nullable() && !null_mask[i]) {
            std::cout << "NULL";
        } else {
            std::cout << host_data[i];
        }
    }
    if (col.size() > max_print) {
        std::cout << ", ... (" << (col.size() - max_print) << " more)";
    }
    std::cout << "]" << std::endl;
}

// Helper function to format bytes
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

/**
 * Generate a column with random data and NULL values
 */
std::unique_ptr<cudf::column> generate_column(
    size_t num_rows,
    double null_probability,
    std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& data_dist,
    std::uniform_real_distribution<double>& null_dist,
    size_t& out_null_count) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = data_dist(gen);
    }
    
    size_t mask_size = cudf::bitmask_allocation_size_bytes(num_rows) / 
                       sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> null_mask(mask_size, 0);
    size_t null_count = 0;
    
    for (size_t i = 0; i < num_rows; ++i) {
        if (null_dist(gen) > null_probability) {
            cudf::set_bit_unsafe(null_mask.data(), i);
        } else {
            null_count++;
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
    
    out_null_count = null_count;
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        std::move(data_buffer),
        std::move(mask_buffer),
        null_count
    );
}

/**
 * Perform coalesce operation on a single column using copy_if_else
 */
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

/**
 * Test 1: Single large column (1GB) with multiple runs
 */
TestResult run_single_column_test(size_t total_data_size, 
                                   double null_probability,
                                   size_t num_runs) {
    std::cout << "=== Single Column Coalesce Test ===" 
              << std::endl << std::endl;
    
    size_t num_rows = total_data_size / sizeof(int32_t);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Number of rows: " << num_rows << std::endl;
    std::cout << "  Data size: " 
              << format_bytes(num_rows * sizeof(int32_t)) << std::endl;
    std::cout << "  NULL probability: " 
              << (null_probability * 100) << "%" << std::endl;
    std::cout << "  Number of runs: " << num_runs << std::endl;
    std::cout << std::endl;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    
    std::cout << "Generating and transferring data to GPU..." 
              << std::endl;
    auto start_transfer = std::chrono::high_resolution_clock::now();
    
    size_t null_count = 0;
    auto input_column = generate_column(num_rows, null_probability, gen, 
                                         data_dist, null_dist, null_count);
    
    cudaDeviceSynchronize();
    auto end_transfer = std::chrono::high_resolution_clock::now();
    
    auto transfer_duration = std::chrono::duration_cast<
        std::chrono::milliseconds>(end_transfer - start_transfer);
    
    std::cout << "  Data transfer time: " << transfer_duration.count() 
              << " ms" << std::endl;
    std::cout << "  NULL count: " << null_count << " ("
              << (100.0 * null_count / num_rows) << "%)" 
              << std::endl << std::endl;
    
    // Warm-up runs
    std::cout << "Running warm-up iterations..." << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        auto warmup_result = coalesce_column(input_column->view());
        cudaDeviceSynchronize();
    }
    
    // Timed runs
    std::cout << "Running " << num_runs << " timed iterations..." 
              << std::endl;
    std::vector<double> run_times;
    run_times.reserve(num_runs);
    
    for (size_t i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unique_ptr<cudf::column> result_column = 
            coalesce_column(input_column->view());
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);
        run_times.push_back(duration.count() / 1000.0);
    }
    
    double avg_time = 0.0;
    for (double t : run_times) {
        avg_time += t;
    }
    avg_time /= run_times.size();
    
    double min_time = *std::min_element(run_times.begin(), 
                                         run_times.end());
    double max_time = *std::max_element(run_times.begin(), 
                                         run_times.end());
    
    double data_gb = num_rows * sizeof(int32_t) / 1e9;
    double throughput = data_gb / (avg_time / 1000.0);
    
    std::cout << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Coalesce time (avg): " << avg_time << " ms" 
              << std::endl;
    std::cout << "  Coalesce time (min): " << min_time << " ms" 
              << std::endl;
    std::cout << "  Coalesce time (max): " << max_time << " ms" 
              << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
    
    TestResult result;
    result.coalesce_time_ms = avg_time;
    result.data_transfer_time_ms = transfer_duration.count();
    result.throughput_gbps = throughput;
    result.num_columns = 1;
    result.num_rows = num_rows;
    result.null_count = null_count;
    result.individual_run_times = run_times;
    
    return result;
}

/**
 * Test 2: Multiple columns with multiple runs
 */
TestResult run_multi_column_test(size_t total_data_size, 
                                  size_t num_columns,
                                  double null_probability,
                                  size_t num_runs) {
    std::cout << "=== Multi-Column Coalesce Test ===" 
              << std::endl << std::endl;
    
    size_t total_rows = total_data_size / sizeof(int32_t);
    size_t rows_per_column = total_rows / num_columns;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Number of columns: " << num_columns << std::endl;
    std::cout << "  Rows per column: " << rows_per_column << std::endl;
    std::cout << "  Total data size: " 
              << format_bytes(num_columns * rows_per_column * 
                              sizeof(int32_t))
              << std::endl;
    std::cout << "  NULL probability: " 
              << (null_probability * 100) << "%" << std::endl;
    std::cout << "  Number of runs: " << num_runs << std::endl;
    std::cout << std::endl;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    
    std::cout << "Generating and transferring " << num_columns 
              << " columns to GPU..." << std::endl;
    
    auto start_transfer = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.reserve(num_columns);
    
    size_t total_null_count = 0;
    
    for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        size_t col_null_count = 0;
        input_columns.push_back(
            generate_column(rows_per_column, null_probability, gen, 
                           data_dist, null_dist, col_null_count)
        );
        total_null_count += col_null_count;
    }
    
    cudaDeviceSynchronize();
    auto end_transfer = std::chrono::high_resolution_clock::now();
    
    auto transfer_duration = std::chrono::duration_cast<
        std::chrono::milliseconds>(end_transfer - start_transfer);
    
    std::cout << "  Data transfer time: " << transfer_duration.count() 
              << " ms" << std::endl << std::endl;
    
    // Warm-up runs
    std::cout << "Running warm-up iterations..." << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
            auto warmup = coalesce_column(input_columns[col_idx]->view());
        }
        cudaDeviceSynchronize();
    }
    
    // Timed runs
    std::cout << "Running " << num_runs << " timed iterations..." 
              << std::endl;
    
    std::vector<double> run_times;
    run_times.reserve(num_runs);
    
    for (size_t run_idx = 0; run_idx < num_runs; ++run_idx) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
            auto result = coalesce_column(input_columns[col_idx]->view());
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);
        run_times.push_back(duration.count() / 1000.0);
    }
    
    double avg_time = 0.0;
    for (double t : run_times) {
        avg_time += t;
    }
    avg_time /= run_times.size();
    
    double min_time = *std::min_element(run_times.begin(), 
                                         run_times.end());
    double max_time = *std::max_element(run_times.begin(), 
                                         run_times.end());
    
    double total_data_gb = num_columns * rows_per_column * 
                           sizeof(int32_t) / 1e9;
    double throughput = total_data_gb / (avg_time / 1000.0);
    
    std::cout << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Total coalesce time (avg): " << avg_time << " ms" 
              << std::endl;
    std::cout << "  Total coalesce time (min): " << min_time << " ms" 
              << std::endl;
    std::cout << "  Total coalesce time (max): " << max_time << " ms" 
              << std::endl;
    std::cout << "  Average time per column: " 
              << (avg_time / num_columns) << " ms" << std::endl;
    std::cout << "  Overall throughput: " << throughput << " GB/s" 
              << std::endl;
    
    TestResult result;
    result.coalesce_time_ms = avg_time;
    result.data_transfer_time_ms = transfer_duration.count();
    result.throughput_gbps = throughput;
    result.num_columns = num_columns;
    result.num_rows = rows_per_column;
    result.null_count = total_null_count;
    result.individual_run_times = run_times;
    
    return result;
}

int main() {
    // Initialize CUDA memory resource
    rmm::mr::cuda_memory_resource mr;
    rmm::mr::set_current_device_resource(&mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;  // 1GB
    constexpr double NULL_PROBABILITY = 0.2;         // 20% NULL
    constexpr size_t NUM_RUNS = 10;                  // Run each test 10 times
    
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << "CUDF Coalesce Performance Benchmark" << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 1: Single large column
    TestResult single_result = run_single_column_test(
        TOTAL_DATA_SIZE, 
        NULL_PROBABILITY,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 2: 100 columns
    TestResult multi_100_result = run_multi_column_test(
        TOTAL_DATA_SIZE,
        100,
        NULL_PROBABILITY,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 3: 200 columns
    TestResult multi_200_result = run_multi_column_test(
        TOTAL_DATA_SIZE,
        200,
        NULL_PROBABILITY,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 4: 300 columns
    TestResult multi_300_result = run_multi_column_test(
        TOTAL_DATA_SIZE,
        300,
        NULL_PROBABILITY,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << "=== Final Summary (GPU Coalesce Time Only) ===" 
              << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << std::left << std::setw(20) << "Configuration"
              << std::right << std::setw(15) << "Avg Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(15) << "Slowdown"
              << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    std::cout << std::left << std::setw(20) << "1 col"
              << std::right << std::setw(15) << std::fixed 
              << std::setprecision(3) << single_result.coalesce_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << single_result.throughput_gbps << " GB/s"
              << std::setw(12) << "1.00x"
              << std::endl;
    
    double slowdown_100 = multi_100_result.coalesce_time_ms / 
                          single_result.coalesce_time_ms;
    std::cout << std::left << std::setw(20) << "100 cols"
              << std::right << std::setw(15) << std::fixed 
              << std::setprecision(3) << multi_100_result.coalesce_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << multi_100_result.throughput_gbps << " GB/s"
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << slowdown_100 << "x"
              << std::endl;
    
    double slowdown_200 = multi_200_result.coalesce_time_ms / 
                          single_result.coalesce_time_ms;
    std::cout << std::left << std::setw(20) << "200 cols"
              << std::right << std::setw(15) << std::fixed 
              << std::setprecision(3) << multi_200_result.coalesce_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << multi_200_result.throughput_gbps << " GB/s"
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << slowdown_200 << "x"
              << std::endl;
    
    double slowdown_300 = multi_300_result.coalesce_time_ms / 
                          single_result.coalesce_time_ms;
    std::cout << std::left << std::setw(20) << "300 cols"
              << std::right << std::setw(15) << std::fixed 
              << std::setprecision(3) << multi_300_result.coalesce_time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << multi_300_result.throughput_gbps << " GB/s"
              << std::setw(12) << std::fixed << std::setprecision(2) 
              << slowdown_300 << "x"
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "Note: All times are GPU coalesce operation only, "
              << "excluding data transfer." << std::endl;
    std::cout << "      Each result is averaged over " << NUM_RUNS 
              << " runs." << std::endl;
    
    std::cout << std::endl 
              << "=== All tests completed successfully ===" << std::endl;
    
    return 0;
}
