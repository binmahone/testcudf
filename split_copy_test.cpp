#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
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

/**
 * Test to understand if cudf::split involves GPU memory copy
 * Tests three scenarios:
 * 1. split() only - returns column_view (zero-copy)
 * 2. split() + create column from view - may involve copy
 * 3. Compare memory addresses to verify zero-copy
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

int main() {
    rmm::mr::cuda_memory_resource mr;
    rmm::mr::set_current_device_resource(&mr);
    
    constexpr size_t TOTAL_SIZE = 1073741824;  // 1GB
    constexpr size_t NUM_SPLITS = 200;
    constexpr size_t NUM_RUNS = 10;
    
    size_t total_rows = TOTAL_SIZE / sizeof(int32_t);
    size_t rows_per_split = total_rows / NUM_SPLITS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "CUDF Split Memory Copy Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Total data: " << format_bytes(TOTAL_SIZE) << std::endl;
    std::cout << "  Number of splits: " << NUM_SPLITS << std::endl;
    std::cout << "  Rows per split: " << rows_per_split << std::endl;
    std::cout << std::endl;
    
    // Create a large column filled with sequential values
    std::cout << "Creating test data..." << std::endl;
    auto test_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        total_rows,
        cudf::mask_state::UNALLOCATED,
        rmm::cuda_stream_default
    );
    
    // Fill with sequential values
    std::vector<int32_t> host_data(total_rows);
    for (size_t i = 0; i < total_rows; ++i) {
        host_data[i] = static_cast<int32_t>(i);
    }
    cudaMemcpy(test_column->mutable_view().data<int32_t>(),
               host_data.data(),
               total_rows * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    
    // Get original data pointer
    const void* original_ptr = test_column->view().data<int32_t>();
    std::cout << "Original column data pointer: " << original_ptr 
              << std::endl;
    std::cout << std::endl;
    
    // Prepare split indices
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(NUM_SPLITS - 1);
    for (size_t i = 1; i < NUM_SPLITS; ++i) {
        split_indices.push_back(i * rows_per_split);
    }
    
    // Test 1: split() only - just get views
    std::cout << "=== Test 1: split() returning views only ===" 
              << std::endl;
    
    std::vector<double> split_only_times;
    split_only_times.reserve(NUM_RUNS);
    
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto split_views = cudf::split(test_column->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);
        split_only_times.push_back(duration.count() / 1000.0);
        
        // Verify first split points to original memory
        if (run == 0) {
            const void* first_split_ptr = split_views[0].data<int32_t>();
            std::cout << "  First split data pointer: " << first_split_ptr 
                      << std::endl;
            if (first_split_ptr == original_ptr) {
                std::cout << "  ✓ Zero-copy confirmed: "
                          << "Same pointer as original!" << std::endl;
            } else {
                std::cout << "  ✗ Memory copied: Different pointer!" 
                          << std::endl;
            }
        }
    }
    
    double avg_split_only = 0.0;
    for (double t : split_only_times) avg_split_only += t;
    avg_split_only /= split_only_times.size();
    
    std::cout << "  Average time: " << avg_split_only << " ms" 
              << std::endl;
    std::cout << std::endl;
    
    // Test 2: split() + materialize to columns
    std::cout << "=== Test 2: split() + create columns from views ===" 
              << std::endl;
    
    std::vector<double> split_materialize_times;
    std::vector<double> materialize_only_times;
    split_materialize_times.reserve(NUM_RUNS);
    materialize_only_times.reserve(NUM_RUNS);
    
    for (size_t run = 0; run < NUM_RUNS; ++run) {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Split
        auto start_split = std::chrono::high_resolution_clock::now();
        auto split_views = cudf::split(test_column->view(), split_indices);
        cudaDeviceSynchronize();
        auto end_split = std::chrono::high_resolution_clock::now();
        
        // Materialize views to columns
        auto start_materialize = std::chrono::high_resolution_clock::now();
        std::vector<std::unique_ptr<cudf::column>> materialized_columns;
        materialized_columns.reserve(split_views.size());
        for (auto& view : split_views) {
            materialized_columns.push_back(
                std::make_unique<cudf::column>(view)
            );
        }
        cudaDeviceSynchronize();
        auto end_materialize = std::chrono::high_resolution_clock::now();
        
        auto split_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_split - start_split);
        auto materialize_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_materialize - start_materialize);
        auto total_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_materialize - start_total);
        
        split_materialize_times.push_back(total_dur.count() / 1000.0);
        materialize_only_times.push_back(materialize_dur.count() / 1000.0);
        
        // Check if materialized column has different pointer
        if (run == 0) {
            const void* materialized_ptr = 
                materialized_columns[0]->view().data<int32_t>();
            std::cout << "  First materialized column pointer: " 
                      << materialized_ptr << std::endl;
            if (materialized_ptr == original_ptr) {
                std::cout << "  ✓ Still zero-copy: "
                          << "Materialized column uses same memory!" 
                          << std::endl;
            } else {
                std::cout << "  ✗ Memory was copied during materialization!" 
                          << std::endl;
            }
        }
    }
    
    double avg_materialize = 0.0;
    for (double t : materialize_only_times) avg_materialize += t;
    avg_materialize /= materialize_only_times.size();
    
    double avg_total = 0.0;
    for (double t : split_materialize_times) avg_total += t;
    avg_total /= split_materialize_times.size();
    
    std::cout << "  Average split time: " << avg_split_only << " ms" 
              << std::endl;
    std::cout << "  Average materialize time: " << avg_materialize << " ms" 
              << std::endl;
    std::cout << "  Average total time: " << avg_total << " ms" 
              << std::endl;
    std::cout << std::endl;
    
    // Test 3: Verify data integrity to ensure views work correctly
    std::cout << "=== Test 3: Verify data integrity ===" << std::endl;
    
    auto split_views = cudf::split(test_column->view(), split_indices);
    
    // Check first and last values of first split
    std::vector<int32_t> first_split_data(rows_per_split);
    cudaMemcpy(first_split_data.data(),
               split_views[0].data<int32_t>(),
               rows_per_split * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    
    std::cout << "  First split - first value: " << first_split_data[0] 
              << " (expected: 0)" << std::endl;
    std::cout << "  First split - last value: " 
              << first_split_data[rows_per_split - 1]
              << " (expected: " << (rows_per_split - 1) << ")" 
              << std::endl;
    
    // Check second split
    std::vector<int32_t> second_split_data(rows_per_split);
    cudaMemcpy(second_split_data.data(),
               split_views[1].data<int32_t>(),
               rows_per_split * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    
    std::cout << "  Second split - first value: " << second_split_data[0] 
              << " (expected: " << rows_per_split << ")" << std::endl;
    
    std::cout << std::endl;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Operation breakdown:" << std::endl;
    std::cout << "  1. split() only (views): " << avg_split_only << " ms" 
              << std::endl;
    std::cout << "  2. Materialize views:    " << avg_materialize << " ms" 
              << std::endl;
    std::cout << "  3. Total:                " << avg_total << " ms" 
              << std::endl;
    std::cout << std::endl;
    
    double materialize_percentage = (avg_materialize / avg_total) * 100.0;
    std::cout << "Materialization overhead: " 
              << std::fixed << std::setprecision(1) 
              << materialize_percentage << "% of total time" << std::endl;
    
    // Theoretical memory copy time
    double theoretical_copy_time = (TOTAL_SIZE / 1e9) / 
                                   (500.0);  // Assume ~500 GB/s bandwidth
    std::cout << std::endl;
    std::cout << "Theoretical full memory copy time (500GB/s): " 
              << (theoretical_copy_time * 1000) << " ms" << std::endl;
    
    if (avg_materialize < theoretical_copy_time * 1000 * 0.5) {
        std::cout << "Conclusion: Likely NO GPU memory copy "
                  << "(time too fast)" << std::endl;
    } else {
        std::cout << "Conclusion: Likely involves GPU memory copy" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

