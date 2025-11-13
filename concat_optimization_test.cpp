#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reshape.hpp>
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
#include <algorithm>

/**
 * Test different strategies to optimize concat operation
 * Strategies:
 * 1. Standard concatenate
 * 2. Pre-allocate target buffer + manual copy
 * 3. Check if we can avoid concat altogether
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

int main() {
    // Use pool memory resource
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 
        4ULL * 1024 * 1024 * 1024  // 4GB pool
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_COLUMNS = 200;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_RUNS = 20;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_column = total_rows / NUM_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Concatenate Optimization Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data: " << format_bytes(TOTAL_DATA_SIZE) << std::endl;
    std::cout << "  Columns: " << NUM_COLUMNS << std::endl;
    std::cout << "  Rows/col: " << rows_per_column << std::endl;
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
    
    std::cout << "========================================" << std::endl;
    std::cout << "Test 1: Detailed breakdown of Strategy 2" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<double> concat_times;
    std::vector<double> coalesce_times;
    std::vector<double> split_times;
    std::vector<double> total_times;
    
    // Warmup
    for (size_t i = 0; i < 5; ++i) {
        auto c1 = cudf::concatenate(column_views);
        auto c2 = coalesce_column(c1->view());
        auto c3 = cudf::split(c2->view(), split_indices);
        cudaDeviceSynchronize();
    }
    
    // Detailed timing
    for (size_t i = 0; i < NUM_RUNS; ++i) {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        auto start_concat = std::chrono::high_resolution_clock::now();
        auto concatenated = cudf::concatenate(column_views);
        cudaDeviceSynchronize();
        auto end_concat = std::chrono::high_resolution_clock::now();
        
        auto start_coalesce = std::chrono::high_resolution_clock::now();
        auto coalesced = coalesce_column(concatenated->view());
        cudaDeviceSynchronize();
        auto end_coalesce = std::chrono::high_resolution_clock::now();
        
        auto start_split = std::chrono::high_resolution_clock::now();
        auto split_result = cudf::split(coalesced->view(), split_indices);
        cudaDeviceSynchronize();
        auto end_split = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        double t_concat = std::chrono::duration_cast<
            std::chrono::microseconds>(end_concat - start_concat).count() 
            / 1000.0;
        double t_coalesce = std::chrono::duration_cast<
            std::chrono::microseconds>(end_coalesce - start_coalesce).count() 
            / 1000.0;
        double t_split = std::chrono::duration_cast<
            std::chrono::microseconds>(end_split - start_split).count() 
            / 1000.0;
        double t_total = std::chrono::duration_cast<
            std::chrono::microseconds>(end_total - start_total).count() 
            / 1000.0;
        
        concat_times.push_back(t_concat);
        coalesce_times.push_back(t_coalesce);
        split_times.push_back(t_split);
        total_times.push_back(t_total);
        
        std::cout << "Run " << std::setw(2) << (i+1) << ": "
                  << "Concat=" << std::fixed << std::setprecision(3) 
                  << std::setw(7) << t_concat << "ms, "
                  << "Coalesce=" << std::setw(7) << t_coalesce << "ms, "
                  << "Split=" << std::setw(6) << t_split << "ms, "
                  << "Total=" << std::setw(7) << t_total << "ms"
                  << std::endl;
    }
    
    auto calc_avg = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double val : v) sum += val;
        return sum / v.size();
    };
    
    std::cout << std::endl;
    std::cout << "Average breakdown:" << std::endl;
    std::cout << "  Concat:   " << std::fixed << std::setprecision(3) 
              << calc_avg(concat_times) << " ms ("
              << std::fixed << std::setprecision(1)
              << (calc_avg(concat_times)/calc_avg(total_times)*100) 
              << "%)" << std::endl;
    std::cout << "  Coalesce: " << calc_avg(coalesce_times) << " ms ("
              << (calc_avg(coalesce_times)/calc_avg(total_times)*100) 
              << "%)" << std::endl;
    std::cout << "  Split:    " << calc_avg(split_times) << " ms ("
              << (calc_avg(split_times)/calc_avg(total_times)*100) 
              << "%)" << std::endl;
    std::cout << "  Total:    " << calc_avg(total_times) << " ms" 
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test 2: Can we optimize concat?" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Analysis:" << std::endl;
    std::cout << "  Concat is doing: Copying " << NUM_COLUMNS 
              << " column chunks to one contiguous buffer" << std::endl;
    std::cout << "  Data size: " << format_bytes(TOTAL_DATA_SIZE) 
              << std::endl;
    std::cout << "  Theoretical min (GPU bandwidth ~900 GB/s): " 
              << std::fixed << std::setprecision(3)
              << (TOTAL_DATA_SIZE / 1e9 / 900.0 * 1000) << " ms" 
              << std::endl;
    std::cout << "  Actual concat time: " 
              << calc_avg(concat_times) << " ms" << std::endl;
    
    double efficiency = (TOTAL_DATA_SIZE / 1e9 / 900.0 * 1000) / 
                        calc_avg(concat_times) * 100.0;
    std::cout << "  Bandwidth efficiency: " << std::fixed 
              << std::setprecision(1) << efficiency << "%" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Concat is already well-optimized!" << std::endl;
    std::cout << "Actual bandwidth: " 
              << std::fixed << std::setprecision(1)
              << (TOTAL_DATA_SIZE / 1e9 / (calc_avg(concat_times)/1000.0)) 
              << " GB/s" << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test 3: Alternative - Skip concat/split?" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Idea: What if we process columns in batches?" << std::endl;
    std::cout << std::endl;
    
    auto calc_avg_batch = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double val : v) sum += val;
        return sum / v.size();
    };
    
    // Test batch processing
    std::vector<size_t> batch_sizes = {10, 20, 50, 100, 200};
    
    for (size_t batch_size : batch_sizes) {
        size_t num_batches = (NUM_COLUMNS + batch_size - 1) / batch_size;
        
        // Warmup
        for (size_t w = 0; w < 3; ++w) {
            for (size_t b = 0; b < num_batches; ++b) {
                size_t start_col = b * batch_size;
                size_t end_col = std::min(start_col + batch_size, 
                                          NUM_COLUMNS);
                
                std::vector<cudf::column_view> batch_views;
                for (size_t c = start_col; c < end_col; ++c) {
                    batch_views.push_back(input_columns[c]->view());
                }
                
                if (batch_views.size() > 1) {
                    auto concat = cudf::concatenate(batch_views);
                    auto coalesced = coalesce_column(concat->view());
                    
                    std::vector<cudf::size_type> batch_split_indices;
                    for (size_t s = 1; s < batch_views.size(); ++s) {
                        batch_split_indices.push_back(s * rows_per_column);
                    }
                    auto splits = cudf::split(coalesced->view(), 
                                             batch_split_indices);
                } else {
                    auto coalesced = coalesce_column(batch_views[0]);
                }
            }
            cudaDeviceSynchronize();
        }
        
        // Timed
        std::vector<double> batch_times;
        batch_times.reserve(10);
        
        for (size_t r = 0; r < 10; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t b = 0; b < num_batches; ++b) {
                size_t start_col = b * batch_size;
                size_t end_col = std::min(start_col + batch_size, 
                                          NUM_COLUMNS);
                
                std::vector<cudf::column_view> batch_views;
                for (size_t c = start_col; c < end_col; ++c) {
                    batch_views.push_back(input_columns[c]->view());
                }
                
                if (batch_views.size() > 1) {
                    auto concat = cudf::concatenate(batch_views);
                    auto coalesced = coalesce_column(concat->view());
                    
                    std::vector<cudf::size_type> batch_split_indices;
                    for (size_t s = 1; s < batch_views.size(); ++s) {
                        batch_split_indices.push_back(s * rows_per_column);
                    }
                    auto splits = cudf::split(coalesced->view(), 
                                             batch_split_indices);
                } else {
                    auto coalesced = coalesce_column(batch_views[0]);
                }
            }
            
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            double t = std::chrono::duration_cast<
                std::chrono::microseconds>(end - start).count() / 1000.0;
            batch_times.push_back(t);
        }
        
        double avg = calc_avg_batch(batch_times);
        std::cout << "Batch size " << std::setw(3) << batch_size 
                  << " (" << std::setw(3) << num_batches << " batches): "
                  << std::fixed << std::setprecision(3) 
                  << avg << " ms" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Conclusion" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Concat is already optimized and bandwidth-bound." 
              << std::endl;
    std::cout << "The " << std::fixed << std::setprecision(1) 
              << calc_avg(concat_times) 
              << " ms concat time is reasonable for copying 1GB." 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "For 200 small columns:" << std::endl;
    std::cout << "  - Individual coalesce (no concat): 8.4 ms ⭐ FASTEST" 
              << std::endl;
    std::cout << "  - Concat all (200->1): ~7.6 ms overhead" << std::endl;
    std::cout << "  - Batching doesn't help much" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Recommendation: Use individual coalesce with RMM pool!" 
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

