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

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>

/**
 * Workload: cast(coalesce(if(y==1, x, 0), 0) as double)
 * Input: x1,y1, x2,y2, ..., x100,y100 (200 columns)
 * Output: 100 columns
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

// Generate x column (int with possible NULLs)
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

// Generate y column (0 or 1, no NULLs)
std::unique_ptr<cudf::column> generate_y_column(
    size_t num_rows,
    std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& binary_dist) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = binary_dist(gen);  // 0 or 1
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

/**
 * Perform: cast(coalesce(if(y==1, x, 0), 0) as double)
 * This is equivalent to: cast(if(y==1, coalesce(x,0), 0) as double)
 */
std::unique_ptr<cudf::column> conditional_coalesce_cast(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    // Step 1: coalesce(x, 0) - replace NULLs in x with 0
    auto zero_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        rmm::cuda_stream_default
    );
    cudaMemset(zero_column->mutable_view().data<int32_t>(), 
               0, num_rows * sizeof(int32_t));
    
    auto x_is_valid = cudf::is_valid(x_col);
    auto x_coalesced = cudf::copy_if_else(
        x_col,
        zero_column->view(),
        x_is_valid->view()
    );
    
    // Step 2: Create y==1 boolean mask
    auto one_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        rmm::cuda_stream_default
    );
    cudaMemset(one_column->mutable_view().data<int32_t>(), 
               1, num_rows * sizeof(int32_t));
    
    // y == 1
    auto y_eq_1 = cudf::binary_operation(
        y_col,
        one_column->view(),
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        rmm::cuda_stream_default
    );
    
    // Step 3: if(y==1, coalesced_x, 0)
    auto result_int = cudf::copy_if_else(
        x_coalesced->view(),
        zero_column->view(),
        y_eq_1->view()
    );
    
    // Step 4: Cast to DOUBLE
    auto result_double = cudf::cast(
        result_int->view(),
        cudf::data_type{cudf::type_id::FLOAT64}
    );
    
    return result_double;
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 
        3ULL * 1024 * 1024 * 1024  // 3GB pool (need more for intermediate)
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;  // 1GB
    constexpr size_t NUM_OUTPUT_COLUMNS = 100;
    constexpr size_t NUM_INPUT_COLUMNS = NUM_OUTPUT_COLUMNS * 2;  // 200
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_WARMUP = 5;
    constexpr size_t NUM_RUNS = 20;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_output = total_rows / NUM_OUTPUT_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Conditional Coalesce Cast Comparison" << std::endl;
    std::cout << "  Workload: cast(coalesce(if(y==1,x,0),0) as double)" 
              << std::endl;
    std::cout << "  (Using RMM Pool Memory Resource)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input columns: " << NUM_INPUT_COLUMNS 
              << " (x1,y1,x2,y2,...,x100,y100)" << std::endl;
    std::cout << "  Output columns: " << NUM_OUTPUT_COLUMNS << std::endl;
    std::cout << "  Input data: " << format_bytes(TOTAL_DATA_SIZE) 
              << " (INT32)" << std::endl;
    std::cout << "  Output data: " << format_bytes(TOTAL_DATA_SIZE * 2) 
              << " (DOUBLE)" << std::endl;
    std::cout << "  Rows per output col: " << rows_per_output << std::endl;
    std::cout << "  Warmup: " << NUM_WARMUP << std::endl;
    std::cout << "  Runs: " << NUM_RUNS << std::endl;
    std::cout << "  RMM Pool: 3GB" << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::cout << "Generating data..." << std::endl;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    std::uniform_int_distribution<int32_t> binary_dist(0, 1);
    
    // Generate x1,y1, x2,y2, ... pattern
    std::vector<std::unique_ptr<cudf::column>> x_columns;
    std::vector<std::unique_ptr<cudf::column>> y_columns;
    x_columns.reserve(NUM_OUTPUT_COLUMNS);
    y_columns.reserve(NUM_OUTPUT_COLUMNS);
    
    for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
        x_columns.push_back(
            generate_x_column(rows_per_output, NULL_PROBABILITY, 
                             gen, data_dist, null_dist)
        );
        y_columns.push_back(
            generate_y_column(rows_per_output, gen, binary_dist)
        );
    }
    cudaDeviceSynchronize();
    std::cout << "Data ready." << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Strategy 1: Individual processing (100 pairs)" 
              << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<double> times_s1;
    times_s1.reserve(NUM_WARMUP + NUM_RUNS);
    
    // Warmup
    std::cout << "Warmup:" << std::endl;
    for (size_t i = 0; i < NUM_WARMUP; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t col = 0; col < NUM_OUTPUT_COLUMNS; ++col) {
            auto result = conditional_coalesce_cast(
                x_columns[col]->view(),
                y_columns[col]->view()
            );
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double t = dur.count() / 1000.0;
        times_s1.push_back(t);
        std::cout << "  " << (i+1) << ": " << std::fixed 
                  << std::setprecision(3) << t << " ms" << std::endl;
    }
    
    // Timed runs
    std::cout << std::endl << "Timed runs:" << std::endl;
    for (size_t i = 0; i < NUM_RUNS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t col = 0; col < NUM_OUTPUT_COLUMNS; ++col) {
            auto result = conditional_coalesce_cast(
                x_columns[col]->view(),
                y_columns[col]->view()
            );
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double t = dur.count() / 1000.0;
        times_s1.push_back(t);
        std::cout << "  " << (i+1) << ": " << std::fixed 
                  << std::setprecision(3) << t << " ms" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Strategy 2: Concat->Process->Split(view)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Prepare concat: interleave x and y columns
    std::vector<cudf::column_view> x_views;
    std::vector<cudf::column_view> y_views;
    x_views.reserve(NUM_OUTPUT_COLUMNS);
    y_views.reserve(NUM_OUTPUT_COLUMNS);
    
    for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
        x_views.push_back(x_columns[i]->view());
        y_views.push_back(y_columns[i]->view());
    }
    
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(NUM_OUTPUT_COLUMNS - 1);
    for (size_t i = 1; i < NUM_OUTPUT_COLUMNS; ++i) {
        split_indices.push_back(i * rows_per_output);
    }
    
    std::vector<double> times_s2;
    times_s2.reserve(NUM_WARMUP + NUM_RUNS);
    
    // Warmup
    std::cout << "Warmup:" << std::endl;
    for (size_t i = 0; i < NUM_WARMUP; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto concat_x = cudf::concatenate(x_views);
        auto concat_y = cudf::concatenate(y_views);
        auto result = conditional_coalesce_cast(
            concat_x->view(),
            concat_y->view()
        );
        auto split_result = cudf::split(result->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double t = dur.count() / 1000.0;
        times_s2.push_back(t);
        std::cout << "  " << (i+1) << ": " << std::fixed 
                  << std::setprecision(3) << t << " ms" << std::endl;
    }
    
    // Timed runs
    std::cout << std::endl << "Timed runs:" << std::endl;
    for (size_t i = 0; i < NUM_RUNS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto concat_x = cudf::concatenate(x_views);
        auto concat_y = cudf::concatenate(y_views);
        auto result = conditional_coalesce_cast(
            concat_x->view(),
            concat_y->view()
        );
        auto split_result = cudf::split(result->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        double t = dur.count() / 1000.0;
        times_s2.push_back(t);
        std::cout << "  " << (i+1) << ": " << std::fixed 
                  << std::setprecision(3) << t << " ms" << std::endl;
    }
    
    // Calculate stats
    auto calc_stats = [](const std::vector<double>& times, size_t skip) {
        double sum = 0.0;
        double min_v = 1e9;
        double max_v = 0.0;
        for (size_t i = skip; i < times.size(); ++i) {
            sum += times[i];
            min_v = std::min(min_v, times[i]);
            max_v = std::max(max_v, times[i]);
        }
        double avg = sum / (times.size() - skip);
        double var = 0.0;
        for (size_t i = skip; i < times.size(); ++i) {
            var += (times[i] - avg) * (times[i] - avg);
        }
        var /= (times.size() - skip);
        return std::make_tuple(avg, min_v, max_v, std::sqrt(var));
    };
    
    auto [avg1, min1, max1, std1] = calc_stats(times_s1, NUM_WARMUP);
    auto [avg2, min2, max2, std2] = calc_stats(times_s2, NUM_WARMUP);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy 1 (Individual, 100 pairs):" << std::endl;
    std::cout << "  Avg: " << std::fixed << std::setprecision(3) << avg1 
              << " ms" << std::endl;
    std::cout << "  Min: " << min1 << " ms" << std::endl;
    std::cout << "  Max: " << max1 << " ms" << std::endl;
    std::cout << "  Std: " << std1 << " ms" << std::endl;
    std::cout << "  CV:  " << std::fixed << std::setprecision(1) 
              << (std1/avg1*100) << "%" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy 2 (Concat x2 -> Process -> Split):" << std::endl;
    std::cout << "  Avg: " << std::fixed << std::setprecision(3) << avg2 
              << " ms" << std::endl;
    std::cout << "  Min: " << min2 << " ms" << std::endl;
    std::cout << "  Max: " << max2 << " ms" << std::endl;
    std::cout << "  Std: " << std2 << " ms" << std::endl;
    std::cout << "  CV:  " << std::fixed << std::setprecision(1) 
              << (std2/avg2*100) << "%" << std::endl;
    std::cout << std::endl;
    
    if (avg1 < avg2) {
        double speedup = avg2 / avg1;
        std::cout << "Result: Strategy 1 is " << std::fixed 
                  << std::setprecision(2) 
                  << speedup << "x faster!" << std::endl;
    } else {
        double speedup = avg1 / avg2;
        std::cout << "Result: Strategy 2 is " << std::fixed 
                  << std::setprecision(2) 
                  << speedup << "x faster!" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Data for Plotting" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy1_times = [";
    for (size_t i = 0; i < times_s1.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << times_s1[i];
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy2_times = [";
    for (size_t i = 0; i < times_s2.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << times_s2[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << std::endl << "Done!" << std::endl;
    
    return 0;
}

