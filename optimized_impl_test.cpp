#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/null_mask.hpp>
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
 * Test different implementations of: cast(coalesce(if(y==1, x, 0), 0) as double)
 * 
 * Current: Multiple steps with temporary buffers
 * Optimized: Fewer steps, less memory
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
    size_t num_rows, double null_probability,
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
    
    rmm::device_buffer data_buffer(host_data.data(), 
                                     num_rows * sizeof(int32_t), 
                                     rmm::cuda_stream_default);
    rmm::device_buffer mask_buffer(null_mask.data(), 
                                     cudf::bitmask_allocation_size_bytes(
                                         num_rows),
                                     rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        std::move(data_buffer), std::move(mask_buffer), 0
    );
}

std::unique_ptr<cudf::column> generate_y_column(
    size_t num_rows, std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& binary_dist) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = binary_dist(gen);
    }
    
    rmm::device_buffer data_buffer(host_data.data(), 
                                     num_rows * sizeof(int32_t), 
                                     rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        std::move(data_buffer), rmm::device_buffer{}, 0
    );
}

/**
 * Implementation 1: Current approach (multiple steps)
 * Steps: coalesce -> y==1 -> if -> cast
 */
std::unique_ptr<cudf::column> impl1_multi_step(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    // coalesce(x, 0)
    auto zero_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero_col->mutable_view().data<int32_t>(), 
               0, num_rows * sizeof(int32_t));
    
    auto x_is_valid = cudf::is_valid(x_col);
    auto x_coalesced = cudf::copy_if_else(
        x_col, zero_col->view(), x_is_valid->view()
    );
    
    // y == 1
    auto one_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(one_col->mutable_view().data<int32_t>(), 
               1, num_rows * sizeof(int32_t));
    
    auto y_eq_1 = cudf::binary_operation(
        y_col, one_col->view(), cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    // if(y==1, coalesced_x, 0)
    auto result_int = cudf::copy_if_else(
        x_coalesced->view(), zero_col->view(), y_eq_1->view()
    );
    
    // cast to double
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Implementation 2: Optimized - combine masks first
 * Combine (y==1) AND is_valid(x) into single mask
 */
std::unique_ptr<cudf::column> impl2_combined_mask(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    auto zero_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero_col->mutable_view().data<int32_t>(), 
               0, num_rows * sizeof(int32_t));
    
    // Create y==1 mask
    auto one_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(one_col->mutable_view().data<int32_t>(), 
               1, num_rows * sizeof(int32_t));
    
    auto y_eq_1 = cudf::binary_operation(
        y_col, one_col->view(), cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    // Create is_valid(x) mask
    auto x_is_valid = cudf::is_valid(x_col);
    
    // Combine masks: (y==1) AND is_valid(x)
    auto combined_mask = cudf::binary_operation(
        y_eq_1->view(), x_is_valid->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    // Single copy_if_else: if(combined, x, 0)
    auto result_int = cudf::copy_if_else(
        x_col, zero_col->view(), combined_mask->view()
    );
    
    // cast to double
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Implementation 3: Direct approach - cast first then operate
 * Idea: Work with doubles from the start
 */
std::unique_ptr<cudf::column> impl3_cast_first(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    // Cast x to double first (handles NULL preservation)
    auto x_double = cudf::cast(x_col, 
                                cudf::data_type{cudf::type_id::FLOAT64});
    
    // Create 0.0 and 1 columns
    auto zero_double = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero_double->mutable_view().data<double>(), 
               0, num_rows * sizeof(double));
    
    auto one_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(one_col->mutable_view().data<int32_t>(), 
               1, num_rows * sizeof(int32_t));
    
    // y==1
    auto y_eq_1 = cudf::binary_operation(
        y_col, one_col->view(), cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    // is_valid(x_double)
    auto x_valid = cudf::is_valid(x_double->view());
    
    // (y==1) AND is_valid(x)
    auto combined = cudf::binary_operation(
        y_eq_1->view(), x_valid->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    // if(combined, x_double, 0.0)
    return cudf::copy_if_else(
        x_double->view(), zero_double->view(), combined->view()
    );
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 4ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_OUTPUT_COLUMNS = 100;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_RUNS = 20;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_output = total_rows / NUM_OUTPUT_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Implementation Optimization Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Testing: cast(coalesce(if(y==1,x,0),0) as double)" 
              << std::endl;
    std::cout << "Data: 100 pairs, " << format_bytes(TOTAL_DATA_SIZE) 
              << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    std::uniform_int_distribution<int32_t> binary_dist(0, 1);
    
    std::vector<std::unique_ptr<cudf::column>> x_columns;
    std::vector<std::unique_ptr<cudf::column>> y_columns;
    
    for (size_t i = 0; i < NUM_OUTPUT_COLUMNS; ++i) {
        x_columns.push_back(generate_x_column(
            rows_per_output, NULL_PROBABILITY, gen, data_dist, null_dist
        ));
        y_columns.push_back(generate_y_column(
            rows_per_output, gen, binary_dist
        ));
    }
    cudaDeviceSynchronize();
    
    auto test_impl = [&](auto impl_func, const std::string& name) {
        std::cout << name << ":" << std::endl;
        
        // Warmup
        for (size_t i = 0; i < 5; ++i) {
            for (size_t col = 0; col < NUM_OUTPUT_COLUMNS; ++col) {
                auto result = impl_func(x_columns[col]->view(), 
                                        y_columns[col]->view());
            }
            cudaDeviceSynchronize();
        }
        
        // Timed runs
        std::vector<double> times;
        for (size_t i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t col = 0; col < NUM_OUTPUT_COLUMNS; ++col) {
                auto result = impl_func(x_columns[col]->view(), 
                                        y_columns[col]->view());
            }
            cudaDeviceSynchronize();
            
            auto end = std::chrono::high_resolution_clock::now();
            double t = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count() / 1000.0;
            times.push_back(t);
        }
        
        double avg = 0.0;
        for (double t : times) avg += t;
        avg /= times.size();
        
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        std::cout << "  Avg: " << std::fixed << std::setprecision(3) 
                  << avg << " ms" << std::endl;
        std::cout << "  Min: " << min_t << " ms" << std::endl;
        std::cout << "  Max: " << max_t << " ms" << std::endl;
        std::cout << std::endl;
        
        return avg;
    };
    
    std::cout << "========================================" << std::endl;
    std::cout << "Running tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    double time1 = test_impl(impl1_multi_step, 
                              "Impl 1: Multi-step (current)");
    
    double time2 = test_impl(impl2_combined_mask, 
                              "Impl 2: Combined mask");
    
    double time3 = test_impl(impl3_cast_first, 
                              "Impl 3: Cast first");
    
    std::cout << "========================================" << std::endl;
    std::cout << "Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    double best = std::min({time1, time2, time3});
    
    std::cout << "Implementation 1 (current):     " << std::fixed 
              << std::setprecision(3) << time1 << " ms";
    if (time1 == best) std::cout << " ⭐ BEST";
    std::cout << std::endl;
    
    std::cout << "Implementation 2 (combined):    " << time2 << " ms";
    if (time2 == best) std::cout << " ⭐ BEST";
    std::cout << " (" << std::fixed << std::setprecision(1) 
              << ((time1 - time2) / time1 * 100) << "% improvement)"
              << std::endl;
    
    std::cout << "Implementation 3 (cast first):  " << time3 << " ms";
    if (time3 == best) std::cout << " ⭐ BEST";
    std::cout << " (" << std::fixed << std::setprecision(1) 
              << ((time1 - time3) / time1 * 100) << "% improvement)"
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "Analysis:" << std::endl;
    
    std::cout << "  Impl 1 steps:" << std::endl;
    std::cout << "    1. is_valid(x)" << std::endl;
    std::cout << "    2. copy_if_else (coalesce x)" << std::endl;
    std::cout << "    3. y == 1" << std::endl;
    std::cout << "    4. copy_if_else (conditional)" << std::endl;
    std::cout << "    5. cast to double" << std::endl;
    std::cout << "    Total: ~5 kernels, 3 temp buffers" << std::endl;
    std::cout << std::endl;
    
    std::cout << "  Impl 2 steps:" << std::endl;
    std::cout << "    1. y == 1" << std::endl;
    std::cout << "    2. is_valid(x)" << std::endl;
    std::cout << "    3. (y==1) AND is_valid(x)" << std::endl;
    std::cout << "    4. copy_if_else (x, 0, combined)" << std::endl;
    std::cout << "    5. cast to double" << std::endl;
    std::cout << "    Total: ~5 kernels, 2-3 temp buffers" << std::endl;
    std::cout << std::endl;
    
    std::cout << "  Impl 3 steps:" << std::endl;
    std::cout << "    1. cast x to double" << std::endl;
    std::cout << "    2. y == 1" << std::endl;
    std::cout << "    3. is_valid(x_double)" << std::endl;
    std::cout << "    4. (y==1) AND is_valid" << std::endl;
    std::cout << "    5. copy_if_else (x_double, 0.0, combined)" 
              << std::endl;
    std::cout << "    Total: ~5 kernels, works with double throughout" 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "Recommendation:" << std::endl;
    if (best == time2) {
        std::cout << "  Use Implementation 2 (combined mask)" << std::endl;
        std::cout << "  Saves " << std::fixed << std::setprecision(1) 
                  << ((time1 - time2) / time1 * 100) 
                  << "% by reducing intermediate buffers" << std::endl;
    } else if (best == time3) {
        std::cout << "  Use Implementation 3 (cast first)" << std::endl;
        std::cout << "  Saves " << std::fixed << std::setprecision(1) 
                  << ((time1 - time3) / time1 * 100) 
                  << "% by working with doubles directly" << std::endl;
    } else {
        std::cout << "  Current implementation is already optimal" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

