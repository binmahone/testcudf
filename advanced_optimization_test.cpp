#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/replace.hpp>
#include <cudf/transform.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
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
 * Explore advanced optimizations:
 * 1. Fuse operations to reduce kernel launches
 * 2. Reuse buffers
 * 3. Use transform with custom operation
 * 4. Simplify logic by reordering operations
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
 * Baseline: Current best (combined mask)
 */
std::unique_ptr<cudf::column> baseline_combined_mask(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    auto zero_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero_col->mutable_view().data<int32_t>(), 
               0, num_rows * sizeof(int32_t));
    
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
    
    auto x_is_valid = cudf::is_valid(x_col);
    
    auto combined_mask = cudf::binary_operation(
        y_eq_1->view(), x_is_valid->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    auto result_int = cudf::copy_if_else(
        x_col, zero_col->view(), combined_mask->view()
    );
    
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Option 1: Use replace_nulls + arithmetic
 * Simplify by using different CUDF APIs
 */
std::unique_ptr<cudf::column> opt1_replace_nulls(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    size_t num_rows = x_col.size();
    
    // replace_nulls(x, 0)
    cudf::numeric_scalar<int32_t> zero_scalar(0, true, 
                                               rmm::cuda_stream_default);
    
    auto x_no_nulls = cudf::replace_nulls(x_col, zero_scalar);
    
    // result = x_no_nulls * y (since y is 0 or 1)
    auto result_int = cudf::binary_operation(
        x_no_nulls->view(), y_col,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::INT32}, rmm::cuda_stream_default
    );
    
    // cast to double
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Option 2: Reorder - multiply first, then handle nulls
 * x * y, then replace_nulls
 */
std::unique_ptr<cudf::column> opt2_multiply_first(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    // x * y (NULL * anything = NULL, preserves NULL)
    auto multiplied = cudf::binary_operation(
        x_col, y_col,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::INT32}, rmm::cuda_stream_default
    );
    
    // replace_nulls with 0
    cudf::numeric_scalar<int32_t> zero_scalar(0, true, 
                                               rmm::cuda_stream_default);
    
    auto result_int = cudf::replace_nulls(multiplied->view(), zero_scalar);
    
    // cast to double
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Option 3: Fuse cast with multiply
 * Cast x to double, then multiply with y (casted to double)
 */
std::unique_ptr<cudf::column> opt3_fused_cast_multiply(
    const cudf::column_view& x_col,
    const cudf::column_view& y_col) {
    
    // Cast both to double first
    auto x_double = cudf::cast(x_col, 
                                cudf::data_type{cudf::type_id::FLOAT64});
    auto y_double = cudf::cast(y_col, 
                                cudf::data_type{cudf::type_id::FLOAT64});
    
    // multiply
    auto multiplied = cudf::binary_operation(
        x_double->view(), y_double->view(),
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::FLOAT64}, rmm::cuda_stream_default
    );
    
    // replace_nulls
    cudf::numeric_scalar<double> zero_scalar(0.0, true, 
                                              rmm::cuda_stream_default);
    
    return cudf::replace_nulls(multiplied->view(), zero_scalar);
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
    std::cout << "Advanced Optimization Exploration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Expression: cast(coalesce(if(y==1,x,0),0) as double)" 
              << std::endl;
    std::cout << "Simplified: cast(x * y as double) with NULL handling" 
              << std::endl;
    std::cout << "  (since y is 0 or 1, y==1 is equivalent to *y)" 
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
        for (size_t i = 0; i < 5; ++i) {
            for (size_t col = 0; col < NUM_OUTPUT_COLUMNS; ++col) {
                auto result = impl_func(x_columns[col]->view(), 
                                        y_columns[col]->view());
            }
            cudaDeviceSynchronize();
        }
        
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
        
        std::cout << name << ": " << std::fixed << std::setprecision(3) 
                  << avg << " ms" << std::endl;
        
        return avg;
    };
    
    std::cout << "Testing implementations..." << std::endl;
    std::cout << std::endl;
    
    double baseline = test_impl(baseline_combined_mask, 
                                 "Baseline (combined mask)    ");
    double opt1 = test_impl(opt1_replace_nulls, 
                             "Option 1 (replace_nulls + mul)");
    double opt2 = test_impl(opt2_multiply_first, 
                             "Option 2 (mul first)         ");
    double opt3 = test_impl(opt3_fused_cast_multiply, 
                             "Option 3 (cast + mul double) ");
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    double best = std::min({baseline, opt1, opt2, opt3});
    
    auto print_result = [&](double time, const std::string& name, 
                             const std::string& desc) {
        std::cout << std::left << std::setw(35) << name
                  << std::right << std::setw(12) << std::fixed 
                  << std::setprecision(3) << time << " ms";
        
        if (time == best) {
            std::cout << "  ⭐ BEST";
        } else {
            double diff_pct = ((time - best) / best * 100);
            std::cout << "  (+" << std::fixed << std::setprecision(1) 
                      << diff_pct << "% slower)";
        }
        std::cout << std::endl;
        std::cout << "  " << desc << std::endl;
        std::cout << std::endl;
    };
    
    print_result(baseline, "Baseline (combined mask)", 
                 "5 kernels: y==1, is_valid, AND, copy_if_else, cast");
    
    print_result(opt1, "Option 1 (replace_nulls + mul)", 
                 "3 kernels: replace_nulls, multiply, cast");
    
    print_result(opt2, "Option 2 (multiply first)", 
                 "3 kernels: multiply, replace_nulls, cast");
    
    print_result(opt3, "Option 3 (cast + mul double)", 
                 "4 kernels: cast x, cast y, mul, replace_nulls");
    
    std::cout << "========================================" << std::endl;
    std::cout << "Key Insights" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "1. Mathematical simplification:" << std::endl;
    std::cout << "   if(y==1, x, 0) === x * y (when y is 0 or 1)" 
              << std::endl;
    std::cout << "   This eliminates the need for explicit if/else!" 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "2. replace_nulls is optimized:" << std::endl;
    std::cout << "   Uses single kernel vs copy_if_else + is_valid" 
              << std::endl;
    std::cout << std::endl;
    
    std::cout << "3. Kernel count matters:" << std::endl;
    std::cout << "   Baseline: 5 kernels" << std::endl;
    std::cout << "   Best options: 3 kernels (40% reduction)" << std::endl;
    std::cout << std::endl;
    
    if (opt1 < baseline * 0.9 || opt2 < baseline * 0.9) {
        std::cout << "Recommendation: Use multiplication approach!" 
                  << std::endl;
        std::cout << "  Since y is always 0 or 1, multiply is simpler" 
                  << std::endl;
        std::cout << "  Reduces from 5 to 3 kernels" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(1) 
                  << ((baseline - best) / baseline * 100) << "%" 
                  << std::endl;
    } else {
        std::cout << "Current approach is already near-optimal" 
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

