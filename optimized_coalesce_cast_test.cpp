#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
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
 * Optimize: cast(coalesce(x, 0) as double)
 * Current: is_valid + copy_if_else + cast (3 kernels)
 * Optimized: replace_nulls + cast (2 kernels!)
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
    size_t null_count = 0;
    
    for (size_t i = 0; i < num_rows; ++i) {
        if (null_dist(gen) > null_probability) {
            cudf::set_bit_unsafe(null_mask.data(), i);
        } else {
            null_count++;
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
        std::move(data_buffer), std::move(mask_buffer), null_count
    );
}

/**
 * Current implementation: is_valid + copy_if_else + cast
 */
std::unique_ptr<cudf::column> current_impl(
    const cudf::column_view& input_column) {
    
    size_t num_rows = input_column.size();
    
    auto zero_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero_column->mutable_view().data<int32_t>(), 
               0, num_rows * sizeof(int32_t));
    
    auto is_valid_mask = cudf::is_valid(input_column);
    auto coalesced = cudf::copy_if_else(
        input_column, zero_column->view(), is_valid_mask->view()
    );
    
    return cudf::cast(coalesced->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Optimized: replace_nulls + cast (2 kernels only!)
 */
std::unique_ptr<cudf::column> optimized_impl(
    const cudf::column_view& input_column) {
    
    // replace_nulls(x, 0) - single optimized kernel
    cudf::numeric_scalar<int32_t> zero(0, true, rmm::cuda_stream_default);
    auto coalesced = cudf::replace_nulls(input_column, zero);
    
    // cast to double
    return cudf::cast(coalesced->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

/**
 * Alternative: Cast to double with NULL, then replace
 */
std::unique_ptr<cudf::column> alt_cast_then_replace(
    const cudf::column_view& input_column) {
    
    // Cast (preserves NULLs)
    auto as_double = cudf::cast(input_column, 
                                 cudf::data_type{cudf::type_id::FLOAT64});
    
    // Replace NULLs with 0.0
    cudf::numeric_scalar<double> zero(0.0, true, rmm::cuda_stream_default);
    return cudf::replace_nulls(as_double->view(), zero);
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 2ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;
    constexpr size_t NUM_COLUMNS = 100;
    constexpr double NULL_PROBABILITY = 0.2;
    constexpr size_t NUM_RUNS = 20;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_column = total_rows / NUM_COLUMNS;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Optimize: cast(coalesce(x,0) as double)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Data: " << NUM_COLUMNS << " columns, " 
              << format_bytes(TOTAL_DATA_SIZE) << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    for (size_t i = 0; i < NUM_COLUMNS; ++i) {
        input_columns.push_back(
            generate_column(rows_per_column, NULL_PROBABILITY, 
                           gen, data_dist, null_dist)
        );
    }
    cudaDeviceSynchronize();
    
    auto test_impl = [&](auto impl_func, const std::string& name) {
        // Warmup
        for (size_t i = 0; i < 5; ++i) {
            for (size_t col = 0; col < NUM_COLUMNS; ++col) {
                auto result = impl_func(input_columns[col]->view());
            }
            cudaDeviceSynchronize();
        }
        
        // Timed
        std::vector<double> times;
        for (size_t i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t col = 0; col < NUM_COLUMNS; ++col) {
                auto result = impl_func(input_columns[col]->view());
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
        
        std::cout << name << ": " << std::fixed << std::setprecision(3) 
                  << avg << " ms (min: " << min_t << ", max: " << max_t 
                  << ")" << std::endl;
        
        return avg;
    };
    
    std::cout << "Testing implementations..." << std::endl;
    std::cout << std::endl;
    
    double time_current = test_impl(current_impl, 
                                     "Current (is_valid + copy_if_else + cast)");
    double time_opt = test_impl(optimized_impl, 
                                 "Optimized (replace_nulls + cast)       ");
    double time_alt = test_impl(alt_cast_then_replace, 
                                 "Alternative (cast + replace_nulls)     ");
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    double best = std::min({time_current, time_opt, time_alt});
    
    auto show_result = [&](double time, const std::string& name, 
                            int kernels) {
        std::cout << name << ": " << std::fixed << std::setprecision(3) 
                  << time << " ms (" << kernels << " kernels)";
        if (time == best) {
            std::cout << " ⭐ BEST";
        } else {
            std::cout << " (+" << std::fixed << std::setprecision(1) 
                      << ((time - best) / best * 100) << "%)";
        }
        std::cout << std::endl;
    };
    
    show_result(time_current, "Current", 3);
    show_result(time_opt, "Optimized (coalesce first)", 2);
    show_result(time_alt, "Alternative (cast first)", 2);
    
    std::cout << std::endl;
    std::cout << "Kernel breakdown:" << std::endl;
    std::cout << "  Current:     is_valid -> copy_if_else -> cast" 
              << std::endl;
    std::cout << "  Optimized:   replace_nulls -> cast" << std::endl;
    std::cout << "  Alternative: cast -> replace_nulls" << std::endl;
    std::cout << std::endl;
    
    double improvement = (time_current - best) / time_current * 100.0;
    
    std::cout << "Best implementation saves: " << std::fixed 
              << std::setprecision(1) << improvement << "%" << std::endl;
    std::cout << "Time saved per 100 columns: " << std::fixed 
              << std::setprecision(3) << (time_current - best) << " ms" 
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "Recommendation:" << std::endl;
    if (best == time_opt) {
        std::cout << "  Use replace_nulls(x, 0) + cast" << std::endl;
        std::cout << "  Benefits:" << std::endl;
        std::cout << "    - replace_nulls is optimized single kernel" 
                  << std::endl;
        std::cout << "    - Avoids is_valid + copy_if_else overhead" 
                  << std::endl;
        std::cout << "    - 33% fewer kernels (2 vs 3)" << std::endl;
    } else if (best == time_alt) {
        std::cout << "  Use cast(x) + replace_nulls(0.0)" << std::endl;
        std::cout << "  Benefits:" << std::endl;
        std::cout << "    - Cast preserves NULLs automatically" << std::endl;
        std::cout << "    - replace_nulls on double" << std::endl;
    } else {
        std::cout << "  Current implementation is optimal" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
}

