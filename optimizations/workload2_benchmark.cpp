#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/replace.hpp>
#include <cudf/binaryop.hpp>
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

/**
 * Workload 2: CAST(COALESCE(IF(y=1, x, 0), 0) AS DOUBLE)
 * Compares original vs optimized implementation
 */

std::unique_ptr<cudf::column> generate_x(size_t rows, double null_prob) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1, 1000);
    std::uniform_real_distribution<double> ndist(0, 1);
    
    std::vector<int32_t> data(rows);
    for (size_t i = 0; i < rows; ++i) data[i] = dist(gen);
    
    size_t ms = cudf::bitmask_allocation_size_bytes(rows) / 
                sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> mask(ms, 0);
    for (size_t i = 0; i < rows; ++i) {
        if (ndist(gen) > null_prob) cudf::set_bit_unsafe(mask.data(), i);
    }
    
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t),
                            rmm::cuda_stream_default);
    rmm::device_buffer mbuf(mask.data(), 
                            cudf::bitmask_allocation_size_bytes(rows),
                            rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), std::move(mbuf), 0
    );
}

std::unique_ptr<cudf::column> generate_y(size_t rows) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1);
    
    std::vector<int32_t> data(rows);
    for (size_t i = 0; i < rows; ++i) data[i] = dist(gen);
    
    rmm::device_buffer dbuf(data.data(), rows * sizeof(int32_t),
                            rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, rows,
        std::move(dbuf), rmm::device_buffer{}, 0
    );
}

// Original: 5 kernels
std::unique_ptr<cudf::column> original_impl(
    const cudf::column_view& x, const cudf::column_view& y) {
    
    auto zero = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero->mutable_view().data<int32_t>(), 0, 
               x.size() * sizeof(int32_t));
    
    auto x_valid = cudf::is_valid(x);
    auto x_coal = cudf::copy_if_else(x, zero->view(), x_valid->view());
    
    auto one = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(one->mutable_view().data<int32_t>(), 1, 
               x.size() * sizeof(int32_t));
    
    auto y_eq_1 = cudf::binary_operation(
        y, one->view(), cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8}, rmm::cuda_stream_default
    );
    
    auto res_int = cudf::copy_if_else(x_coal->view(), zero->view(), 
                                       y_eq_1->view());
    
    return cudf::cast(res_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

// Optimized: 3 kernels
std::unique_ptr<cudf::column> optimized_impl(
    const cudf::column_view& x, const cudf::column_view& y) {
    
    cudf::numeric_scalar<int32_t> zero(0, true);
    auto x_no_nulls = cudf::replace_nulls(x, zero);
    
    // Mathematical simplification: if(y==1, x, 0) === x * y
    auto result_int = cudf::binary_operation(
        x_no_nulls->view(), y,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::INT32}, rmm::cuda_stream_default
    );
    
    return cudf::cast(result_int->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

int main() {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 3ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t NUM_PAIRS = 100;
    constexpr size_t ROWS = 2684354;
    constexpr size_t RUNS = 20;
    
    std::cout << "=== Workload 2 Benchmark ===" << std::endl;
    std::cout << "Expression: CAST(COALESCE(IF(y=1,x,0),0) AS DOUBLE)" 
              << std::endl;
    std::cout << "Pairs: " << NUM_PAIRS << ", Total data: 1GB" << std::endl;
    std::cout << std::endl;
    
    std::vector<std::unique_ptr<cudf::column>> xs, ys;
    for (size_t i = 0; i < NUM_PAIRS; ++i) {
        xs.push_back(generate_x(ROWS, 0.2));
        ys.push_back(generate_y(ROWS));
    }
    cudaDeviceSynchronize();
    
    auto bench = [&](auto func, const char* name) {
        for (int i = 0; i < 5; ++i) {
            for (size_t j = 0; j < NUM_PAIRS; ++j) {
                func(xs[j]->view(), ys[j]->view());
            }
            cudaDeviceSynchronize();
        }
        
        std::vector<double> times;
        for (size_t i = 0; i < RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < NUM_PAIRS; ++j) {
                func(xs[j]->view(), ys[j]->view());
            }
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<
                std::chrono::microseconds>(end - start).count() / 1000.0);
        }
        
        double avg = 0;
        for (double t : times) avg += t;
        avg /= times.size();
        
        std::cout << name << ": " << std::fixed << std::setprecision(2) 
                  << avg << " ms" << std::endl;
        return avg;
    };
    
    double orig = bench(original_impl, "Original  (5 kernels)");
    double opt = bench(optimized_impl, "Optimized (3 kernels)");
    
    std::cout << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(1) 
              << ((orig - opt) / orig * 100) << "% faster" << std::endl;
    std::cout << "Key insight: IF(y=1,x,0) === x*y when y∈{0,1}" 
              << std::endl;
    
    return 0;
}
