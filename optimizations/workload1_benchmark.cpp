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

/**
 * Workload 1: CAST(COALESCE(x, 0) AS DOUBLE)
 * Compares original vs optimized implementation
 */

std::unique_ptr<cudf::column> generate_column(
    size_t num_rows, double null_prob) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1, 1000);
    std::uniform_real_distribution<double> ndist(0, 1);
    
    std::vector<int32_t> data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) data[i] = dist(gen);
    
    size_t mask_size = cudf::bitmask_allocation_size_bytes(num_rows) / 
                       sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> mask(mask_size, 0);
    for (size_t i = 0; i < num_rows; ++i) {
        if (ndist(gen) > null_prob) cudf::set_bit_unsafe(mask.data(), i);
    }
    
    rmm::device_buffer dbuf(data.data(), num_rows * sizeof(int32_t),
                            rmm::cuda_stream_default);
    rmm::device_buffer mbuf(mask.data(), 
                            cudf::bitmask_allocation_size_bytes(num_rows),
                            rmm::cuda_stream_default);
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, num_rows,
        std::move(dbuf), std::move(mbuf), 0
    );
}

// Original: 3 kernels
std::unique_ptr<cudf::column> original_impl(const cudf::column_view& x) {
    auto zero = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, x.size(),
        cudf::mask_state::UNALLOCATED, rmm::cuda_stream_default
    );
    cudaMemset(zero->mutable_view().data<int32_t>(), 0, 
               x.size() * sizeof(int32_t));
    
    auto mask = cudf::is_valid(x);
    auto coalesced = cudf::copy_if_else(x, zero->view(), mask->view());
    return cudf::cast(coalesced->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

// Optimized: 2 kernels
std::unique_ptr<cudf::column> optimized_impl(const cudf::column_view& x) {
    cudf::numeric_scalar<int32_t> zero(0, true);
    auto coalesced = cudf::replace_nulls(x, zero);
    return cudf::cast(coalesced->view(), 
                      cudf::data_type{cudf::type_id::FLOAT64});
}

int main() {
    // CRITICAL: Use RMM pool for stable performance
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr, 2ULL * 1024 * 1024 * 1024
    );
    rmm::mr::set_current_device_resource(&pool_mr);
    
    constexpr size_t NUM_COLS = 100;
    constexpr size_t ROWS = 2684354;  // ~1GB total
    constexpr size_t RUNS = 20;
    
    std::cout << "=== Workload 1 Benchmark ===" << std::endl;
    std::cout << "Expression: CAST(COALESCE(x, 0) AS DOUBLE)" << std::endl;
    std::cout << "Columns: " << NUM_COLS << ", Total data: 1GB" << std::endl;
    std::cout << std::endl;
    
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (size_t i = 0; i < NUM_COLS; ++i) {
        cols.push_back(generate_column(ROWS, 0.2));
    }
    cudaDeviceSynchronize();
    
    auto bench = [&](auto func, const char* name) {
        for (int i = 0; i < 5; ++i) {
            for (auto& c : cols) func(c->view());
            cudaDeviceSynchronize();
        }
        
        std::vector<double> times;
        for (size_t i = 0; i < RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            for (auto& c : cols) func(c->view());
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
    
    double orig = bench(original_impl, "Original  (3 kernels)");
    double opt = bench(optimized_impl, "Optimized (2 kernels)");
    
    std::cout << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(1) 
              << ((orig - opt) / orig * 100) << "% faster" << std::endl;
    
    return 0;
}
