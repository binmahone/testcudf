#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>

/**
 * Simple direct test of memory fragmentation and efficiency
 */

std::string format_gb(size_t bytes) {
    return std::to_string(bytes / (1024.0 * 1024 * 1024)) + " GB";
}

void print_memory_state(const std::string& label) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "  " << std::setw(30) << std::left << label 
              << ": Free = " << std::setw(10) << format_gb(free)
              << " / " << format_gb(total) << std::endl;
}

void test_allocator(const std::string& name) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    print_memory_state("Before allocator setup");
    
    std::cout << "\n--- Phase 1: Large varying-size allocations ---" 
              << std::endl;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist(1000000, 10000000);
    
    {
        std::vector<std::unique_ptr<cudf::column>> columns;
        
        // Allocate 20 columns of varying sizes
        for (int i = 0; i < 20; ++i) {
            size_t rows = dist(gen);
            columns.push_back(cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::INT32}, rows));
            
            if (i == 0 || i == 9 || i == 19) {
                print_memory_state("After " + std::to_string(i+1) + 
                                   " allocations");
            }
        }
        
        std::cout << "\n--- Phase 2: Free every other one (create holes) ---"
                  << std::endl;
        
        // Free every other column (creates fragmentation)
        for (size_t i = 1; i < columns.size(); i += 2) {
            columns[i].reset();
        }
        cudaDeviceSynchronize();
        print_memory_state("After freeing odd-indexed");
        
        std::cout << "\n--- Phase 3: Try to reuse freed space ---" 
                  << std::endl;
        
        // Try to allocate in the holes
        for (int i = 0; i < 5; ++i) {
            size_t rows = 5000000;
            columns.push_back(cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::INT32}, rows));
        }
        print_memory_state("After 5 more allocations");
        
        std::cout << "\n--- Phase 4: Free all ---" << std::endl;
        columns.clear();
        cudaDeviceSynchronize();
        print_memory_state("After freeing all");
    }
    
    std::cout << "\n--- Phase 5: Test memory is truly freed ---" 
              << std::endl;
    
    // Small pause to let allocator settle
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    print_memory_state("After settling");
    
    // Allocate again to see if memory is reused
    {
        std::vector<std::unique_ptr<cudf::column>> columns;
        for (int i = 0; i < 10; ++i) {
            columns.push_back(cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::INT32}, 5000000));
        }
        print_memory_state("After second round");
    }
    
    cudaDeviceSynchronize();
    print_memory_state("Final state");
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Simple Memory Fragmentation Test" << std::endl;
    std::cout << "Direct observation of GPU memory usage" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "\nInitial GPU state: " << format_gb(free) 
              << " free / " << format_gb(total) << " total" << std::endl;
    
    // Test 1: Pool
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 1: POOL MEMORY RESOURCE (8GB pre-allocated)" 
                  << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024
        );
        rmm::mr::set_current_device_resource(&pool_mr);
        
        test_allocator("Pool");
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test 2: Async
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 2: CUDA ASYNC MEMORY RESOURCE (driver-managed)" 
                  << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_async_memory_resource async_mr;
        rmm::mr::set_current_device_resource(&async_mr);
        
        test_allocator("Async");
    }
    
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test 3: Arena
    {
        std::cout << "\n\n" << std::string(70, '#') << std::endl;
        std::cout << "TEST 3: ARENA MEMORY RESOURCE (8GB pre-allocated)" 
                  << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource> arena_mr(
            &cuda_mr, 8ULL * 1024 * 1024 * 1024
        );
        rmm::mr::set_current_device_resource(&arena_mr);
        
        test_allocator("Arena");
    }
    
    std::cout << "\n\n" << std::string(70, '=') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nKey observations:" << std::endl;
    std::cout << "1. Pool/Arena: Pre-allocate 8GB, then allocate from pool"
              << std::endl;
    std::cout << "   - Fast allocation (no actual GPU malloc)" << std::endl;
    std::cout << "   - Memory usage predictable" << std::endl;
    std::cout << "   - May waste space if workload < 8GB" << std::endl;
    
    std::cout << "\n2. Async: Allocates on-demand from driver" << std::endl;
    std::cout << "   - Slower allocation (actual GPU malloc each time)"
              << std::endl;
    std::cout << "   - Memory usage adapts to workload" << std::endl;
    std::cout << "   - May have driver-level overhead" << std::endl;
    
    std::cout << "\n3. Fragmentation:" << std::endl;
    std::cout << "   - Watch if memory returns to baseline after freeing"
              << std::endl;
    std::cout << "   - Pool/Arena: Holes stay in pre-allocated region"
              << std::endl;
    std::cout << "   - Async: Driver may defragment automatically"
              << std::endl;
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

