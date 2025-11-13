cmake_minimum_required(VERSION 3.18)
project(CoalesceTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find cudf
# You may need to set CUDF_ROOT or CMAKE_PREFIX_PATH to point to cudf installation
find_package(cudf REQUIRED)

# Add executables
add_executable(coalesce_test coalesce_test.cpp)
add_executable(coalesce_strategy_test coalesce_strategy_test.cpp)
add_executable(split_copy_test split_copy_test.cpp)
add_executable(coalesce_comparison coalesce_comparison.cpp)
add_executable(memory_leak_test memory_leak_test.cpp)
add_executable(rmm_pool_test rmm_pool_test.cpp)
add_executable(virtual_column_test virtual_column_test.cpp)
add_executable(concat_optimization_test concat_optimization_test.cpp)
add_executable(conditional_coalesce_test conditional_coalesce_test.cpp)
add_executable(multithreaded_test multithreaded_test.cpp)
add_executable(multithread_scalability multithread_scalability.cpp)
add_executable(memory_requirement_test memory_requirement_test.cpp)
add_executable(optimized_impl_test optimized_impl_test.cpp)
add_executable(advanced_optimization_test advanced_optimization_test.cpp)
add_executable(optimized_coalesce_cast_test optimized_coalesce_cast_test.cpp)

# Link libraries
target_link_libraries(coalesce_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(coalesce_strategy_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(split_copy_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(coalesce_comparison
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(memory_leak_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(rmm_pool_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(virtual_column_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(concat_optimization_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(conditional_coalesce_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(multithreaded_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(multithread_scalability
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(memory_requirement_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(optimized_impl_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(advanced_optimization_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(optimized_coalesce_cast_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(coalesce_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(coalesce_strategy_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(split_copy_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(coalesce_comparison PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(memory_leak_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(rmm_pool_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(virtual_column_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(concat_optimization_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(conditional_coalesce_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(multithreaded_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(multithread_scalability PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(memory_requirement_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(optimized_impl_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(advanced_optimization_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

set_target_properties(optimized_coalesce_cast_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)

