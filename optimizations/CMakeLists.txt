cmake_minimum_required(VERSION 3.18)
project(CUDFOptimizations LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)

add_executable(workload1_benchmark workload1_benchmark.cpp)
add_executable(workload2_benchmark workload2_benchmark.cpp)
add_executable(multithread_benchmark multithread_benchmark.cpp)
add_executable(kernel_submission_test kernel_submission_test.cpp)

target_link_libraries(workload1_benchmark PRIVATE cudf::cudf CUDA::cudart)
target_link_libraries(workload2_benchmark PRIVATE cudf::cudf CUDA::cudart)
target_link_libraries(multithread_benchmark PRIVATE cudf::cudf CUDA::cudart)
target_link_libraries(kernel_submission_test PRIVATE cudf::cudf CUDA::cudart)

set_target_properties(workload1_benchmark PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80")
set_target_properties(workload2_benchmark PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80")
set_target_properties(multithread_benchmark PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80")
set_target_properties(kernel_submission_test PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80")
