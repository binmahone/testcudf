cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  MemoryResourceComparison
  VERSION 1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)

add_executable(memory_resource_test memory_resource_test.cpp)

target_link_libraries(memory_resource_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(memory_resource_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

message(STATUS "Build configuration complete!")
message(STATUS "For async advantage tests, see async_scenarios/ subdirectory")

