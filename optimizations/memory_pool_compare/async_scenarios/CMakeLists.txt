cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  AsyncAdvantageTest
  VERSION 1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)

add_executable(async_advantage_test async_advantage_test.cpp)

target_link_libraries(async_advantage_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(async_advantage_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

add_executable(fragmentation_test fragmentation_test.cpp)

target_link_libraries(fragmentation_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(fragmentation_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

add_executable(simple_fragmentation_test simple_fragmentation_test.cpp)

target_link_libraries(simple_fragmentation_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(simple_fragmentation_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

add_executable(fragmentation_performance_test fragmentation_performance_test.cpp)

target_link_libraries(fragmentation_performance_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(fragmentation_performance_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

add_executable(multithread_fragmentation_test multithread_fragmentation_test.cpp)

target_link_libraries(multithread_fragmentation_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(multithread_fragmentation_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

add_executable(extreme_fragmentation_test extreme_fragmentation_test.cpp)

target_link_libraries(extreme_fragmentation_test
  PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(extreme_fragmentation_test 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIE>
)

message(STATUS "Async tests build configured!")

