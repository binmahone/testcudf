cmake_minimum_required(VERSION 3.18)
project(nsys_profiling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)

# Concurrent mode executable
add_executable(concurrent_mode concurrent_mode.cpp)
target_link_libraries(concurrent_mode
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(concurrent_mode PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Mutex mode executable
add_executable(mutex_mode mutex_mode.cpp)
target_link_libraries(mutex_mode
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(mutex_mode PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

