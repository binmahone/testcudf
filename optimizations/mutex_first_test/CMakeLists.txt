cmake_minimum_required(VERSION 3.18)
project(MutexFirstTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf REQUIRED)

add_executable(original_with_mutex_test original_with_mutex_test.cpp)
add_executable(original_async_test original_async_test.cpp)
add_executable(original_arena_test original_arena_test.cpp)

target_link_libraries(original_with_mutex_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(original_async_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_link_libraries(original_arena_test
    PRIVATE
    cudf::cudf
    CUDA::cudart
)

target_compile_options(original_with_mutex_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_compile_options(original_async_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_compile_options(original_arena_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

