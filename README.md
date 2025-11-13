# CUDF Coalesce Test

This program demonstrates the `coalesce(x, 0)` operation using libcudf.

## Description

The program creates a table with a column `x` containing some NULL values, 
then applies the coalesce operation which replaces NULL values with 0.

Input column: `[1, NULL, 3, NULL, 5, 6, NULL, 8]`
Output column: `[1, 0, 3, 0, 5, 6, 0, 8]`

## Prerequisites

- CUDA Toolkit (11.0 or later)
- libcudf installed
- CMake 3.18 or later
- C++17 compatible compiler

## Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/cudf/installation
make
```

If cudf is installed in a non-standard location, you may need to set:
```bash
export CMAKE_PREFIX_PATH=/path/to/cudf/lib/cmake
```

## Running

```bash
./coalesce_test
```

## Expected Output

```
=== CUDF Coalesce(x, 0) Test ===

Input column (x):
[1, NULL, 3, NULL, 5, 6, NULL, 8]

Result of coalesce(x, 0):
[1, 0, 3, 0, 5, 6, 0, 8]

Table created with 1 column(s) and 8 row(s)

=== Test completed successfully ===
```

## Notes

- The program uses `cudf::replace_nulls()` to implement the coalesce operation
- Memory management is handled by RMM (RAPIDS Memory Manager)
- The null mask follows CUDF's convention: 1 = valid, 0 = null

