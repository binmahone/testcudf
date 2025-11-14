#!/bin/bash

set -e

export CONDA_PREFIX=$HOME/anaconda3/envs/cudf_test

if [ ! -d "$CONDA_PREFIX" ]; then
    echo "Error: cudf_test environment not found!"
    echo "Please ensure cudf is installed."
    exit 1
fi

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc

cmake --build . -j$(nproc)

echo ""
echo "Build complete! Available tests:"
echo "  cd build && ./original_with_mutex_test  (Pool memory)"
echo "  cd build && ./original_async_test        (Async memory)"
echo "  cd build && ./original_arena_test        (Arena memory)"
echo ""
echo "Or run all tests:"
echo "  bash run_all.sh"

