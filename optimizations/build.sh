#!/bin/bash

set -e

echo "Building CUDF optimization benchmarks..."
echo ""

export CONDA_PREFIX=$HOME/anaconda3/envs/cudf_test

if [ ! -d "$CONDA_PREFIX" ]; then
    echo "Error: cudf_test environment not found!"
    echo "Run './setup_env.sh' first."
    exit 1
fi

mkdir -p build
cd build

cmake .. \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc

make -j$(nproc)

echo ""
echo "Build complete! Run './run_benchmarks.sh'"
