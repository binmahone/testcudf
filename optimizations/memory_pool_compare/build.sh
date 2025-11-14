#!/bin/bash

echo "Building Memory Resource Comparison Test..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake .. && make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "Build complete! Run './build/memory_resource_test'"
else
    echo "Build failed!"
    exit 1
fi

