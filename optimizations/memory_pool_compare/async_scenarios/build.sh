#!/bin/bash

echo "Building Async Advantage Test..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake .. && make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "Build complete!"
    echo ""
    echo "Available tests:"
    echo "  ./build/extreme_fragmentation_test       - EXTREME 10MB-2GB, 8 threads (NEW!)"
    echo "  ./build/multithread_fragmentation_test   - Multi-threaded fragmentation"
    echo "  ./build/async_advantage_test             - Dynamic workload patterns"
    echo "  ./build/fragmentation_performance_test   - Single-thread fragmentation impact"
    echo "  ./build/fragmentation_test               - Memory efficiency analysis"
    echo "  ./build/simple_fragmentation_test        - Direct GPU memory observation"
else
    echo "Build failed!"
    exit 1
fi

