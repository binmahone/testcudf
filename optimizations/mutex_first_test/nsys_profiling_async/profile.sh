#!/bin/bash

# Script to collect nsys profiles for both concurrent and mutex modes
# Using ASYNC memory resource

if [ ! -f "build/concurrent_mode" ] || [ ! -f "build/mutex_mode" ]; then
    echo "Error: Executables not found. Please run build.sh first."
    exit 1
fi

OUTPUT_DIR="nsys_reports"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Collecting nsys profiles (ASYNC memory)"
echo "=========================================="
echo ""

# Profile concurrent mode
echo "Profiling CONCURRENT mode..."
nsys profile \
    --output="$OUTPUT_DIR/concurrent_mode" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --cuda-graph-trace=node \
    ./build/concurrent_mode

echo ""
echo "Concurrent mode profiling completed."
echo "Output: $OUTPUT_DIR/concurrent_mode.nsys-rep"
echo ""

# Profile mutex mode
echo "Profiling MUTEX mode..."
nsys profile \
    --output="$OUTPUT_DIR/mutex_mode" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --cuda-graph-trace=node \
    ./build/mutex_mode

echo ""
echo "Mutex mode profiling completed."
echo "Output: $OUTPUT_DIR/mutex_mode.nsys-rep"
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Profiles saved in: $OUTPUT_DIR/"
echo ""
echo "To view the reports:"
echo "  nsys-ui $OUTPUT_DIR/concurrent_mode.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/mutex_mode.nsys-rep"
echo ""
echo "To export statistics:"
echo "  nsys stats $OUTPUT_DIR/concurrent_mode.nsys-rep"
echo "  nsys stats $OUTPUT_DIR/mutex_mode.nsys-rep"

