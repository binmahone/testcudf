#!/bin/bash

echo "========================================================"
echo "Running All Memory Resource Tests on Original Workloads"
echo "========================================================"
echo ""

cd build

echo "=== TEST 1: Pool Memory Resource ==="
echo ""
./original_with_mutex_test
echo ""
echo ""

echo "=== TEST 2: CUDA Async Memory Resource ==="
echo ""
./original_async_test
echo ""
echo ""

echo "=== TEST 3: Arena Memory Resource ==="
echo ""
./original_arena_test
echo ""
echo ""

echo "========================================================"
echo "All tests completed!"
echo "========================================================"

