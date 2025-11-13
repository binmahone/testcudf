#!/bin/bash

set -e

export LD_LIBRARY_PATH=$HOME/anaconda3/envs/cudf_test/lib:$LD_LIBRARY_PATH

cd build

echo "========================================"
echo "CUDF Coalesce Optimization Benchmarks"
echo "========================================"
echo ""

echo "[1/3] Workload 1: CAST(COALESCE(x,0) AS DOUBLE)"
echo "----------------------------------------"
./workload1_benchmark
echo ""
echo ""

echo "[2/3] Workload 2: CAST(COALESCE(IF(y=1,x,0),0) AS DOUBLE)"
echo "----------------------------------------"
./workload2_benchmark
echo ""
echo ""

echo "[3/3] Multi-threading: 1 vs 4 threads"
echo "----------------------------------------"
./multithread_benchmark
echo ""

echo "========================================"
echo "All benchmarks completed successfully!"
echo "========================================"
