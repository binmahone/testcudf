#!/bin/bash

# Setup CUDF environment for benchmarks
# Run this once to create the conda environment

set -e

echo "========================================"
echo "Setting up CUDF environment"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found!"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

ENV_NAME="cudf_test"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment."
        echo "Setup complete!"
        exit 0
    fi
fi

echo "Creating conda environment: ${ENV_NAME}"
echo ""

# Create environment with all dependencies
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "Installing CUDF and build tools..."
echo ""

# Install libcudf and compilers
conda install -n ${ENV_NAME} \
    -c rapidsai \
    -c conda-forge \
    -c nvidia \
    libcudf \
    cuda-version=12.2 \
    cmake \
    gcc_linux-64=12 \
    gxx_linux-64=12 \
    -y

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo "Then run: ./build.sh"

