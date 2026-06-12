#!/bin/bash
# Unified Setup Script for Qiskit Aer (Source Build)

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="qiskit_aer"

echo "=========================================================="
echo "Starting Unified Qiskit Aer Setup (GPU + Tensor Networks)"
echo "Targeting CUDA 13.0 Drivers with CUDA 12.x Toolkit"
echo "=========================================================="

# 1. Create/Update Environment
if mamba env list | grep -q "^$ENV_NAME "; then
  echo "--> Step 1: Environment '$ENV_NAME' already exists. Updating..."
  mamba env update -n $ENV_NAME -f "$SCRIPT_DIR/qiskit_aer_env.yml" -y
else
  echo "--> Step 1: Environment '$ENV_NAME' does not exist. Creating..."
  mamba env create -n $ENV_NAME -f "$SCRIPT_DIR/qiskit_aer_env.yml" -y
fi

# 2. Activate Environment logic for the script
# We source conda.sh to make 'conda activate' available in the subshell
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/mamba.sh"
mamba activate $ENV_NAME

# 3. Install Qiskit Base
echo "--> Step 2: Installing Qiskit 1.x and build dependencies..."
uv pip install -v "qiskit>=2.0" pybind11 scikit-build-core

# 4. Source Build of Qiskit Aer
echo "--> Step 3: Building Qiskit Aer from source..."
BUILD_DIR="$PROJECT_ROOT/qiskit_aer_build"

# Clean previous build if it exists
if [ -d "$BUILD_DIR" ]; then
  echo "    Removing existing build directory..."
  rm -rf "$BUILD_DIR"
fi

git clone https://github.com/Qiskit/qiskit-aer.git "$BUILD_DIR"
cd "$BUILD_DIR"

# Set environment variables for the compiler to find HPC libraries
export CUQUANTUM_ROOT=$CONDA_PREFIX
export CUTENSORNET_ROOT=$CONDA_PREFIX
export BLAS_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "    Compiling with cuTensorNet support enabled..."
# We use scikit-build-core via uv to install from the local directory
SKBUILD_CONFIGURE_OPTIONS="-DAER_ENABLE_CUQUANTUM=True -DAER_THRUST_BACKEND=CUDA" \
  uv pip install -v .
cd ../
rm -fr "$BUILD_DIR"

echo "=========================================================="
echo "SUCCESS: Qiskit Aer built and verified."
echo "NOTE: To use this environment in your jobs, always run:"
echo "      conda activate $ENV_NAME"
echo "      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "=========================================================="
