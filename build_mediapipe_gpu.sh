#!/bin/bash
# Build MediaPipe with GPU support for Jetson
# This script builds MediaPipe from source with OpenGL ES and GPU acceleration enabled

set -e

echo "🔧 Building MediaPipe with GPU support for Jetson..."
echo ""

# Check if we're on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This doesn't appear to be a Jetson device"
    echo "   GPU support may not work correctly"
fi

# Check if venv exists
if [ ! -d "blazepose_venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./setup_blazepose_venv.sh first"
    exit 1
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source blazepose_venv/bin/activate

# Install build dependencies
echo "📦 Installing build dependencies..."
sudo apt-get update

# Core build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-numpy

# Function to install packages, ignoring missing ones
install_packages() {
    local packages=("$@")
    for pkg in "${packages[@]}"; do
        if sudo apt-get install -y "$pkg" 2>/dev/null; then
            echo "   ✅ Installed: $pkg"
        else
            echo "   ⚠️  Skipped: $pkg (not available)"
        fi
    done
}

# OpenGL/OpenGL ES libraries (Jetson-specific packages)
echo "📦 Installing OpenGL/OpenGL ES libraries..."
install_packages \
    libegl1-mesa-dev \
    libgles-dev \
    libgles1 \
    libgles2 \
    libgles2-mesa-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    libglib2.0-dev

# GStreamer (for video processing)
echo "📦 Installing GStreamer libraries..."
install_packages \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# FFmpeg libraries
echo "📦 Installing FFmpeg libraries..."
install_packages \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev

# Image processing libraries
echo "📦 Installing image processing libraries..."
install_packages \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev

# Protocol buffers
echo "📦 Installing Protocol Buffer libraries..."
install_packages \
    libprotobuf-dev \
    protobuf-compiler

# Set up Bazel (MediaPipe's build system)
echo "📦 Installing Bazel for ARM64..."
BAZEL_VERSION="6.4.0"
if ! command -v bazel &> /dev/null; then
    # For ARM64 (Jetson), we need to use the ARM64 installer or build from source
    # Check architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        echo "   Detected ARM64 architecture (Jetson)"
        # For ARM64, we can use Bazelisk or install via bazelisk
        if ! command -v bazelisk &> /dev/null; then
            echo "   Installing Bazelisk (Bazel wrapper)..."
            wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-arm64 -O /tmp/bazelisk
            chmod +x /tmp/bazelisk
            sudo mv /tmp/bazelisk /usr/local/bin/bazel
        else
            sudo ln -sf $(which bazelisk) /usr/local/bin/bazel
        fi
    else
        # For x86_64
        wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        sudo ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    fi
else
    echo "✅ Bazel already installed"
fi

# Clone MediaPipe if not already present
MEDIAPIPE_DIR="mediapipe"
if [ ! -d "$MEDIAPIPE_DIR" ]; then
    echo "📦 Cloning MediaPipe repository..."
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    git checkout v0.10.18  # Match the version you have
else
    echo "✅ MediaPipe directory already exists"
    cd mediapipe
    git fetch
    git checkout v0.10.18
fi

# Build MediaPipe Python package with GPU support
echo "🔨 Building MediaPipe with GPU support..."
echo "   This may take 30-60 minutes..."

# Set environment variable to enable GPU (MediaPipe setup.py checks this)
export MEDIAPIPE_DISABLE_GPU=0

# Configure for Jetson (ARM64)
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH=$(python3 -c "import sys; print(sys.path[-1])")

# Build MediaPipe using setup.py with GPU support
# The setup.py will use Bazel internally with the correct GPU flags
echo "   Building with GPU support (this will take 30-60 minutes)..."
echo "   ⚠️  Note: MediaPipe Tasks API GPU support on Jetson is experimental"
echo "   If this fails, the test script will automatically fall back to CPU"

# Check if setup.py exists
if [ ! -f "mediapipe/setup.py" ]; then
    echo "❌ setup.py not found in mediapipe directory"
    echo "   MediaPipe may not have been cloned correctly"
    exit 1
fi

cd mediapipe

# Build using pip install from source (this uses setup.py internally)
echo "   Installing MediaPipe from source with GPU support..."
pip uninstall -y mediapipe 2>/dev/null || true

# Install with GPU enabled - setup.py will detect MEDIAPIPE_DISABLE_GPU=0
# Use absolute path to ensure we're in the right directory
MEDIAPIPE_DISABLE_GPU=0 pip install "$(pwd)" --no-build-isolation --verbose 2>&1 | tee /tmp/mediapipe_build.log || {
    echo ""
    echo "❌ Build failed. Checking error log..."
    tail -50 /tmp/mediapipe_build.log
    echo ""
    echo "⚠️  MediaPipe Tasks API GPU support on Jetson may not be fully supported yet."
    echo "   The test script (test_blazepose.py) will automatically use CPU mode."
    echo "   For better GPU performance, consider using:"
    echo "   - trt-pose (see INSTALL_TRT_POSE.md)"
    echo "   - openpifpaf (see test_openpifpaf.py)"
    echo ""
    echo "   Reinstalling CPU-only MediaPipe from PyPI..."
    cd ..
    pip install mediapipe || {
        echo "❌ Failed to install MediaPipe even from PyPI"
        exit 1
    }
    echo "✅ MediaPipe (CPU-only) reinstalled from PyPI"
    exit 0
}

cd ..

echo ""
echo "✅ MediaPipe with GPU support installed!"
echo ""
echo "Now test with: python test_blazepose.py"

