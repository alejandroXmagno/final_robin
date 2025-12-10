#!/bin/bash
# Installation script for trt-pose on Jetson
# This script installs all dependencies needed for GPU-accelerated pose estimation

set -e

echo "🚀 Installing trt-pose for Jetson..."
echo ""

# Check if we're on Jetson
if ! uname -a | grep -q tegra; then
    echo "⚠️  Warning: This script is designed for Jetson devices"
fi

# Step 1: Install PyTorch with CUDA support for Jetson
echo "📦 Step 1: Installing PyTorch with CUDA support..."
echo "   Note: PyTorch for Jetson must be installed from NVIDIA's repository"
echo "   Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "   Or run:"
echo "   wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl"
echo "   pip3 install torch-2.1.0-cp310-cp310-linux_aarch64.whl"
echo ""

# Step 2: Install torch2trt
echo "📦 Step 2: Installing torch2trt..."
if [ -d "/tmp/torch2trt" ]; then
    echo "   torch2trt already cloned, skipping..."
else
    cd /tmp
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
fi

# Set CUDA_HOME (Jetson typically has CUDA at /usr/local/cuda or /usr/local/cuda-12.X)
if [ -d "/usr/local/cuda-12.6" ]; then
    export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "   ⚠️  Warning: CUDA installation not found in standard locations"
    echo "   Please set CUDA_HOME manually"
fi

cd /tmp/torch2trt
# Install to user site-packages (no sudo needed, uses your PyTorch installation)
python3 setup.py install --user --plugins
echo "   ✅ torch2trt installed"
echo ""

# Step 3: Install trt_pose
echo "📦 Step 3: Installing trt_pose..."
if [ -d "/tmp/trt_pose" ]; then
    echo "   trt_pose already cloned, skipping..."
else
    cd /tmp
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
fi

cd /tmp/trt_pose
sudo python3 setup.py install
echo "   ✅ trt_pose installed"
echo ""

# Step 4: Download model
echo "📦 Step 4: Downloading pre-trained model..."
MODEL_PATH="$HOME/resnet18_baseline_att_224x224_A_epoch_249.pth"
if [ -f "$MODEL_PATH" ]; then
    echo "   Model already exists at $MODEL_PATH"
else
    echo "   Downloading model..."
    wget https://github.com/NVIDIA-AI-IOT/trt_pose/raw/master/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth -O "$MODEL_PATH" || {
        echo "   ⚠️  Failed to download model. You can download it manually from:"
        echo "   https://github.com/NVIDIA-AI-IOT/trt_pose/raw/master/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth"
    }
fi
echo ""

echo "✅ Installation complete!"
echo ""
echo "⚠️  IMPORTANT: Make sure PyTorch with CUDA is installed first!"
echo "   Check with: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "   Should print: True"

