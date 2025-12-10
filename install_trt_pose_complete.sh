#!/bin/bash
# Complete installation script for trt_pose on Jetson JetPack 6
# This script properly installs PyTorch, torch2trt, and trt_pose

set -e

echo "🚀 Complete trt_pose Installation for Jetson JetPack 6"
echo "=================================================="
echo ""

# Detect JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' | head -1)
    echo "📦 Detected JetPack: $JETPACK_VERSION"
else
    echo "⚠️  Warning: Not running on Jetson?"
    exit 1
fi

# Detect CUDA version
if [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_HOME="/usr/local/cuda-12.6"
    CUDA_VERSION="12.6"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_HOME="/usr/local/cuda-12"
    CUDA_VERSION="12"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    CUDA_VERSION=$(readlink -f /usr/local/cuda | grep -oP 'cuda-\K[0-9.]+' || echo "unknown")
else
    echo "❌ CUDA not found!"
    exit 1
fi

echo "📦 CUDA detected: $CUDA_VERSION at $CUDA_HOME"
export CUDA_HOME

# Step 1: Fix MPI libraries and uninstall old PyTorch
echo ""
echo "📦 Step 1: Fixing MPI libraries and removing old PyTorch installations..."

# Fix MPI version mismatch (PyTorch may expect libmpi.so.20 but system has libmpi.so.40)
if [ -f "/usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20" ]; then
    echo "   Creating MPI compatibility symlinks..."
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20 2>/dev/null || echo "   ⚠️  Could not create MPI symlink (may need sudo)"
fi
if [ -f "/usr/lib/aarch64-linux-gnu/libmpi.so.40" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/libmpi.so.20" ]; then
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi.so.40 /usr/lib/aarch64-linux-gnu/libmpi.so.20 2>/dev/null || echo "   ⚠️  Could not create MPI symlink (may need sudo)"
fi

echo "   Removing old PyTorch installations..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip3 uninstall -y torch2trt trt-pose 2>/dev/null || true

# Step 2: Install proper PyTorch for JetPack 6
echo ""
echo "📦 Step 2: Installing PyTorch with CUDA for JetPack 6..."
echo "   This may take several minutes..."

# For JetPack 6 with cuDNN 9, use PyTorch 2.8.0 from jetson-ai-lab.io
echo "   Installing PyTorch 2.8.0 (compatible with cuDNN 9) from jetson-ai-lab.io..."
pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 || {
    echo "⚠️  jetson-ai-lab.io failed, trying alternative PyPI index..."
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 || {
        echo "❌ Failed to install PyTorch 2.8.0"
        echo "   Trying fallback: PyTorch 2.3.0 from NVIDIA developer downloads..."
        PYTORCH_WHL="torch-2.3.0a0+40ec155e58.nv24.03.13384722-cp310-cp310-linux_aarch64.whl"
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/$PYTORCH_WHL"
        cd /tmp
        if [ ! -f "$PYTORCH_WHL" ]; then
            wget -q --show-progress "$PYTORCH_URL" -O "$PYTORCH_WHL" || {
                echo "❌ All PyTorch installation methods failed!"
                exit 1
            }
        fi
        pip3 install --no-cache-dir /tmp/$PYTORCH_WHL || {
            echo "❌ Failed to install PyTorch"
            exit 1
        }
    }
}

# Verify PyTorch installation
echo ""
echo "   Verifying PyTorch installation..."
python3 -c "import torch; print(f'   ✅ PyTorch {torch.__version__} installed'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}')" || {
    echo "❌ PyTorch installation failed!"
    exit 1
}

# Step 3: Install torch2trt
echo ""
echo "📦 Step 3: Installing torch2trt..."
cd /tmp
if [ -d "torch2trt" ]; then
    rm -rf torch2trt
fi

git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
export CUDA_HOME
python3 setup.py install --user --plugins || {
    echo "❌ torch2trt installation failed!"
    exit 1
}
echo "   ✅ torch2trt installed"

# Step 4: Install trt_pose
echo ""
echo "📦 Step 4: Installing trt_pose..."
cd /tmp
if [ -d "trt_pose" ]; then
    rm -rf trt_pose
fi

git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
cd trt_pose
python3 setup.py install --user || {
    echo "❌ trt_pose installation failed!"
    exit 1
}
echo "   ✅ trt_pose installed"

# Step 5: Download model
echo ""
echo "📦 Step 5: Downloading pre-trained model..."
MODEL_PATH="$HOME/resnet18_baseline_att_224x224_A_epoch_249.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "   Installing gdown for Google Drive downloads..."
    pip3 install --user --no-cache-dir gdown >/dev/null 2>&1 || echo "   (gdown installation skipped, will try alternative method)"
    
    echo "   Downloading model from Google Drive (this may take a few minutes)..."
    # Google Drive file ID from trt_pose README: 1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd
    if command -v gdown >/dev/null 2>&1; then
        gdown "https://drive.google.com/uc?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd" -O "$MODEL_PATH" || {
            echo "   ⚠️  gdown failed, trying wget method..."
            wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd" -O "$MODEL_PATH" || {
                echo "⚠️  Failed to download model automatically."
                echo "   Please download manually using one of these methods:"
                echo "   1. Install gdown: pip3 install --user gdown"
                echo "      Then run: gdown https://drive.google.com/uc?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd -O $MODEL_PATH"
                echo "   2. Or visit: https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd"
                echo "      And save the file to: $MODEL_PATH"
            }
        }
    else
        # Fallback: Try wget method (may not work for large files due to Google Drive restrictions)
        wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd" -O "$MODEL_PATH" 2>&1 | grep -q "confirm=" && {
            # Google Drive requires confirmation for large files
            CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
            wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd" -O "$MODEL_PATH" && rm -f /tmp/cookies.txt || {
                echo "⚠️  Failed to download model automatically."
                echo "   Please install gdown: pip3 install --user gdown"
                echo "   Then run: gdown https://drive.google.com/uc?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd -O $MODEL_PATH"
            }
        } || {
            echo "⚠️  Failed to download model automatically."
            echo "   Please install gdown: pip3 install --user gdown"
            echo "   Then run: gdown https://drive.google.com/uc?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd -O $MODEL_PATH"
        }
    fi
    
    if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
        echo "   ✅ Model downloaded successfully to $MODEL_PATH"
    else
        echo "   ⚠️  Model file not found or empty. Please download manually."
    fi
else
    echo "   Model already exists at $MODEL_PATH"
fi

# Step 6: Final verification
echo ""
echo "📦 Step 6: Final verification..."
python3 << 'PYTHON_VERIFY'
import sys
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
    
    import torch2trt
    print("   ✅ torch2trt imported")
    
    import trt_pose.coco
    import trt_pose.models
    print("   ✅ trt_pose imported")
    
    print("")
    print("🎉 All components installed successfully!")
    print("   trt_pose is ready to use with GPU acceleration!")
    
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)
PYTHON_VERIFY

echo ""
echo "✅ Installation complete!"
echo ""
echo "You can now use trt_pose with GPU acceleration in your ROS2 node."

