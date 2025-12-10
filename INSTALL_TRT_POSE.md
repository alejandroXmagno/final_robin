# Installing trt-pose on Jetson

## Prerequisites

**IMPORTANT**: You must install PyTorch with CUDA support FIRST before installing torch2trt and trt-pose.

### Step 0: Install Required Dependencies

**For JetPack 6 (R36):**
```bash
# Install OpenMPI (required for PyTorch)
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin

# Install other dependencies
sudo apt-get install -y python3-pip python3-dev
```

### Step 1: Install PyTorch with CUDA Support

Your current PyTorch installation is CPU-only. You need to install the Jetson-compatible PyTorch with CUDA.

1. **Check your JetPack version:**
   ```bash
   cat /etc/nv_tegra_release
   # You should see: R36 (JetPack 6)
   ```

2. **Download and install PyTorch for Jetson:**
   
   **For JetPack 6 (R36) - Python 3.10:**
   ```bash
   # Uninstall CPU-only PyTorch first
   pip3 uninstall torch torchvision -y
   
   # Download PyTorch wheel for JetPack 6
   # Visit: https://forums.developer.nvidia.com/t/pytorch-packages-for-jetpack-6-1/309084
   # Or try:
   wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
   
   # Install it
   pip3 install torch-2.1.0-cp310-cp310-linux_aarch64.whl
   
   # If you get libmpi_cxx.so.20 error, install OpenMPI:
   sudo apt-get install -y libopenmpi-dev openmpi-bin
   
   # Verify CUDA is available
   python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   # Should print: CUDA available: True
   ```

   **For latest JetPack 6 PyTorch wheels, visit:**
   https://forums.developer.nvidia.com/t/pytorch-packages-for-jetpack-6-1/309084

### Step 2: Install torch2trt

Once PyTorch with CUDA is installed:

```bash
# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6  # or /usr/local/cuda if that's your version

# Clone and install torch2trt
cd /tmp
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install --user --plugins
```

### Step 3: Install trt_pose

```bash
cd /tmp
git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
cd trt_pose
python3 setup.py install --user
```

### Step 4: Download the Model

```bash
wget https://github.com/NVIDIA-AI-IOT/trt_pose/raw/master/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth -O ~/resnet18_baseline_att_224x224_A_epoch_249.pth
```

### Step 5: Verify Installation

```bash
python3 -c "
import torch
import trt_pose
import torch2trt
print('✅ PyTorch CUDA:', torch.cuda.is_available())
print('✅ trt_pose installed')
print('✅ torch2trt installed')
"
```

## Troubleshooting

- **If CUDA_HOME error**: Make sure CUDA_HOME is set before running setup.py
- **If PyTorch CUDA not available**: You must install Jetson-compatible PyTorch (not the CPU version from pip)
- **If import errors**: Make sure all packages are installed to the same Python environment

