#!/bin/bash
# Setup script for BlazePose virtual environment

set -e

VENV_DIR="blazepose_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Setting up BlazePose virtual environment..."
echo ""

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
        exit 0
    fi
fi

echo "📦 Creating virtual environment..."
python3 -m venv "$VENV_DIR"

echo "✅ Virtual environment created at $VENV_DIR"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install ROS2 dependencies (cv_bridge, rclpy, etc.)
echo "📦 Installing ROS2 Python dependencies..."
pip install opencv-python numpy

# Note: ROS2 packages (rclpy, cv_bridge) need to be installed from system
# They're typically installed via apt, not pip
echo "⚠️  Note: ROS2 packages (rclpy, cv_bridge) should be installed system-wide"
echo "   If you get import errors, install them with:"
echo "   sudo apt-get install ros-humble-rclpy ros-humble-cv-bridge"

# Install MediaPipe (latest version with Tasks API support)
echo "📦 Installing MediaPipe..."
pip install --upgrade mediapipe

# Verify Tasks API is available
echo "📦 Verifying MediaPipe Tasks API..."
python3 -c "
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker
    from mediapipe.tasks.python.base_options import BaseOptions
    print('✅ MediaPipe Tasks API is available')
except ImportError as e:
    print(f'❌ MediaPipe Tasks API not available: {e}')
    print('   Try: pip install --upgrade mediapipe')
    exit(1)
"

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To use the virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 test_blazepose.py"
echo ""
echo "To deactivate:"
echo "  deactivate"

