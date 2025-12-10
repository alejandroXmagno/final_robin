#!/bin/bash
# Fix BlazePose GPU by upgrading MediaPipe to version with Tasks API

set -e

echo "🔧 Fixing BlazePose GPU acceleration..."
echo ""

# Check if venv exists
if [ ! -d "blazepose_venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./setup_blazepose_venv.sh first"
    exit 1
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source blazepose_venv/bin/activate

# Check current MediaPipe version
echo "📦 Current MediaPipe version:"
pip show mediapipe | grep Version || echo "   Not installed"

# Uninstall old version
echo ""
echo "🗑️  Uninstalling old MediaPipe..."
pip uninstall -y mediapipe 2>/dev/null || true

# Install latest MediaPipe (0.10.0+ has Tasks API)
echo ""
echo "📦 Installing latest MediaPipe (with Tasks API support)..."
pip install --upgrade mediapipe

# Verify Tasks API is available
echo ""
echo "✅ Verifying MediaPipe Tasks API..."
python3 -c "
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker
    from mediapipe.tasks.python.base_options import BaseOptions
    print('✅ MediaPipe Tasks API is now available!')
    print('   GPU acceleration should work now.')
except ImportError as e:
    print(f'❌ MediaPipe Tasks API still not available: {e}')
    print('   Try: pip install mediapipe==0.10.0')
    exit(1)
"

echo ""
echo "✅ Done! Now run: python test_blazepose.py"



