# Enabling GPU Support for BlazePose on Jetson

## Problem
The pip-installed MediaPipe package doesn't include GPU support. When you try to use GPU acceleration, you'll see errors like:
```
ImageCloneCalculator: GPU processing is disabled in build flags
NotImplementedError: ValidatedGraphConfig Initialization failed.
```

## Solution Options

### Option 1: Build MediaPipe from Source (Recommended for GPU)

This is the most reliable way to get GPU support on Jetson:

```bash
# Run the build script (takes 30-60 minutes)
./build_mediapipe_gpu.sh
```

**Requirements:**
- At least 8GB free disk space
- 30-60 minutes build time
- All dependencies will be installed automatically

### Option 2: Use CPU Fallback (Quick but Slower)

If you don't want to build from source, you can modify `test_blazepose.py` to use CPU:

```python
delegate=python.BaseOptions.Delegate.CPU  # Change GPU to CPU
```

### Option 3: Use Alternative GPU-Accelerated Pose Detection

This codebase already has `trt-pose` setup which uses TensorRT and works well on Jetson:

```bash
# See INSTALL_TRT_POSE.md for instructions
# Or use test_openpifpaf.py which uses PyTorch with CUDA
```

## Verifying GPU Support

After building MediaPipe with GPU support, test it:

```bash
source blazepose_venv/bin/activate
python test_blazepose.py
```

You should NOT see the "GPU processing is disabled" error.

## Troubleshooting

### Build Fails
- Ensure you have enough disk space (8GB+)
- Check that all dependencies installed correctly
- Try building with fewer parallel jobs: `bazel build -c opt --jobs=2 ...`

### GPU Still Not Working After Build
- Verify CUDA is available: `nvcc --version`
- Check OpenGL ES: `glxinfo | grep "OpenGL ES"`
- Try running with `sudo` (some GPU access requires elevated permissions)

### Performance Issues
- GPU acceleration on Jetson may still be slower than dedicated GPU solutions
- Consider using `trt-pose` or `openpifpaf` for better Jetson performance



