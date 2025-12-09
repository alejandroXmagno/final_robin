import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys

model_path = "pose_landmarker_lite.task"

# --- GPU options ---
# Try GPU first, fall back to CPU if GPU is not available
use_gpu = True
base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.GPU   # THIS ENABLES GPU / TensorRT
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # good for streaming video
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create the PoseLandmarker with GPU, fall back to CPU if it fails
try:
    detector = vision.PoseLandmarker.create_from_options(options)
    print("✅ BlazePose initialized with GPU acceleration")
except (NotImplementedError, RuntimeError) as e:
    if "GPU" in str(e) or "gpu" in str(e).lower():
        print("⚠️  GPU acceleration not available. Falling back to CPU...")
        print(f"   Error: {e}")
        print("   To enable GPU, run: ./build_mediapipe_gpu.sh")
        print("   See ENABLE_BLAZEPOSE_GPU.md for details")
        
        # Fall back to CPU
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        print("✅ BlazePose initialized with CPU (slower performance)")
        use_gpu = False
    else:
        print(f"❌ Failed to initialize BlazePose: {e}")
        sys.exit(1)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    # Draw keypoints
    if result.pose_landmarks:
        for lm in result.pose_landmarks[0]:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    window_title = "BlazePose GPU" if use_gpu else "BlazePose CPU"
    cv2.imshow(window_title, frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
