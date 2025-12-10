import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys

model_path = "gesture_recognizer.task"

# --- CPU mode (GPU not needed for gesture recognition) ---
base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.CPU  # CPU is fine
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # good for streaming video
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_hands=2  # Detect up to 2 hands
)

# Create the GestureRecognizer
try:
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print("✅ Gesture Recognizer initialized (CPU mode)")
except Exception as e:
    print(f"❌ Failed to initialize Gesture Recognizer: {e}")
    sys.exit(1)

# Open camera - try different indices and backends
print("\n📷 Opening camera...")
cap = None

# Try different camera backends
backends = [
    (cv2.CAP_V4L2, "V4L2"),
    (cv2.CAP_ANY, "ANY"),
]

for backend_id, backend_name in backends:
    for camera_idx in range(5):  # Try cameras 0-4
        try:
            test_cap = cv2.VideoCapture(camera_idx, backend_id)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    cap = test_cap
                    print(f"✅ Camera {camera_idx} opened successfully (backend: {backend_name})")
                    break
                else:
                    test_cap.release()
            else:
                if test_cap:
                    test_cap.release()
        except Exception as e:
            if test_cap:
                test_cap.release()
            continue
    
    if cap is not None:
        break

if cap is None:
    print("❌ No camera found. Please check your camera connection.")
    print("\n   Troubleshooting:")
    
    # Check if camera is in use
    import subprocess
    try:
        result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True)
        if result.stdout and 'COMMAND' in result.stdout:
            print("\n   ⚠️  Camera /dev/video0 is currently in use:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        print(f"      Process: {parts[0]} (PID: {parts[1]})")
            print("\n   Options:")
            print("   1. Stop the process using the camera")
            print("   2. Use a different camera (the script tried indices 0-4)")
            print("   3. Try: sudo killall realsense (if it's a realsense process)")
    except:
        pass
    
    print("\n   Other checks:")
    print("   1. Check if camera is connected: ls -la /dev/video*")
    print("   2. Try running with sudo (if permissions issue)")
    print("   3. Check if another application is using the camera")
    
    # Show available video devices
    try:
        result = subprocess.run(['ls', '-la', '/dev/video*'], capture_output=True, text=True, shell=True)
        if result.stdout:
            print("\n   Available video devices:")
            print(result.stdout)
    except:
        pass
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_idx = 0

print("\n📋 Recognized gestures will be displayed on screen")
print("   Press ESC to exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Recognize gestures
    result = recognizer.recognize_for_video(mp_image, frame_idx)
    frame_idx += 1

    # Draw hand landmarks and gestures
    if result.gestures:
        for hand_idx, gesture_list in enumerate(result.gestures):
            if gesture_list and len(gesture_list) > 0:
                # Get the top gesture
                top_gesture = gesture_list[0]
                gesture_name = top_gesture.category_name
                gesture_score = top_gesture.score
                
                # Draw hand landmarks if available
                if hand_idx < len(result.hand_landmarks) and result.hand_landmarks[hand_idx]:
                    landmarks = result.hand_landmarks[hand_idx]
                    for landmark in landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Display gesture name and confidence
                if hand_idx < len(result.handedness):
                    handedness = result.handedness[hand_idx]
                    if handedness and len(handedness) > 0:
                        hand_label = handedness[0].category_name  # "Left" or "Right"
                        display_text = f"{hand_label}: {gesture_name} ({gesture_score:.2f})"
                    else:
                        display_text = f"Hand {hand_idx+1}: {gesture_name} ({gesture_score:.2f})"
                else:
                    display_text = f"Hand {hand_idx+1}: {gesture_name} ({gesture_score:.2f})"
                
                # Draw text on frame
                y_pos = 30 + (hand_idx * 30)
                cv2.putText(frame, display_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Gesture recognition stopped")

