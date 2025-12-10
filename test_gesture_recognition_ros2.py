#!/usr/bin/env python3
"""
Gesture Recognition using MediaPipe with ROS2 RealSense Camera
Subscribes to ROS2 camera topic instead of accessing camera directly
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import numpy as np
import time

model_path = "gesture_recognizer.task"


class GestureRecognizerNode(Node):
    """ROS2 node for gesture recognition using MediaPipe"""
    
    def __init__(self):
        super().__init__('gesture_recognizer_node')
        
        # Initialize MediaPipe Gesture Recognizer
        self.get_logger().info("🔧 Initializing MediaPipe Gesture Recognizer...")
        
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_hands=2
        )
        
        try:
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            self.get_logger().info("✅ Gesture Recognizer initialized (CPU mode)")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize Gesture Recognizer: {e}")
            sys.exit(1)
        
        # Image bridge for ROS2
        self.bridge = CvBridge()
        
        # Frame counter for video mode
        self.frame_idx = 0
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.last_fps_update = time.time()
        
        # Check if camera topic exists
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=2)
            available_topics = result.stdout
            if '/camera/realsense/color/image_raw' in available_topics:
                self.get_logger().info("✅ Camera topic found: /camera/realsense/color/image_raw")
            else:
                self.get_logger().warn("⚠️  Camera topic not found!")
                self.get_logger().warn("   Start camera with: ros2 launch roverrobotics_driver realsense_simple.launch.py")
        except Exception as e:
            self.get_logger().warn(f"Could not check topics: {e}")
        
        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/realsense/color/image_raw',
            self.image_callback,
            1  # Queue size 1 for latest frame only
        )
        
        self.get_logger().info("📹 Subscribed to /camera/realsense/color/image_raw")
        self.get_logger().info("🖥️  Opening display window...")
        self.get_logger().info("⌨️  Press 'q' or ESC to quit")
        
        # Create a placeholder image
        self.placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.placeholder_image, "Waiting for camera frames...", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(self.placeholder_image, "Topic: /camera/realsense/color/image_raw", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        self.current_image = self.placeholder_image
        
        # Track frames received
        self.frames_received = 0
        
        # Thread lock for display updates (if needed)
        import threading
        self.display_lock = threading.Lock()
        
    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            # Convert ROS2 image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
            
            # Recognize gestures
            result = self.recognizer.recognize_for_video(mp_image, self.frame_idx)
            self.frame_idx += 1
            
            # Draw results on image (copy needed since we modify it)
            display_image = cv_image.copy()
            
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
                                x = int(landmark.x * display_image.shape[1])
                                y = int(landmark.y * display_image.shape[0])
                                cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)
                        
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
                        cv2.putText(display_image, display_text, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            self.fps_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_fps_update
            
            # Update FPS every second
            if elapsed >= 1.0:
                self.current_fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.last_fps_update = current_time
            
            # Draw FPS on image
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display_image, fps_text, (display_image.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert RGB to BGR for OpenCV display (only when needed)
            display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            
            # Update display immediately (no FPS limit)
            with self.display_lock:
                self.current_image = display_image_bgr
            
            # Update display window directly in callback for maximum FPS
            cv2.imshow("Gesture Recognition (ROS2)", display_image_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.get_logger().info("🛑 Shutting down...")
                cv2.destroyAllWindows()
                rclpy.shutdown()
            
            self.frames_received += 1
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    


def main(args=None):
    rclpy.init(args=args)
    
    node = GestureRecognizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("✅ Gesture recognition stopped")
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

