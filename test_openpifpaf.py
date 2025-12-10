#!/usr/bin/env python3
"""
Standalone OpenPifPaf Pose Detection Viewer
Runs OpenPifPaf at full speed and displays results in CV2 window
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import sys
import os
import threading
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import OpenPifPaf
try:
    import torch
    import openpifpaf
    OPENPIFPAF_AVAILABLE = True
except ImportError as e:
    print(f"Error: OpenPifPaf not available: {e}")
    print("Install with: pip3 install --user openpifpaf")
    sys.exit(1)


class OpenPifPafViewer(Node):
    """Standalone viewer for OpenPifPaf pose detection"""
    
    def __init__(self):
        super().__init__('openpifpaf_viewer')
        
        # Set high CPU priority for this process
        self._set_high_priority()
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.get_logger().error("CUDA not available! OpenPifPaf needs GPU.")
            sys.exit(1)
        
        self.get_logger().info("🚀 Initializing OpenPifPaf...")
        
        # Initialize OpenPifPaf predictor
        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
        
        # Verify model is on GPU
        model_device = next(self.predictor.model.parameters()).device
        self.get_logger().info(f"✅ OpenPifPaf initialized on {model_device}")
        self.get_logger().info("   Using shufflenetv2k16 model")
        
        # Set high priority for PyTorch threads
        self._set_torch_thread_priority()
        
        # Image bridge
        self.bridge = CvBridge()
    
    def _set_high_priority(self):
        """Set high CPU priority for this process"""
        try:
            # Set nice value to -10 (higher priority, requires root for negative values)
            # Try without root first (positive nice values)
            current_nice = os.nice(0)
            try:
                # Try to set to -10 (requires root)
                os.nice(-10 - current_nice)
                self.get_logger().info("✅ Set process nice value to -10 (high priority)")
            except PermissionError:
                # If we can't set negative, try to set to 0 (normal priority, but explicit)
                try:
                    os.nice(-current_nice)  # Set to 0
                    self.get_logger().info("✅ Set process nice value to 0 (normal priority)")
                except:
                    self.get_logger().warn("⚠️  Could not set process priority (requires root for high priority)")
        except Exception as e:
            self.get_logger().warn(f"⚠️  Could not set process priority: {e}")
    
    def _set_torch_thread_priority(self):
        """Set high priority for PyTorch threads"""
        try:
            # Set PyTorch to use fewer threads but higher priority
            torch.set_num_threads(2)  # Use 2 threads for inference
            
            # Set thread priority using threading module
            import threading
            current_thread = threading.current_thread()
            if hasattr(threading, 'set_native_priority'):
                threading.set_native_priority(current_thread, -10)
                self.get_logger().info("✅ Set PyTorch thread priority")
        except Exception as e:
            self.get_logger().debug(f"Could not set thread priority: {e}")
    
    def _set_cpu_affinity(self):
        """Set CPU affinity to high-performance cores"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            import psutil
            p = psutil.Process()
            
            # Get CPU count
            cpu_count = psutil.cpu_count()
            
            # On Jetson, typically cores 0-3 are A57 (big), 4-5 are Denver (little)
            # For Jetson Xavier/Orin: cores 0-5 are usually the big cores
            # Pin to last 2 cores (usually less used by system)
            if cpu_count >= 4:
                # Use the last 2 cores (typically less loaded)
                cores = [cpu_count - 2, cpu_count - 1]
                p.cpu_affinity(cores)
                self.get_logger().info(f"✅ Set CPU affinity to cores {cores}")
            else:
                # If fewer cores, use all
                p.cpu_affinity(list(range(cpu_count)))
                self.get_logger().info(f"✅ Set CPU affinity to all {cpu_count} cores")
        except Exception as e:
            self.get_logger().debug(f"Could not set CPU affinity: {e}")
        
        # Frame tracking for FPS
        self.frame_timestamps = []
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # Current image
        self.current_image = None
        self.current_predictions = None
        self.frames_received = 0
        
        # Check if camera topic exists
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=2)
            available_topics = result.stdout
            if '/camera/realsense/color/image_raw' in available_topics:
                self.get_logger().info("✅ Camera topic found: /camera/realsense/color/image_raw")
            else:
                self.get_logger().warn("⚠️  Camera topic not found!")
                self.get_logger().warn("   Available camera topics:")
                for line in available_topics.split('\n'):
                    if 'camera' in line.lower() or 'image' in line.lower():
                        self.get_logger().warn(f"      {line}")
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
        self.get_logger().info("⌨️  Press 'q' to quit")
        
        # Timer to warn if no frames received
        self.no_frames_timer = self.create_timer(5.0, self.check_for_frames)
        
        # Create a placeholder image to show immediately
        self.placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.placeholder_image, "Waiting for camera frames...", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(self.placeholder_image, "Topic: /camera/realsense/color/image_raw", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        self.current_image = self.placeholder_image
        
        # Log display status
        display_env = os.environ.get('DISPLAY', 'NOT SET')
        self.get_logger().info(f"DISPLAY environment: {display_env}")
        
        # Test if we can open a window immediately
        try:
            cv2.namedWindow('OpenPifPaf Pose Detection (GPU)', cv2.WINDOW_NORMAL)
            cv2.imshow('OpenPifPaf Pose Detection (GPU)', self.placeholder_image)
            cv2.waitKey(1)
            self.get_logger().info("✅ Display window opened successfully!")
        except Exception as e:
            self.get_logger().error(f"❌ Cannot open display window: {e}")
            self.get_logger().error("   Check DISPLAY environment variable and X11 forwarding")
        
        # Timer for display update (runs as fast as possible)
        self.display_timer = self.create_timer(0.01, self.update_display)  # 100Hz display update
        
        # Log display status
        self.get_logger().info(f"DISPLAY environment: {os.environ.get('DISPLAY', 'NOT SET')}")
        
    def image_callback(self, msg):
        """Process each camera frame with OpenPifPaf"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_image = cv_image
            self.frames_received += 1
            
            if self.frames_received == 1:
                self.get_logger().info(f"✅ First frame received! Image size: {cv_image.shape}")
            
            # Convert BGR to RGB for OpenPifPaf
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Run OpenPifPaf inference (GPU-accelerated)
            predictions, _, meta = self.predictor.numpy_image(rgb_image)
            self.current_predictions = predictions
            
            # Update FPS
            current_time = time.time()
            self.frame_timestamps.append(current_time)
            # Keep only last 30 timestamps
            if len(self.frame_timestamps) > 30:
                self.frame_timestamps.pop(0)
            
            # Calculate FPS
            if len(self.frame_timestamps) >= 2:
                time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
                if time_span > 0:
                    self.fps = (len(self.frame_timestamps) - 1) / time_span
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def update_display(self):
        """Update the display window with pose detection results"""
        if self.current_image is None:
            return
        
        try:
            # Create display image
            display_image = self.current_image.copy()
            h, w = display_image.shape[:2]
            
            # Show waiting message if no frames received yet
            if self.frames_received == 0:
                cv2.putText(display_image, f"Frames received: {self.frames_received}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_image, "Waiting for camera...", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw pose predictions
            if self.current_predictions is not None and len(self.current_predictions) > 0:
                # OpenPifPaf COCO keypoint connections
                # Format: (start_idx, end_idx)
                connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                    (5, 6),  # Shoulders
                    (5, 7), (7, 9),  # Left arm
                    (6, 8), (8, 10),  # Right arm
                    (5, 11), (6, 12),  # Torso
                    (11, 12),  # Hips
                    (11, 13), (13, 15),  # Left leg
                    (12, 14), (14, 16),  # Right leg
                ]
                
                # Draw each detected person
                for person in self.current_predictions:
                    keypoints = person.data  # Shape: (17, 3) where 3 = (x, y, confidence)
                    
                    # Draw keypoints
                    for kp in keypoints:
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                        if conf > 0.3:  # Only draw if confidence > 0.3
                            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw connections
                    for start_idx, end_idx in connections:
                        if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                            kp1 = keypoints[start_idx]
                            kp2 = keypoints[end_idx]
                            
                            if kp1[2] > 0.3 and kp2[2] > 0.3:  # Both keypoints visible
                                x1, y1 = int(kp1[0]), int(kp1[1])
                                x2, y2 = int(kp2[0]), int(kp2[1])
                                cv2.line(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw person count
                person_text = f"People: {len(self.current_predictions)}"
                cv2.putText(display_image, person_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw FPS (top right)
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(display_image, fps_text, (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw GPU status
            gpu_text = f"GPU: {next(self.predictor.model.parameters()).device}"
            cv2.putText(display_image, gpu_text, (w - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw model info
            model_text = "Model: shufflenetv2k16"
            cv2.putText(display_image, model_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the image
            try:
                cv2.imshow('OpenPifPaf Pose Detection (GPU)', display_image)
                # Check for 'q' key to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info('Quitting...')
                    cv2.destroyAllWindows()
                    raise SystemExit
            except cv2.error as e:
                error_str = str(e).lower()
                if 'cannot connect to x server' in error_str or 'display' in error_str or 'badwindow' in error_str:
                    if not hasattr(self, '_display_error_logged'):
                        self.get_logger().error('❌ Cannot open display window!')
                        self.get_logger().error(f'   DISPLAY={os.environ.get("DISPLAY", "NOT SET")}')
                        self.get_logger().error('   Make sure you have X11 forwarding enabled if using SSH')
                        self.get_logger().error('   Or run this on the Jetson directly (not via SSH)')
                        self._display_error_logged = True
                    self.destroy_timer(self.display_timer)
                else:
                    self.get_logger().error(f'OpenCV error: {e}')
                    
        except Exception as e:
            self.get_logger().error(f'Error updating display: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def check_for_frames(self):
        """Warn if no frames have been received"""
        if self.frames_received == 0:
            self.get_logger().warn("⚠️  No camera frames received yet!")
            self.get_logger().warn("   Make sure camera is running:")
            self.get_logger().warn("   ros2 launch roverrobotics_driver realsense_simple.launch.py")
            # Check if topic is publishing
            import subprocess
            try:
                result = subprocess.run(['ros2', 'topic', 'hz', '/camera/realsense/color/image_raw'], 
                                      capture_output=True, text=True, timeout=1)
                if 'no new messages' in result.stderr.lower() or result.returncode != 0:
                    self.get_logger().warn("   Topic exists but not publishing data")
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    
    viewer = OpenPifPafViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            viewer.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

