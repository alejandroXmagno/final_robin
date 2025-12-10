#!/usr/bin/env python3
"""
RealSense D435i RGB Viewer with BlazePose Detection
Detects human poses, waving gestures, and calculates person location
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

# Try to import MediaPipe for BlazePose
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Install with: pip3 install mediapipe")


class RealSenseViewer(Node):
    def __init__(self):
        super().__init__('realsense_viewer')
        
        # Check for display
        self.has_display = 'DISPLAY' in os.environ
        if not self.has_display:
            self.get_logger().warn('DISPLAY environment variable not set. Cannot show OpenCV windows.')
            self.get_logger().warn('Images will still be received and processed, but not displayed.')
        
        # Initialize MediaPipe Pose
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose_module = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            # Use model_complexity=0 (Lite) for much faster inference on ARM/Jetson
            self.pose = self.mp_pose_module.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0=Lite (fastest), 1=Full, 2=Heavy
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.get_logger().info('MediaPipe BlazePose initialized (Lite model for speed)')
        else:
            self.pose = None
            self.mp_pose_module = None
            self.get_logger().error('MediaPipe not available! Install with: pip3 install mediapipe')
        
        self.bridge = CvBridge()
        self.rgb_image = None
        self.raw_depth_image = None  # Store raw depth for location calculation
        self.rgb_received = False
        self.depth_received = False
        self.display_enabled = self.has_display
        
        # Pose detection state
        self.last_detection_time = 0
        self.detection_interval = 1.0 / 5.0  # 5 Hz
        self.current_pose_results = None
        self.is_waving = False
        self.person_location = None  # (x, y, z) in camera frame
        
        # Subscribers for RGB and depth images with BEST_EFFORT QoS for lower latency
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only keep latest frame, drop old ones
        )
        
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/realsense/color/image_raw',
            self.rgb_callback,
            camera_qos
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/realsense/depth/image_rect_raw',
            self.depth_callback,
            camera_qos
        )
        
        # Frame processing state
        self.pose_processing = False  # Prevent re-entry
        
        # Timer to update display (only if we have a display)
        if self.display_enabled:
            self.display_timer = self.create_timer(0.033, self.update_display)  # ~30 FPS
        
        # Always log status periodically
        self.status_timer = self.create_timer(5.0, self.log_status)
        
        self.get_logger().info('RealSense Viewer with BlazePose started')
        self.get_logger().info('Subscribed to: /camera/realsense/color/image_raw and /camera/realsense/depth/image_rect_raw')
        self.get_logger().info('Pose detection running at 5 Hz')
        if self.display_enabled:
            self.get_logger().info('Press \'q\' in the window to quit')
        else:
            self.get_logger().info('Press Ctrl+C to quit')
    
    def log_status(self):
        """Log status periodically"""
        rgb_status = "received" if self.rgb_image is not None else "waiting"
        depth_status = "received" if self.raw_depth_image is not None else "waiting"
        pose_status = "detected" if self.current_pose_results else "none"
        self.get_logger().info(f'Status: RGB={rgb_status}, Depth={depth_status}, Pose={pose_status}, Waving={self.is_waving}')
    
    def rgb_callback(self, msg):
        """Callback for RGB image - run pose detection at 5 Hz"""
        current_time = time.time()
        
        # Skip if we're still processing the previous frame
        if self.pose_processing:
            return
        
        # Check if enough time has passed since last detection (5 Hz = 200ms)
        time_since_last = current_time - self.last_detection_time
        if time_since_last < self.detection_interval:
            # Store frame for display but don't process
            try:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                if not self.rgb_received:
                    self.get_logger().info('RGB image received!')
                    self.rgb_received = True
            except Exception:
                pass
            return
        
        try:
            self.pose_processing = True
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self.rgb_received:
                self.get_logger().info('RGB image received!')
                self.rgb_received = True
            self.detect_pose()
            self.last_detection_time = current_time
        except Exception as e:
            self.get_logger().error(f'Error in RGB callback: {e}')
        finally:
            self.pose_processing = False
    
    def depth_callback(self, msg):
        """Callback for depth image - store raw depth for location calculation"""
        try:
            # Convert depth image to OpenCV format
            try:
                depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                # Store as float32 in mm
                self.raw_depth_image = depth_cv.astype(np.float32)
            except:
                depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if depth_cv.dtype != np.float32:
                    self.raw_depth_image = depth_cv.astype(np.float32)
                else:
                    self.raw_depth_image = depth_cv
            
            if not self.depth_received:
                self.get_logger().info('Depth image received!')
                self.depth_received = True
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def detect_pose(self):
        """Run BlazePose detection on current RGB image"""
        if not MEDIAPIPE_AVAILABLE or self.pose is None or self.rgb_image is None:
            return
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            self.current_pose_results = self.pose.process(rgb_image)
            
            # Check for waving and calculate location
            if self.current_pose_results.pose_landmarks:
                self.check_waving()
                self.calculate_person_location()
            else:
                self.is_waving = False
                self.person_location = None
        except Exception as e:
            self.get_logger().error(f'Error in pose detection: {e}')
    
    def check_waving(self):
        """Check if person is waving (hand above shoulder)"""
        if not self.current_pose_results or not self.current_pose_results.pose_landmarks:
            self.is_waving = False
            return
        
        landmarks = self.current_pose_results.pose_landmarks.landmark
        mp_pose = self.mp_pose_module.PoseLandmark
        
        # Get key points
        try:
            # Left side
            left_shoulder = landmarks[mp_pose.LEFT_SHOULDER]
            left_wrist = landmarks[mp_pose.LEFT_WRIST]
            
            # Right side
            right_shoulder = landmarks[mp_pose.RIGHT_SHOULDER]
            right_wrist = landmarks[mp_pose.RIGHT_WRIST]
            
            # Check if either hand is above its corresponding shoulder
            # In image coordinates, y decreases upward, so we check if wrist.y < shoulder.y
            left_waving = (left_wrist.visibility > 0.5 and 
                          left_shoulder.visibility > 0.5 and
                          left_wrist.y < left_shoulder.y)
            
            right_waving = (right_wrist.visibility > 0.5 and 
                           right_shoulder.visibility > 0.5 and
                           right_wrist.y < right_shoulder.y)
            
            self.is_waving = left_waving or right_waving
        except (AttributeError, IndexError) as e:
            self.is_waving = False
    
    def calculate_person_location(self):
        """Calculate person location relative to camera using depth data"""
        if (not self.current_pose_results or 
            not self.current_pose_results.pose_landmarks or
            self.raw_depth_image is None or
            self.rgb_image is None):
            self.person_location = None
            return
        
        try:
            landmarks = self.current_pose_results.pose_landmarks.landmark
            mp_pose = self.mp_pose_module.PoseLandmark
            
            # Use torso center (average of shoulders and hips) for location
            # Get image dimensions
            h, w = self.rgb_image.shape[:2]
            
            # Get torso keypoints
            left_shoulder = landmarks[mp_pose.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.LEFT_HIP]
            right_hip = landmarks[mp_pose.RIGHT_HIP]
            
            # Calculate center of torso in image coordinates
            center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0
            center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4.0
            
            # Convert to pixel coordinates
            pixel_x = int(center_x * w)
            pixel_y = int(center_y * h)
            
            # Get depth at this pixel (depth is in mm)
            if (0 <= pixel_x < self.raw_depth_image.shape[1] and 
                0 <= pixel_y < self.raw_depth_image.shape[0]):
                depth_mm = self.raw_depth_image[pixel_y, pixel_x]
                
                # Skip invalid depth readings
                if depth_mm > 0 and depth_mm < 10000:  # Valid range: 0-10m
                    # Convert to meters
                    depth_m = depth_mm / 1000.0
                    
                    # Calculate 3D position in camera frame
                    # RealSense D435i: FOV ~69° horizontal, ~42° vertical at 640x480
                    fx = w / (2.0 * np.tan(np.radians(69.0 / 2.0)))  # Approximate focal length
                    fy = h / (2.0 * np.tan(np.radians(42.0 / 2.0)))
                    cx = w / 2.0
                    cy = h / 2.0
                    
                    # Convert pixel to normalized coordinates
                    x_norm = (pixel_x - cx) / fx
                    y_norm = (pixel_y - cy) / fy
                    
                    # Calculate 3D position (camera frame: x=right, y=down, z=forward)
                    x = depth_m * x_norm
                    y = depth_m * y_norm
                    z = depth_m
                    
                    self.person_location = (x, y, z)
                else:
                    self.person_location = None
            else:
                self.person_location = None
        except Exception as e:
            self.get_logger().error(f'Error calculating person location: {e}')
            self.person_location = None
    
    def update_display(self):
        """Update the display window with RGB, pose overlay, and info"""
        if not self.display_enabled:
            return
        
        if self.rgb_image is None:
            return
        
        try:
            # Create a copy for display
            display_image = self.rgb_image.copy()
            
            # Draw pose landmarks if available
            if (MEDIAPIPE_AVAILABLE and 
                self.current_pose_results and 
                self.current_pose_results.pose_landmarks):
                
                # Draw pose landmarks and connections
                self.mp_drawing.draw_landmarks(
                    display_image,
                    self.current_pose_results.pose_landmarks,
                    self.mp_pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    )
                )
            
            # Add text overlay with pose and location info
            y_offset = 30
            line_height = 25
            
            # Pose status
            if self.current_pose_results and self.current_pose_results.pose_landmarks:
                cv2.putText(display_image, "Pose: DETECTED", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Pose: NONE", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            y_offset += line_height
            
            # Waving status
            if self.is_waving:
                cv2.putText(display_image, "Waving: YES", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_image, "Waving: NO", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            y_offset += line_height
            
            # Location
            if self.person_location:
                x, y, z = self.person_location
                location_text = f"Location: X={x:.2f}m Y={y:.2f}m Z={z:.2f}m"
                cv2.putText(display_image, location_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_image, "Location: N/A", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Show the image
            cv2.imshow('RealSense D435i - BlazePose Detection', display_image)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quitting...')
                cv2.destroyAllWindows()
                raise SystemExit
        except cv2.error as e:
            # OpenCV display error
            error_str = str(e).lower()
            if 'cannot connect to x server' in error_str or 'display' in error_str or 'badwindow' in error_str:
                self.get_logger().error('Cannot open display window. Make sure you have X11 forwarding enabled.')
                self.display_enabled = False
                self.display_timer.cancel()
            else:
                self.get_logger().error(f'OpenCV error: {e}')


def main():
    rclpy.init()
    
    viewer = RealSenseViewer()
    
    try:
        rclpy.spin(viewer)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        viewer.get_logger().error(f'Error in viewer: {e}')
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ignore shutdown errors


if __name__ == '__main__':
    main()
