#!/usr/bin/env python3
"""
Frontier exploration script that finds unmapped areas and navigates to them.
Also detects waving people and navigates to them.
Visualizes the target point and path in RViz.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point, Pose, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image, PointCloud2, PointField, LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import time
import cv2
import struct
import os
import signal
import sys
import subprocess
import threading
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Use OpenPifPaf for GPU-accelerated pose detection on Jetson
# OpenPifPaf is a state-of-the-art pose estimation library that works well on Jetson
OPENPIFPAF_AVAILABLE = False
OPENPIFPAF_ERROR = None
try:
    import torch
    if not torch.cuda.is_available():
        OPENPIFPAF_AVAILABLE = False
        OPENPIFPAF_ERROR = "PyTorch CUDA not available"
    else:
        try:
            import openpifpaf
            OPENPIFPAF_AVAILABLE = True
        except ImportError as e:
            OPENPIFPAF_AVAILABLE = False
            OPENPIFPAF_ERROR = f"openpifpaf not installed: {e}. Install with: pip3 install --user openpifpaf"
except (ImportError, OSError) as e:
    OPENPIFPAF_AVAILABLE = False
    OPENPIFPAF_ERROR = f"PyTorch not available: {e}"

# Try to import TensorRT support
TENSORRT_AVAILABLE = False
TENSORRT_METHOD = None
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    TENSORRT_METHOD = "TensorRT"
except ImportError:
    pass

# Try TensorFlow with TensorRT (TF-TRT)
if not TENSORRT_AVAILABLE:
    try:
        import tensorflow as tf
        # Check if TensorRT is available in TensorFlow
        if hasattr(tf, 'python') and hasattr(tf.python, 'compiler'):
            TENSORRT_AVAILABLE = True
            TENSORRT_METHOD = "TensorFlow-TensorRT (TF-TRT)"
    except ImportError:
        pass

# Try ONNX Runtime with TensorRT backend
if not TENSORRT_AVAILABLE:
    try:
        import onnxruntime as ort
        # Check if TensorRT execution provider is available
        providers = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in providers:
            TENSORRT_AVAILABLE = True
            TENSORRT_METHOD = "ONNX Runtime with TensorRT"
    except ImportError:
        pass

# Function to check CUDA availability
def check_cuda_availability():
    """Check if CUDA is available and being used"""
    cuda_available = False
    cuda_method = "Unknown"
    cuda_details = []
    
    # Method 1: Check nvidia-smi directly (most reliable for NVIDIA GPUs)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if gpu_names:
                cuda_available = True
                cuda_method = f"NVIDIA GPU detected via nvidia-smi ({len(gpu_names)} GPU(s))"
                cuda_details.append(f"GPUs: {', '.join(gpu_names)}")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # Method 2: Check for CUDA libraries in system
    if not cuda_available:
        cuda_lib_paths = [
            '/usr/local/cuda/lib64/libcudart.so',
            '/usr/lib/x86_64-linux-gnu/libcudart.so',
            '/usr/local/cuda/lib/libcudart.so'
        ]
        for lib_path in cuda_lib_paths:
            if os.path.exists(lib_path):
                cuda_available = True
                cuda_method = f"CUDA library found at {lib_path}"
                break
    
    # Method 3: Check via PyTorch (most common ML framework)
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            cuda_method = f"PyTorch (CUDA {cuda_version}, {device_name})"
            cuda_details.append(f"PyTorch CUDA devices: {torch.cuda.device_count()}")
    except (ImportError, OSError) as e:
        # PyTorch not available or missing dependencies (e.g., OpenMPI)
        if not cuda_available:
            cuda_details.append(f"PyTorch check failed: {str(e)}")
    
    # Method 4: Check via TensorFlow (MediaPipe uses TensorFlow Lite)
    try:
        import tensorflow as tf
        if hasattr(tf.config, 'list_physical_devices'):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if not cuda_available:
                    cuda_available = True
                    cuda_method = f"TensorFlow (found {len(gpus)} GPU(s))"
                cuda_details.append(f"TensorFlow GPU devices: {len(gpus)}")
            else:
                if not cuda_available:
                    cuda_details.append("TensorFlow: no GPU devices found")
        else:
            # Older TensorFlow versions
            try:
                if tf.test.is_gpu_available():
                    if not cuda_available:
                        cuda_available = True
                        cuda_method = "TensorFlow (GPU available)"
            except Exception:
                if not cuda_available:
                    cuda_details.append("TensorFlow: GPU check unavailable")
    except ImportError:
        pass
    except Exception as e:
        if not cuda_available:
            cuda_details.append(f"TensorFlow check failed: {str(e)}")
    
    # Method 5: Check for Jetson (NVIDIA embedded platform)
    if not cuda_available:
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower() or 'nvidia' in model.lower():
                    cuda_available = True
                    cuda_method = f"Jetson platform detected ({model})"
        except (FileNotFoundError, PermissionError):
            pass
        except Exception:
            pass
    
    # Method 6: Check environment variables
    if not cuda_available:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            cuda_method = "CUDA environment variable set (CUDA_VISIBLE_DEVICES)"
            # If explicitly set, assume CUDA is intended to be used
            if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() != '':
                cuda_available = True
    
    # Compile final message
    if cuda_details:
        details_str = " | ".join(cuda_details)
        if cuda_available:
            cuda_method = f"{cuda_method} ({details_str})"
    
    if not cuda_available:
        cuda_method = "No CUDA detected - using CPU"
    
    return cuda_available, cuda_method


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Set high CPU priority for pose detection
        self._set_high_priority()
        
        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Publishers for visualization
        self.target_marker_pub = self.create_publisher(Marker, 'exploration_target', 10)
        self.path_pub = self.create_publisher(Path, 'exploration_path', 10)
        self.frontier_markers_pub = self.create_publisher(MarkerArray, 'frontier_markers', 10)
        self.person_location_marker_pub = self.create_publisher(MarkerArray, 'person_locations', 10)
        # Alternative: PointCloud2 for person locations (more efficient for many points)
        self.person_location_cloud_pub = self.create_publisher(PointCloud2, 'person_locations_cloud', 10)

        # Publisher for direct velocity commands (pivot turns, wall hugging)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Pivot turn state (360 degree turn every 30 seconds for better person detection)
        self.last_pivot_time = time.time()
        self.pivot_interval = 30.0  # 30 seconds between pivots
        self.pivot_in_progress = False
        self.pivot_start_time = 0.0
        self.pivot_duration = 8.0  # seconds to complete 360 degree turn
        self.pivot_angular_speed = (2 * math.pi) / self.pivot_duration  # rad/s for full rotation
        self.pivot_min_clearance = 0.6096  # 2 feet minimum clearance to pivot

        # Wall hugging state (after border is fully mapped)
        self.wall_hugging_mode = False
        self.wall_hug_complete = False
        self.wall_hug_start_pose = None
        self.wall_hug_distance_traveled = 0.0
        self.last_wall_hug_pose = None
        
        # Temporal exploration / patrol mode
        self.patrol_mode = False  # Activated after initial exploration
        self.visit_timestamps = {}  # Dict[(grid_x, grid_y)] = last_visit_time
        self.visit_radius = 1.0  # meters - mark area as visited within this radius
        self.revisit_threshold = 120.0  # seconds - revisit areas not seen in 2 minutes
        self.last_visit_update_time = 0
        self.visit_update_interval = 2.0  # Update visit timestamps every 2 seconds

        # Map saving
        self.map_save_path = '/home/stickykeys/final_robin/saved_maps'
        self.map_saved = False

        # Emergency obstacle avoidance state - INCREASED for safety!
        self.obstacle_too_close = False
        self.last_obstacle_time = 0
        self.obstacle_backup_duration = 4.0  # seconds to back up when obstacle detected (increased)
        self.obstacle_backup_start = 0
        self.is_backing_up = False
        # BALANCED SAFETY DISTANCES - conservative but allows navigation
        # These are EMERGENCY-ONLY distances, Nav2 costmap handles normal obstacle avoidance
        self.min_obstacle_distance = 0.5  # meters - STOP and cancel nav if < 50cm
        self.critical_obstacle_distance = 0.3  # meters - EMERGENCY BACKUP if < 30cm
        self.slowdown_obstacle_distance = 1.0  # meters - just for awareness logging
        self.last_scan = None  # Store last LiDAR scan for clearance checks

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Subscribe to planned path from nav2
        self.planned_path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # Subscribe to LiDAR scan for emergency obstacle avoidance
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Initialize OpenPifPaf for GPU-accelerated pose detection
        self.pose_predictor = None
        self.use_openpifpaf = False
        
        if OPENPIFPAF_AVAILABLE:
            try:
                import torch
                import openpifpaf
                
                cuda_available, cuda_method = check_cuda_availability()
                cuda_status = "✅ ENABLED" if cuda_available else "❌ DISABLED"
                self.get_logger().info(f'🔍 CUDA Status: {cuda_status} ({cuda_method})')
                
                if torch.cuda.is_available():
                    self.get_logger().info('🚀 Initializing OpenPifPaf for GPU-accelerated pose detection...')
                    
                    # Initialize OpenPifPaf predictor (automatically uses CUDA if available)
                    # Use a lightweight model for Jetson (shufflenetv2k16 is fast)
                    self.pose_predictor = openpifpaf.Predictor(
                        checkpoint='shufflenetv2k16'  # Lightweight model for Jetson
                    )
                    
                    # Verify model is on GPU
                    model_device = next(self.pose_predictor.model.parameters()).device
                    self.use_openpifpaf = True
                    self.get_logger().info(f'✅ OpenPifPaf initialized successfully on {model_device}!')
                    self.get_logger().info('   Using shufflenetv2k16 model (optimized for edge devices)')
                    
                    # Set high priority for PyTorch threads
                    self._set_torch_thread_priority()
                else:
                    self.get_logger().error('❌ CUDA not available! Person detection disabled.')
            except Exception as e:
                self.get_logger().error(f'❌ Failed to initialize OpenPifPaf: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                self.pose_predictor = None
                self.use_openpifpaf = False
        else:
            if OPENPIFPAF_ERROR:
                self.get_logger().error(f'❌ OpenPifPaf not available: {OPENPIFPAF_ERROR}')
            else:
                self.get_logger().error('❌ OpenPifPaf not available! Person detection disabled.')
        
        # Check for display
        self.has_display = 'DISPLAY' in os.environ
        if not self.has_display:
            self.get_logger().warn('DISPLAY environment variable not set. Cannot show OpenCV windows.')
            self.get_logger().warn('Images will still be received and processed, but not displayed.')
        self.display_enabled = self.has_display
        
        # Image processing
        self.bridge = CvBridge()
        self.rgb_image = None
        self.raw_depth_image = None
        self.last_pose_detection_time = 0
        self.pose_detection_interval = 1.0 / 5.0  # 5 Hz target
        self.current_pose_results = None
        self.frame_skip_counter = 0  # Skip frames when behind
        self.last_rgb_time = 0  # Track frame timing
        self.pose_processing = False  # Prevent re-entry
        
        # TF buffer for transforms - initialized once, NOT in callbacks
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cached_robot_pose = None  # Cache robot pose to avoid blocking lookups
        self.last_tf_lookup_time = 0
        self.tf_lookup_interval = 0.1  # Only lookup TF every 100ms max
        self.slam_working = False  # Track if SLAM is providing good transforms
        self.last_slam_check_time = 0
        
        # Emergency stop state
        self.estop_active = False
        self.estop_sub = self.create_subscription(
            Bool,
            '/estop_trigger',
            self.estop_callback,
            10
        )
        self.estop_reset_sub = self.create_subscription(
            Bool,
            '/estop_reset',
            self.estop_reset_callback,
            10
        )
        
        # Latency tracking for camera
        self.frame_timestamps = []  # Track last N frame times for FPS calculation
        self.last_frame_time = 0
        self.actual_fps = 0.0
        self.frame_latency_ms = 0.0
        self.is_waving = False
        self.person_location_camera_frame = None  # (x, y, z) in camera frame (current detection)
        self.person_location_map_frame = None  # (x, y) in map frame (for approach logic)
        self.detected_people = []  # List of all detected people: [(x, y, timestamp, is_waving), ...]
        self.person_marker_id_counter = 0  # Counter for unique marker IDs
        self.waving_text_visible = False  # Track if waving text marker is currently visible
        self.approaching_person = False  # Flag to track if we're approaching a person
        self.person_wait_start_time = None
        self.person_wait_duration = 10.0  # Wait 10 seconds near person
        self.person_location_timeout = 5.0  # Remove person markers after 5 seconds of no detection
        
        # Camera mounting offset (relative to base_link/lidar)
        # 13 inches = 0.3302 m in z (above), 3 inches = 0.0762 m in x (forward)
        self.camera_offset_x = 0.0762  # meters forward
        self.camera_offset_z = 0.3302  # meters above
        self.camera_tilt_angle = np.radians(20.0)  # Camera tilted up 20 degrees
        self.person_stop_distance = 0.3048  # 1 foot = 0.3048 meters
        
        # Subscribe to RealSense camera images with BEST_EFFORT QoS for lower latency
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
        
        # State
        self.current_map = None
        self.map_metadata = None
        self.navigation_goal_handle = None
        self.exploring = True
        self.last_goal_time = time.time()
        self.goal_timeout = 120.0  # seconds to wait before giving up on a goal
        self.current_target = None
        self.actual_goal_location = None  # Track the actual goal location sent to Nav2 (for completion check)
        self.current_path = None
        self.visited_frontiers = set()  # Track visited frontiers to avoid repeating
        self.unreachable_frontiers = set()  # Track frontiers that failed to reach
        self.map_frame_available = False  # Track if map frame exists
        self._last_enclosure_check_result = (False, 0.0, False)  # Store last enclosure check result
        
        # Stuck detection: track if robot stays in same 1-foot area for 15+ seconds
        self.position_history = []  # List of (x, y, timestamp) tuples
        self.stuck_threshold_distance = 0.3048  # 1 foot in meters
        self.stuck_threshold_time = 15.0  # 15 seconds
        self.is_stuck = False
        self.navigating_to_clear_area = False  # Flag to track if we're navigating to clear area
        
        # Exploration parameters - PROGRESSIVE NAVIGATION with shorter paths
        self.min_frontier_size = 8  # minimum number of frontier cells
        self.exploration_radius = 3.5  # max distance for primary search - SHORT paths! (was 10.0)
        self.cluster_threshold = 1.0  # distance threshold for clustering frontiers
        self.min_distance_to_goal = 0.3  # minimum distance to consider a new goal
        self.goal_reached_distance = 0.6096  # 2 feet - consider goal reached when within this distance
        self.current_frontier_key = None  # Track current frontier key for marking as visited
        
        # Wait for nav2 action server
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        if not self.nav_action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('NavigateToPose action server not available after 30s!')
            self.get_logger().error('Make sure nav2 is running.')
            return
        
        self.get_logger().info('NavigateToPose action server available!')
        
        # Start exploration timer (reduced from 5.0 to 3.0 for more frequent frontier checks)
        self.explore_timer = self.create_timer(3.0, self.explore_callback)
        
        # Start goal timeout checker
        self.timeout_timer = self.create_timer(1.0, self.check_goal_timeout)
        
        # Timer to publish visualization markers (1 Hz is enough for RViz)
        self.viz_timer = self.create_timer(1.0, self.publish_visualization)
        
        # Timer to republish path for visualization
        self.path_pub_timer = self.create_timer(0.5, self.publish_path)
        
        # Timer to check for waving person and handle person approach
        self.person_detection_timer = self.create_timer(0.25, self.check_person_detection)
        
        # Timer to check if robot is stuck in same area
        self.stuck_detection_timer = self.create_timer(1.0, self.check_stuck_condition)

        # Timer for periodic 360-degree pivot turns (better LiDAR coverage)
        # DISABLED: Commented out to prevent regular spinning
        # self.pivot_timer = self.create_timer(0.1, self.pivot_turn_callback)
        
        # Timer to check for continuous obstacle boundary (enclosure detection) every 5 seconds
        self.enclosure_check_timer = self.create_timer(5.0, self.check_enclosure_callback)

        # Timer for wall hugging behavior
        self.wall_hug_timer = self.create_timer(0.1, self.wall_hug_callback)

        # Timer for emergency obstacle avoidance (runs at 50Hz for very fast response)
        self.obstacle_avoidance_timer = self.create_timer(0.02, self.obstacle_avoidance_callback)

        # Timer to update display (only if we have a display) - 10 FPS is enough for visualization
        if self.display_enabled:
            self.display_timer = self.create_timer(0.1, self.update_display)  # 10 FPS (reduced from 30)
            self.get_logger().info('CV2 display window enabled at 10 FPS. Press \'q\' in the window to quit.')
        
        # Timer to check SLAM health
        self.slam_health_timer = self.create_timer(2.0, self.check_slam_health)
    
    def estop_callback(self, msg):
        """Callback for emergency stop trigger"""
        if msg.data:
            self.estop_active = True
            self.get_logger().warn('EMERGENCY STOP ACTIVATED!')
    
    def estop_reset_callback(self, msg):
        """Callback for emergency stop reset"""
        if msg.data:
            self.estop_active = False
            self.get_logger().info('Emergency stop reset')

    def scan_callback(self, msg):
        """
        Process LiDAR scan data to detect close obstacles.
        This runs at the scan rate (~10Hz typically) for immediate detection.
        EMERGENCY-ONLY SAFETY SYSTEM - only intervenes for imminent collisions!
        Normal obstacle avoidance is handled by Nav2 costmaps.
        """
        if msg is None or len(msg.ranges) == 0:
            return
        
        # Store scan for clearance checks (pivot, etc.)
        self.last_scan = msg

        # Check for obstacles within safety radius
        # Scan the front 180 degrees for emergency detection only
        num_ranges = len(msg.ranges)
        # Front 180 degrees: from -90 to +90 degrees
        arc_degrees = 180  # Reduced from 240 to avoid over-detection
        arc_fraction = arc_degrees / 360.0
        front_arc = int(num_ranges * arc_fraction)
        # Center the arc on the front of the robot
        start_idx = int(num_ranges * (1.0 - arc_fraction) / 2.0)
        end_idx = start_idx + front_arc

        min_distance = float('inf')
        critical_obstacle_detected = False
        warning_obstacle_detected = False
        
        # Track which angular sector has the closest obstacle for better logging
        min_distance_angle = 0
        valid_readings = 0

        for i in range(start_idx, min(end_idx, num_ranges)):
            distance = msg.ranges[i]
            # Filter out invalid readings (inf, nan, 0)
            if not math.isfinite(distance) or distance < msg.range_min or distance > msg.range_max:
                continue

            valid_readings += 1
            if distance < min_distance:
                min_distance = distance
                # Calculate angle in degrees (-90 to +90)
                min_distance_angle = ((i - start_idx) / front_arc) * 180 - 90

            # Critical zone: immediate backup required (< 30cm)
            if distance < self.critical_obstacle_distance:
                critical_obstacle_detected = True

            # Warning zone: cancel navigation (< 50cm)  
            if distance < self.min_obstacle_distance:
                warning_obstacle_detected = True

        # Diagnostic: Log if no valid readings (possible sensor issue)
        if valid_readings == 0:
            if not hasattr(self, '_last_no_scan_warning'):
                self._last_no_scan_warning = 0
            if time.time() - self._last_no_scan_warning > 5.0:
                self.get_logger().warn(f'⚠️  No valid LiDAR readings in scan! Check sensor.')
                self._last_no_scan_warning = time.time()

        # Update obstacle state - prioritize critical over warning
        if critical_obstacle_detected:
            if not self.obstacle_too_close:
                self.get_logger().error(
                    f'🚨 COLLISION DANGER! Obstacle at {min_distance:.2f}m (angle: {min_distance_angle:.0f}°)! '
                    f'EMERGENCY BACKUP INITIATED!'
                )
            self.obstacle_too_close = True
            self.last_obstacle_time = time.time()
            # Start backup sequence IMMEDIATELY
            if not self.is_backing_up:
                self.is_backing_up = True
                self.obstacle_backup_start = time.time()
                # Cancel any current navigation IMMEDIATELY
                self.cancel_current_goal()
                # Force immediate stop before backing up
                self.stop_robot()
        elif warning_obstacle_detected:
            if not self.obstacle_too_close:
                self.get_logger().warn(
                    f'⚠️  WARNING: Obstacle at {min_distance:.2f}m (angle: {min_distance_angle:.0f}°)! '
                    f'Canceling navigation - Nav2 costmap will handle avoidance'
                )
                # Cancel navigation goal but let Nav2 costmap handle the rest
                self.cancel_current_goal()
                # DON'T force stop - let Nav2 handle it through costmaps
            self.obstacle_too_close = True
            self.last_obstacle_time = time.time()
        else:
            # Clear after 1.0s without obstacle detection (increased from 0.5s for stability)
            if self.obstacle_too_close and (time.time() - self.last_obstacle_time) > 1.0:
                self.get_logger().info(f'✓ Path clear (min_dist={min_distance:.2f}m), resuming normal operation')
                self.obstacle_too_close = False
                self.is_backing_up = False

    def obstacle_avoidance_callback(self):
        """
        Emergency obstacle avoidance - runs at 50Hz (0.02s interval).
        Takes ABSOLUTE priority over all other behaviors to prevent collisions.
        This is the LAST LINE OF DEFENSE against hitting obstacles!
        """
        if self.estop_active:
            return

        current_time = time.time()

        # If backing up due to critical obstacle
        if self.is_backing_up:
            elapsed = current_time - self.obstacle_backup_start

            if elapsed < self.obstacle_backup_duration:
                # Continue backing up at FASTER speed for safety
                twist = Twist()
                twist.linear.x = -0.25  # Back up at 25 cm/s (increased from 15 cm/s)
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return
            else:
                # Finished backing up, turn slightly to avoid obstacle on retry
                self.is_backing_up = False
                self.stop_robot()
                self.get_logger().info('✓ Emergency backup complete - will choose different path')
                return

        # If obstacle detected but not backing up, don't interfere with Nav2
        # Let Nav2 handle obstacle avoidance through costmaps
        # Only the LiDAR safety system should stop the robot, not this timer
        if self.obstacle_too_close:
            # Just return, don't send stop commands - Nav2 is handling it
            return

    def rgb_callback(self, msg):
        """Callback for RGB image - run pose detection at 5 Hz"""
        current_time = time.time()
        
        # Track frame timing for FPS calculation
        self.frame_timestamps.append(current_time)
        # Keep only last 10 timestamps
        if len(self.frame_timestamps) > 10:
            self.frame_timestamps.pop(0)
        # Calculate actual FPS
        if len(self.frame_timestamps) >= 2:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                self.actual_fps = (len(self.frame_timestamps) - 1) / time_span
        
        # Calculate latency from message timestamp
        try:
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.frame_latency_ms = (current_time - msg_time) * 1000
        except:
            self.frame_latency_ms = 0
        
        # Skip if we're still processing the previous frame
        if self.pose_processing:
            return
        
        # NO FRAME RATE LIMITING - process every frame for maximum performance
        # (Removed 5Hz limit to allow full-speed GPU processing)
        
        try:
            self.pose_processing = True
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.detect_pose()
            self.last_pose_detection_time = current_time
            self.last_frame_time = current_time
        except Exception as e:
            self.get_logger().error(f'Error in RGB callback: {e}')
        finally:
            self.pose_processing = False
    
    def depth_callback(self, msg):
        """Callback for depth image - store raw depth for location calculation"""
        # Only process depth if we need it for person location
        if self.current_pose_results is None or not self.current_pose_results.pose_landmarks:
            return  # Skip depth processing if no person detected
        
        try:
            # Use passthrough for fastest conversion
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Only convert to float32 if needed
            if depth_cv.dtype == np.uint16:
                self.raw_depth_image = depth_cv.astype(np.float32)
            elif depth_cv.dtype == np.float32:
                self.raw_depth_image = depth_cv
            else:
                self.raw_depth_image = depth_cv.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def detect_pose(self):
        """Run pose detection on current RGB image using OpenPifPaf (GPU-accelerated)"""
        if self.rgb_image is None:
            return
        
        try:
            if self.use_openpifpaf and self.pose_predictor is not None:
                # Convert BGR to RGB for OpenPifPaf
                rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
                
                # Run OpenPifPaf inference on GPU
                predictions, _, meta = self.pose_predictor.numpy_image(rgb_image)
                
                # Create a simple pose results object compatible with existing code
                class Landmark:
                    def __init__(self, x, y, z, visibility):
                        self.x = x
                        self.y = y
                        self.z = z
                        self.visibility = visibility
                
                class Landmarks:
                    def __init__(self, landmarks_list):
                        self.landmark = landmarks_list
                
                class PoseResults:
                    def __init__(self, landmarks):
                        self.pose_landmarks = landmarks
                
                # OpenPifPaf returns predictions for each person
                # We'll use the first person detected
                if len(predictions) > 0:
                    # Get first person's keypoints
                    person = predictions[0]
                    keypoints = person.data  # Shape: (17, 3) where 3 = (x, y, confidence)
                    
                    h, w = rgb_image.shape[:2]
                    landmarks_list = []
                    
                    # COCO keypoint order: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
                    # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
                    # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
                    # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
                    
                    for kp in keypoints:
                        x_pixel = kp[0]
                        y_pixel = kp[1]
                        confidence = kp[2]
                        
                        # Normalize to [0, 1]
                        x_norm = x_pixel / w if w > 0 else 0.5
                        y_norm = y_pixel / h if h > 0 else 0.5
                        
                        landmarks_list.append(Landmark(x_norm, y_norm, 0.0, confidence))
                    
                    # Create MediaPipe-compatible format
                    self.current_pose_results = PoseResults(Landmarks(landmarks_list))
                else:
                    self.current_pose_results = None
            else:
                self.current_pose_results = None
                return
            
            # Check for person and calculate location (always, regardless of robot state)
            if self.current_pose_results and self.current_pose_results.pose_landmarks:
                self.check_waving()
                self.calculate_person_location_camera_frame()
                
                # Transform to map frame and add to detected people list
                if self.person_location_camera_frame is not None:
                    self.transform_person_to_map_frame()
                    if self.person_location_map_frame is not None:
                        person_x, person_y = self.person_location_map_frame
                        current_time = time.time()
                        # Add or update person location with waving state
                        self.add_person_location(person_x, person_y, current_time, self.is_waving)
                        self.get_logger().debug(f'Person location in map frame: ({person_x:.2f}, {person_y:.2f}), waving={self.is_waving}')
                    else:
                        self.get_logger().debug('Person location transform to map frame failed')
            else:
                self.is_waving = False
                self.person_location_camera_frame = None
                self.person_location_map_frame = None
        except Exception as e:
            self.get_logger().error(f'Error in pose detection: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def add_person_location(self, x, y, timestamp, is_waving=False):
        """Add or update a person location in the detected people list"""
        # Check if we already have a person at this location (within 0.5m)
        person_added = False
        for i, (px, py, pt, pw) in enumerate(self.detected_people):
            dist = math.sqrt((px - x)**2 + (py - y)**2)
            if dist < 0.5:  # Same person (within 0.5m)
                # Update existing person location, timestamp, and waving state
                # Keep waving=True if they were waving recently (sticky for better visibility)
                was_waving = pw and (timestamp - pt) < 2.0  # Keep waving state for 2 seconds
                self.detected_people[i] = (x, y, timestamp, is_waving or was_waving)
                person_added = True
                break
        
        if not person_added:
            # New person detected
            self.detected_people.append((x, y, timestamp, is_waving))
            self.get_logger().info(f'New person detected at ({x:.2f}, {y:.2f}), waving={is_waving}')
    
    def cleanup_old_person_locations(self):
        """Remove person locations that haven't been seen recently"""
        current_time = time.time()
        self.detected_people = [
            (x, y, t, w) for (x, y, t, w) in self.detected_people
            if (current_time - t) < self.person_location_timeout
        ]
    
    def check_waving(self):
        """Check if person is waving (hand above shoulder)"""
        if not self.current_pose_results or not self.current_pose_results.pose_landmarks:
            self.is_waving = False
            return
        
        landmarks = self.current_pose_results.pose_landmarks.landmark
        
        try:
            # COCO keypoint indices: 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
            # MediaPipe-compatible format: indices match COCO for these keypoints
            left_shoulder = landmarks[5]  # COCO left_shoulder
            left_wrist = landmarks[9]     # COCO left_wrist
            right_shoulder = landmarks[6]  # COCO right_shoulder
            right_wrist = landmarks[10]    # COCO right_wrist
            
            # Check if either hand is above its corresponding shoulder
            left_waving = (left_wrist.visibility > 0.5 and 
                          left_shoulder.visibility > 0.5 and
                          left_wrist.y < left_shoulder.y)
            
            right_waving = (right_wrist.visibility > 0.5 and 
                           right_shoulder.visibility > 0.5 and
                           right_wrist.y < right_shoulder.y)
            
            self.is_waving = left_waving or right_waving
        except (AttributeError, IndexError) as e:
            self.is_waving = False
    
    def calculate_person_location_camera_frame(self):
        """Calculate person location in camera frame using depth data"""
        if (not self.current_pose_results or 
            not self.current_pose_results.pose_landmarks or
            self.raw_depth_image is None or
            self.rgb_image is None):
            self.person_location_camera_frame = None
            return
        
        try:
            landmarks = self.current_pose_results.pose_landmarks.landmark
            
            # Use torso center (average of shoulders and hips) for location
            h, w = self.rgb_image.shape[:2]
            
            # Get torso keypoints (COCO format: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip)
            left_shoulder = landmarks[5]   # COCO left_shoulder
            right_shoulder = landmarks[6]   # COCO right_shoulder
            left_hip = landmarks[11]        # COCO left_hip
            right_hip = landmarks[12]       # COCO right_hip
            
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
                    
                    self.person_location_camera_frame = (x, y, z)
                else:
                    self.person_location_camera_frame = None
            else:
                self.person_location_camera_frame = None
        except Exception as e:
            self.get_logger().error(f'Error calculating person location: {e}')
            self.person_location_camera_frame = None
    
    def transform_person_to_map_frame(self):
        """Transform person location from camera frame to map frame (non-blocking)"""
        if self.person_location_camera_frame is None:
            self.person_location_map_frame = None
            return
        
        try:
            # Camera frame: x=right, y=down, z=forward
            # Base_link frame: x=forward, y=left, z=up
            cam_x, cam_y, cam_z = self.person_location_camera_frame
            
            # Step 1: Account for camera tilt (20 degrees up)
            cos_tilt = np.cos(self.camera_tilt_angle)
            sin_tilt = np.sin(self.camera_tilt_angle)
            
            # Rotate in camera optical frame (around x-axis to undo tilt)
            cam_y_corrected = cam_y * cos_tilt - cam_z * sin_tilt
            cam_z_corrected = cam_y * sin_tilt + cam_z * cos_tilt
            
            # Step 2: Convert camera optical frame to base_link frame
            base_x = cam_z_corrected + self.camera_offset_x  # forward (camera z + offset)
            base_y = -cam_x  # left (negative camera x)
            base_z = -cam_y_corrected + self.camera_offset_z  # up (negative camera y + offset)
            
            # Transform to map frame - use very short timeout to avoid blocking!
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01)  # 10ms timeout - non-blocking
                )
                
                # Get robot pose in map frame
                robot_x = transform.transform.translation.x
                robot_y = transform.transform.translation.y
                robot_z = transform.transform.translation.z
                
                # Get robot orientation (yaw)
                q = transform.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                
                # Transform point from base_link to map frame
                # Rotate by robot yaw and translate by robot position
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                
                map_x = robot_x + base_x * cos_yaw - base_y * sin_yaw
                map_y = robot_y + base_x * sin_yaw + base_y * cos_yaw
                
                self.person_location_map_frame = (map_x, map_y)
            except Exception as e:
                self.get_logger().warn(f'Could not transform person location to map frame: {e}')
                self.person_location_map_frame = None
        except Exception as e:
            self.get_logger().error(f'Error transforming person location: {e}')
            self.person_location_map_frame = None
    
    def check_person_detection(self):
        """Check for waving person and handle approach logic"""
        # If we're already approaching a person, check if we've arrived and are waiting
        if self.approaching_person:
            if self.person_wait_start_time is not None:
                # Check if wait time is over
                elapsed = time.time() - self.person_wait_start_time
                if elapsed >= self.person_wait_duration:
                    self.get_logger().info('Finished waiting near person, resuming exploration')
                    self.approaching_person = False
                    self.person_wait_start_time = None
                    self.person_location_map_frame = None
                    self.is_waving = False
                    # Clear current goal to allow new exploration
                    if self.navigation_goal_handle is not None:
                        self.cancel_current_goal()
            return
        
        # Check if we detect a waving person
        if self.is_waving and self.person_location_camera_frame is not None:
            # Transform to map frame
            self.transform_person_to_map_frame()
            
            if self.person_location_map_frame is not None:
                person_x, person_y = self.person_location_map_frame
                
                # Check if person is within reasonable range (not too far)
                robot_pose = self.get_robot_pose()
                if robot_pose:
                    rx, ry = robot_pose
                    dist = math.sqrt((rx - person_x)**2 + (ry - person_y)**2)
                    
                    # Only approach if person is within 10 meters and not already very close
                    if dist < 10.0 and dist > self.person_stop_distance:
                        self.get_logger().info(f'Detected waving person at ({person_x:.2f}, {person_y:.2f}), distance: {dist:.2f}m')
                        
                        # Cancel current exploration goal
                        if self.navigation_goal_handle is not None:
                            self.get_logger().info('Canceling exploration goal to approach person')
                            self.cancel_current_goal()
                        
                        # Calculate goal position (1 foot away from person)
                        # Goal should be on the line from robot to person, but 1 foot closer
                        goal_dist = dist - self.person_stop_distance
                        if goal_dist > 0.1:  # Only navigate if we need to move
                            goal_x = rx + (person_x - rx) * (goal_dist / dist)
                            goal_y = ry + (person_y - ry) * (goal_dist / dist)
                            
                            # Navigate to person
                            self.approaching_person = True
                            self.navigate_to_person(goal_x, goal_y, person_x, person_y)
                    elif dist <= self.person_stop_distance:
                        # Already close enough, start waiting
                        self.get_logger().info('Already close to person, starting wait period')
                        self.approaching_person = True
                        self.person_wait_start_time = time.time()
                        if self.navigation_goal_handle is not None:
                            self.cancel_current_goal()
    
    def navigate_to_person(self, goal_x, goal_y, person_x, person_y):
        """Navigate to a position near the person"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_x)
        goal_msg.pose.pose.position.y = float(goal_y)
        goal_msg.pose.pose.position.z = 0.0
        
        # Calculate orientation towards the person
        robot_pose = self.get_robot_pose()
        if robot_pose:
            rx, ry = robot_pose
            yaw = math.atan2(person_y - ry, person_x - rx)
            goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f'Navigating to person at ({goal_x:.2f}, {goal_y:.2f})')
        
        # Update target for visualization
        self.current_target = (goal_x, goal_y)
        
        # Send goal asynchronously
        send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=None
        )
        
        # Use a callback to handle the result
        def goal_response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn('Person approach goal was rejected!')
                    self.approaching_person = False
                    self.person_location_map_frame = None
                    return
                
                self.navigation_goal_handle = goal_handle
                self.last_goal_time = time.time()
                
                # Set up result callback
                result_future = goal_handle.get_result_async()
                
                def result_callback(result_future):
                    try:
                        result = result_future.result()
                        if result.status == 4:  # SUCCEEDED
                            self.get_logger().info('Reached person! Starting 10 second wait...')
                            self.person_wait_start_time = time.time()
                        else:
                            self.get_logger().info(f'Person approach finished with status: {result.status}')
                            self.approaching_person = False
                            self.person_location_map_frame = None
                    except Exception as e:
                        self.get_logger().warn(f'Error getting person approach result: {e}')
                        self.approaching_person = False
                        self.person_location_map_frame = None
                    finally:
                        self.navigation_goal_handle = None
                        self.current_target = None
                        self.actual_goal_location = None
                
                result_future.add_done_callback(result_callback)
            except Exception as e:
                self.get_logger().warn(f'Error in person approach goal response: {e}')
                self.approaching_person = False
                self.person_location_map_frame = None
        
        send_goal_future.add_done_callback(goal_response_callback)
        
    def map_callback(self, msg):
        """Store the latest map"""
        self.current_map = msg
        self.map_metadata = msg.info
        
    def path_callback(self, msg):
        """Store the planned path from nav2"""
        self.current_path = msg
        # Update header to ensure it's in map frame
        if self.current_path:
            self.current_path.header.frame_id = 'map'
    
    def publish_path(self):
        """Publish the current path for visualization"""
        if self.current_path and self.map_frame_available:
            self.path_pub.publish(self.current_path)
    
    def check_goal_timeout(self):
        """Check if current goal has timed out or is close enough to be considered reached"""
        if self.navigation_goal_handle is not None:
            # Check timeout
            elapsed = time.time() - self.last_goal_time
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f'Goal timed out after {elapsed:.1f}s, canceling...')
                self.cancel_current_goal()
                self.last_goal_time = time.time()
                self.current_target = None
                self.actual_goal_location = None
                self.current_frontier_key = None
                return
            
            # Check if we're close enough to consider the goal reached (2 feet)
            # Skip this check for person approach goals (those need to get closer)
            # Use actual_goal_location if available (the goal sent to Nav2), otherwise use current_target (frontier location)
            goal_location = self.actual_goal_location if self.actual_goal_location else self.current_target
            if goal_location and not self.approaching_person:
                robot_pose = self.get_robot_pose()
                if robot_pose:
                    rx, ry = robot_pose
                    gx, gy = goal_location
                    dist = math.sqrt((rx - gx)**2 + (ry - gy)**2)
                    
                    # Log distance periodically for debugging (every 5 seconds)
                    if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:
                        self.get_logger().debug(
                            f'Goal check: distance={dist:.2f}m, threshold={self.goal_reached_distance:.2f}m, '
                            f'goal_location={goal_location}, robot=({rx:.2f}, {ry:.2f})'
                        )
                    
                    if dist <= self.goal_reached_distance:
                        self.get_logger().info(
                            f'✓ Goal reached! Within {dist:.2f}m of goal '
                            f'(threshold: {self.goal_reached_distance:.2f}m / 2 ft)'
                        )
                        # Mark frontier as visited
                        if self.current_frontier_key:
                            self.visited_frontiers.add(self.current_frontier_key)
                            self.get_logger().info(f'Marked frontier {self.current_frontier_key} as visited')
                        # Cancel navigation and clear state
                        self.cancel_current_goal()
                        self.current_target = None
                        self.actual_goal_location = None
                        self.current_frontier_key = None
    
    def cancel_current_goal(self):
        """Cancel the current navigation goal"""
        if self.navigation_goal_handle is not None:
            try:
                future = self.nav_action_client._cancel_goal_async(self.navigation_goal_handle)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            except Exception as e:
                self.get_logger().warn(f'Error canceling goal: {e}')
            finally:
                self.navigation_goal_handle = None
    
    def find_frontiers(self):
        """Find frontier points in the current map (boundaries between known and unknown)"""
        if self.current_map is None or self.map_metadata is None:
            return []
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Find frontiers: both unknown cells adjacent to free cells AND free cells adjacent to unknown
        # This dual approach ensures we catch all frontiers and prioritize areas with more unknown space
        frontiers = []
        height, width = map_data.shape
        
        # Method 1: Find free cells (value = 0) adjacent to unknown cells (value = -1)
        # This is more efficient and gives us frontiers on the known side
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if this is a free cell
                if map_data[y, x] == 0:
                    # Check 8-connected neighbors for unknown space (more comprehensive)
                    neighbors = [
                        map_data[y-1, x-1], map_data[y-1, x], map_data[y-1, x+1],
                        map_data[y, x-1],                     map_data[y, x+1],
                        map_data[y+1, x-1], map_data[y+1, x], map_data[y+1, x+1]
                    ]
                    
                    # If any neighbor is unknown, this is a frontier
                    if -1 in neighbors:
                        # Convert to world coordinates
                        world_x = self.map_metadata.origin.position.x + (x + 0.5) * self.map_metadata.resolution
                        world_y = self.map_metadata.origin.position.y + (y + 0.5) * self.map_metadata.resolution
                        frontiers.append((world_x, world_y))
        
        # Method 2: Also find unknown cells adjacent to free cells (complementary approach)
        # This helps catch frontiers on the unknown side
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if this is an unknown cell
                if map_data[y, x] == -1:
                    # Check 4-connected neighbors for free space
                    neighbors = [
                        map_data[y-1, x], map_data[y+1, x],
                        map_data[y, x-1], map_data[y, x+1]
                    ]
                    
                    # If any neighbor is free space, this is a frontier
                    if 0 in neighbors:
                        # Convert to world coordinates
                        world_x = self.map_metadata.origin.position.x + (x + 0.5) * self.map_metadata.resolution
                        world_y = self.map_metadata.origin.position.y + (y + 0.5) * self.map_metadata.resolution
                        # Only add if not already in list (avoid duplicates)
                        if (world_x, world_y) not in frontiers:
                            frontiers.append((world_x, world_y))
        
        return frontiers

    def check_enclosed_and_explored(self):
        """
        Check if robot is in an enclosed area and if 80% of reachable space is explored.

        Returns:
            tuple: (is_enclosed, exploration_percentage, should_stop_exploring)
            - is_enclosed: True if robot is surrounded by connected obstacles
            - exploration_percentage: Percentage of reachable area that is explored (0-100)
            - should_stop_exploring: True if enclosed AND >= 80% explored
        """
        if self.current_map is None or self.map_metadata is None:
            return False, 0.0, False

        # Get robot position in grid coordinates
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False, 0.0, False

        robot_x, robot_y = robot_pose
        resolution = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y

        # Convert robot position to grid coordinates
        robot_grid_x = int((robot_x - origin_x) / resolution)
        robot_grid_y = int((robot_y - origin_y) / resolution)

        # Get map as numpy array
        height = self.current_map.info.height
        width = self.current_map.info.width
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape((height, width))

        # Validate robot position is within map
        if not (0 <= robot_grid_x < width and 0 <= robot_grid_y < height):
            return False, 0.0, False

        # Create obstacle mask (occupied cells have value > 50)
        obstacle_mask = (map_data > 50).astype(np.uint8)

        # Dilate obstacles to bridge gaps up to 5 grid spaces
        # Using a 5x5 kernel will close gaps of up to ~5 cells
        gap_bridge_size = 5
        kernel = np.ones((gap_bridge_size, gap_bridge_size), np.uint8)
        dilated_obstacles = cv2.dilate(obstacle_mask, kernel, iterations=1)

        # Create a mask for flood fill (need to add 1-pixel border for cv2.floodFill)
        flood_mask = np.zeros((height + 2, width + 2), np.uint8)

        # Copy dilated obstacles to flood mask (offset by 1)
        flood_mask[1:-1, 1:-1] = dilated_obstacles * 255

        # Flood fill from robot position to find reachable area
        # cv2.floodFill modifies the mask in place
        flood_fill_image = np.zeros((height, width), np.uint8)

        # Only flood fill if robot is in a free area (not on obstacle)
        if dilated_obstacles[robot_grid_y, robot_grid_x] == 0:
            cv2.floodFill(flood_fill_image, flood_mask, (robot_grid_x, robot_grid_y), 255)
        else:
            # Robot is on/near an obstacle after dilation, can't determine enclosure
            return False, 0.0, False

        # The flood-filled area is now marked with 255 in flood_fill_image
        reachable_mask = (flood_fill_image == 255)

        # Check if enclosed: flood fill should NOT reach map edges
        # Check all four edges
        top_edge = np.any(reachable_mask[0, :])
        bottom_edge = np.any(reachable_mask[-1, :])
        left_edge = np.any(reachable_mask[:, 0])
        right_edge = np.any(reachable_mask[:, -1])

        is_enclosed = not (top_edge or bottom_edge or left_edge or right_edge)

        if not is_enclosed:
            # Not enclosed, return early
            return False, 0.0, False

        # Calculate exploration percentage within reachable area
        # Free cells (explored): map_data == 0
        # Unknown cells (unexplored): map_data == -1
        # We only care about cells in the reachable area

        reachable_cells = reachable_mask.sum()
        if reachable_cells == 0:
            return is_enclosed, 0.0, False

        # Count explored (free) cells in reachable area
        explored_in_reachable = np.sum((map_data == 0) & reachable_mask)

        # Count unknown cells in reachable area
        unknown_in_reachable = np.sum((map_data == -1) & reachable_mask)

        # Total explorable area (free + unknown, excluding obstacles)
        total_explorable = explored_in_reachable + unknown_in_reachable

        if total_explorable == 0:
            exploration_percentage = 100.0
        else:
            exploration_percentage = (explored_in_reachable / total_explorable) * 100.0

        should_stop = is_enclosed and exploration_percentage >= 80.0

        # Note: The big obvious print is now done in check_enclosure_callback()
        # This function just returns the result

        return is_enclosed, exploration_percentage, should_stop
    
    def check_enclosure_callback(self):
        """Callback to check for continuous obstacle boundary every 5 seconds"""
        if not self.exploring:
            return
        
        if self.current_map is None:
            return
        
        # Check if map frame is available
        if not self.map_frame_available:
            return
        
        self.get_logger().info('🔍 Checking for continuous obstacle boundary around robot...')
        
        is_enclosed, exploration_pct, should_stop = self.check_enclosed_and_explored()
        
        # Store result for use in explore_callback
        self._last_enclosure_check_result = (is_enclosed, exploration_pct, should_stop)
        
        if is_enclosed:
            # BIG OBVIOUS PRINT when continuous boundary is found
            self.get_logger().info('')
            self.get_logger().info('=' * 80)
            self.get_logger().info('🚨🚨🚨 CONTINUOUS OBSTACLE BOUNDARY DETECTED! 🚨🚨🚨')
            self.get_logger().info('=' * 80)
            self.get_logger().info(f'   Robot is ENCLOSED by continuous obstacle boundary!')
            self.get_logger().info(f'   Exploration within enclosed area: {exploration_pct:.1f}%')
            self.get_logger().info(f'   Should stop exploring: {should_stop}')
            self.get_logger().info('=' * 80)
            self.get_logger().info('')
        else:
            self.get_logger().debug('   No continuous boundary detected - robot has exit paths')
    
    def update_visit_timestamps(self):
        """Update timestamps for areas the robot has recently visited"""
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_visit_update_time < self.visit_update_interval:
            return
        
        self.last_visit_update_time = current_time
        
        if self.current_map is None or self.map_metadata is None:
            return
        
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        
        rx, ry = robot_pose
        resolution = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y
        
        # Mark all cells within visit_radius as visited
        visit_cells = int(self.visit_radius / resolution)
        
        # Convert robot position to grid
        robot_grid_x = int((rx - origin_x) / resolution)
        robot_grid_y = int((ry - origin_y) / resolution)
        
        # Update timestamps for cells in visit radius
        for dy in range(-visit_cells, visit_cells + 1):
            for dx in range(-visit_cells, visit_cells + 1):
                # Check if within circular radius
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= visit_cells:
                    grid_x = robot_grid_x + dx
                    grid_y = robot_grid_y + dy
                    
                    # Round to 1 decimal for dictionary key
                    # Convert back to world coordinates for key
                    world_x = grid_x * resolution + origin_x
                    world_y = grid_y * resolution + origin_y
                    key = (round(world_x, 1), round(world_y, 1))
                    
                    self.visit_timestamps[key] = current_time
    
    def calculate_temporal_revisit_factor(self, x, y):
        """
        Calculate temporal revisit factor for a location.
        Returns higher values for areas not visited recently.
        
        Args:
            x, y: World coordinates
            
        Returns:
            float: 0.0 to 1.0, where 1.0 means area should be revisited (not seen recently)
        """
        if not self.patrol_mode:
            # In exploration mode, don't use temporal factor
            return 0.0
        
        current_time = time.time()
        
        # Check if this area or nearby areas have been visited
        # Look in a small radius around the point
        check_radius = 2.0  # meters
        resolution = self.map_metadata.resolution if self.map_metadata else 0.05
        check_cells = int(check_radius / resolution)
        
        min_time_since_visit = float('inf')
        
        # Sample a grid around the point
        for dy in range(-2, 3):  # Check 5x5 grid of cells
            for dx in range(-2, 3):
                check_x = x + dx * check_radius / 2.0
                check_y = y + dy * check_radius / 2.0
                key = (round(check_x, 1), round(check_y, 1))
                
                if key in self.visit_timestamps:
                    time_since_visit = current_time - self.visit_timestamps[key]
                    min_time_since_visit = min(min_time_since_visit, time_since_visit)
        
        # If never visited, return maximum priority
        if min_time_since_visit == float('inf'):
            return 1.0
        
        # If visited very recently, return 0
        if min_time_since_visit < 30.0:  # Less than 30 seconds ago
            return 0.0
        
        # If visited beyond revisit threshold, return high priority
        if min_time_since_visit >= self.revisit_threshold:
            return 1.0
        
        # Linear interpolation between 30s and revisit_threshold
        # 30s -> 0.0, revisit_threshold -> 1.0
        factor = (min_time_since_visit - 30.0) / (self.revisit_threshold - 30.0)
        return min(max(factor, 0.0), 1.0)
    
    def check_pivot_clearance(self):
        """
        Check if there's sufficient clearance to perform a 360° pivot.
        Requires at least 2 feet (0.6096m) clearance in all directions.
        
        Returns:
            bool: True if clearance is sufficient, False otherwise
        """
        if not hasattr(self, 'last_scan') or self.last_scan is None:
            return False
        
        scan = self.last_scan
        
        # Check all angles for minimum clearance
        min_distance = float('inf')
        
        for i, distance in enumerate(scan.ranges):
            # Skip invalid readings
            if distance < scan.range_min or distance > scan.range_max or not np.isfinite(distance):
                continue
            
            min_distance = min(min_distance, distance)
        
        # Check if minimum distance is at least 2 feet (0.6096m)
        has_clearance = min_distance >= self.pivot_min_clearance
        
        if has_clearance:
            self.get_logger().debug(
                f'Pivot clearance OK: {min_distance:.2f}m (need {self.pivot_min_clearance:.2f}m)'
            )
        else:
            self.get_logger().debug(
                f'Pivot clearance insufficient: {min_distance:.2f}m (need {self.pivot_min_clearance:.2f}m)'
            )
        
        return has_clearance

    def cluster_frontiers(self, frontiers):
        """Cluster nearby frontier points"""
        if not frontiers:
            return []
        
        clusters = []
        used = set()
        
        for i, (x1, y1) in enumerate(frontiers):
            if i in used:
                continue
            
            cluster = [(x1, y1)]
            used.add(i)
            
            # Find nearby frontiers
            for j, (x2, y2) in enumerate(frontiers):
                if j in used:
                    continue
                
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < self.cluster_threshold:
                    cluster.append((x2, y2))
                    used.add(j)
            
            if len(cluster) >= self.min_frontier_size:
                # Use centroid of cluster
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                clusters.append((cx, cy, len(cluster)))
        
        # Sort by cluster size (larger = better)
        clusters.sort(key=lambda x: x[2], reverse=True)
        return clusters
    
    def check_map_frame_available(self):
        """Check if map frame is available in tf tree (non-blocking)"""
        try:
            # Try to lookup transform - if it works, map frame exists
            self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01)  # 10ms - non-blocking
            )
            if not self.map_frame_available:
                self.get_logger().info('Map frame is now available!')
            self.map_frame_available = True
            return True
        except Exception:
            self.map_frame_available = False
            return False
    
    def get_robot_pose(self):
        """Get current robot pose from tf (with caching to avoid blocking)"""
        if not self.map_frame_available:
            return self.cached_robot_pose  # Return cached if available
        
        # Rate limit TF lookups to avoid excessive calls
        current_time = time.time()
        if (current_time - self.last_tf_lookup_time) < self.tf_lookup_interval:
            return self.cached_robot_pose  # Return cached pose
            
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01)  # 10ms - non-blocking
            )
            
            self.cached_robot_pose = (
                transform.transform.translation.x,
                transform.transform.translation.y
            )
            self.last_tf_lookup_time = current_time
            self.slam_working = True
            return self.cached_robot_pose
        except Exception as e:
            # Return cached pose on failure, don't block
            self.slam_working = False
            if self.cached_robot_pose is None and self.map_frame_available:
                self.get_logger().debug(f'Could not get robot pose: {e}')
            return self.cached_robot_pose
    
    def is_reachable(self, x, y):
        """Check if a point is reachable (in free space)"""
        if self.current_map is None or self.map_metadata is None:
            return False
        
        # Convert world coordinates to map indices
        map_x = int((x - self.map_metadata.origin.position.x) / self.map_metadata.resolution)
        map_y = int((y - self.map_metadata.origin.position.y) / self.map_metadata.resolution)
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Check bounds
        if map_x < 0 or map_x >= self.current_map.info.width:
            return False
        if map_y < 0 or map_y >= self.current_map.info.height:
            return False
        
        # Check if the cell is free (value = 0)
        cell_value = map_data[map_y, map_x]
        return cell_value == 0
    
    def find_nearby_reachable_point(self, x, y, max_search_radius=2.0):
        """Find a nearby reachable point in known free space"""
        if self.current_map is None or self.map_metadata is None:
            return None
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Convert to map coordinates
        map_x = int((x - self.map_metadata.origin.position.x) / self.map_metadata.resolution)
        map_y = int((y - self.map_metadata.origin.position.y) / self.map_metadata.resolution)
        
        # Search in expanding circles for a reachable point
        search_radius_cells = int(max_search_radius / self.map_metadata.resolution)
        
        for radius in range(1, search_radius_cells + 1):
            # Check points in a circle around the target
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Only check points on the circle perimeter (more efficient)
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    check_x = map_x + dx
                    check_y = map_y + dy
                    
                    # Check bounds
                    if (check_x < 0 or check_x >= self.current_map.info.width or
                        check_y < 0 or check_y >= self.current_map.info.height):
                        continue
                    
                    # Check if this cell is free
                    if map_data[check_y, check_x] == 0:
                        # Convert back to world coordinates
                        world_x = self.map_metadata.origin.position.x + (check_x + 0.5) * self.map_metadata.resolution
                        world_y = self.map_metadata.origin.position.y + (check_y + 0.5) * self.map_metadata.resolution
                        return (world_x, world_y)
        
        return None
    
    def find_obstacle_free_area(self, center_x, center_y, search_radius=5.0):
        """Find an obstacle-free area near the given center point (at least 4x4 feet)"""
        if self.current_map is None or self.map_metadata is None:
            return None
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Convert center to map coordinates
        center_map_x = int((center_x - self.map_metadata.origin.position.x) / self.map_metadata.resolution)
        center_map_y = int((center_y - self.map_metadata.origin.position.y) / self.map_metadata.resolution)
        
        # Search in expanding circles for a large obstacle-free area
        search_radius_cells = int(search_radius / self.map_metadata.resolution)
        
        # Look for areas with at least 4x4 feet of free space
        # 4 feet = 1.2192 meters, convert to cells
        min_free_size_meters = 1.2192  # 4 feet in meters
        min_free_size_cells = int(min_free_size_meters / self.map_metadata.resolution) + 1  # Add 1 for safety
        # Make sure it's at least 3 cells even if resolution is very fine
        min_free_size_cells = max(min_free_size_cells, 3)
        
        # Score potential areas by size (prefer larger areas)
        best_area = None
        best_score = 0
        
        for radius in range(1, search_radius_cells + 1):
            # Check points in a circle around the center
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Only check points on the circle perimeter (more efficient)
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    check_x = center_map_x + dx
                    check_y = center_map_y + dy
                    
                    # Check bounds (need margin for checking the area)
                    if (check_x < min_free_size_cells or check_x >= self.current_map.info.width - min_free_size_cells or
                        check_y < min_free_size_cells or check_y >= self.current_map.info.height - min_free_size_cells):
                        continue
                    
                    # Check if there's a min_free_size_cells x min_free_size_cells area of free space here
                    all_free = True
                    free_count = 0
                    for local_dy in range(-min_free_size_cells//2, min_free_size_cells//2 + 1):
                        for local_dx in range(-min_free_size_cells//2, min_free_size_cells//2 + 1):
                            cell_x = check_x + local_dx
                            cell_y = check_y + local_dy
                            if map_data[cell_y, cell_x] == 0:  # Free space
                                free_count += 1
                            else:
                                all_free = False
                                break
                        if not all_free:
                            break
                    
                    if all_free:
                        # Found a good obstacle-free area
                        world_x = self.map_metadata.origin.position.x + (check_x + 0.5) * self.map_metadata.resolution
                        world_y = self.map_metadata.origin.position.y + (check_y + 0.5) * self.map_metadata.resolution
                        
                        # Verify the point is reachable
                        if not self.is_reachable(world_x, world_y):
                            # Try to find a nearby reachable point
                            nearby = self.find_nearby_reachable_point(world_x, world_y, max_search_radius=1.0)
                            if nearby:
                                world_x, world_y = nearby
                            else:
                                continue  # Skip this area if not reachable
                        
                        # Score by area size (prefer larger areas) and distance (prefer closer)
                        area_size = free_count
                        dist_from_center = math.sqrt((world_x - center_x)**2 + (world_y - center_y)**2)
                        score = area_size / (dist_from_center + 0.1)  # Higher is better
                        
                        if score > best_score:
                            best_score = score
                            best_area = (world_x, world_y)
        
        return best_area
    
    def check_stuck_condition(self):
        """Check if robot has been stuck in the same 1-foot area for 15+ seconds"""
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        
        current_time = time.time()
        rx, ry = robot_pose
        
        # Add current position to history
        self.position_history.append((rx, ry, current_time))
        
        # Keep only recent positions (last 20 seconds)
        cutoff_time = current_time - 20.0
        self.position_history = [(x, y, t) for (x, y, t) in self.position_history if t > cutoff_time]
        
        if len(self.position_history) < 2:
            self.is_stuck = False
            return
        
        # Check if all recent positions are within 1 foot of each other
        # Get positions from the last 15 seconds
        recent_positions = [(x, y, t) for (x, y, t) in self.position_history 
                           if (current_time - t) <= self.stuck_threshold_time]
        
        if len(recent_positions) < 2:
            self.is_stuck = False
            return
        
        # Check if all positions are within stuck_threshold_distance of the first position
        first_x, first_y, first_t = recent_positions[0]
        all_within_threshold = True
        
        for x, y, t in recent_positions[1:]:
            dist = math.sqrt((x - first_x)**2 + (y - first_y)**2)
            if dist > self.stuck_threshold_distance:
                all_within_threshold = False
                break
        
        if all_within_threshold and (current_time - first_t) >= self.stuck_threshold_time:
            if not self.is_stuck:
                self.get_logger().warn(f'Robot detected as stuck in area around ({first_x:.2f}, {first_y:.2f}) for {current_time - first_t:.1f} seconds')
                self.is_stuck = True
                # Try to navigate to an obstacle-free area
                self.navigate_to_obstacle_free_area()
        else:
            self.is_stuck = False
    
    def navigate_to_obstacle_free_area(self):
        """Navigate to an obstacle-free area when stuck"""
        if self.navigating_to_clear_area:
            return  # Already navigating to clear area
        
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        
        rx, ry = robot_pose
        
        # Find an obstacle-free area nearby
        clear_area = self.find_obstacle_free_area(rx, ry, search_radius=5.0)
        
        if clear_area is None:
            self.get_logger().warn('Could not find obstacle-free area nearby')
            return
        
        clear_x, clear_y = clear_area
        
        # Check if we're already close to this area
        dist = math.sqrt((rx - clear_x)**2 + (ry - clear_y)**2)
        if dist < 1.0:  # Already in a clear area
            self.get_logger().info('Already in obstacle-free area')
            self.is_stuck = False
            self.position_history.clear()  # Reset position history
            return
        
        self.get_logger().info(f'Navigating to obstacle-free area at ({clear_x:.2f}, {clear_y:.2f})')
        self.navigating_to_clear_area = True
        
        # Cancel current goal if any
        if self.navigation_goal_handle is not None:
            self.cancel_current_goal()
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(clear_x)
        goal_msg.pose.pose.position.y = float(clear_y)
        goal_msg.pose.pose.position.z = 0.0
        
        # Calculate orientation towards the clear area
        yaw = math.atan2(clear_y - ry, clear_x - rx)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Update target for visualization
        self.current_target = (clear_x, clear_y)
        
        # Send goal asynchronously
        send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=None
        )
        
        def goal_response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn('Obstacle-free area goal was rejected!')
                    self.navigating_to_clear_area = False
                    self.current_target = None
                    return
                
                self.navigation_goal_handle = goal_handle
                self.last_goal_time = time.time()
                
                result_future = goal_handle.get_result_async()
                
                def result_callback(result_future):
                    try:
                        result = result_future.result()
                        if result.status == 4:  # SUCCEEDED
                            self.get_logger().info('Reached obstacle-free area! Resuming exploration')
                            self.is_stuck = False
                            self.position_history.clear()  # Reset position history
                        else:
                            self.get_logger().info(f'Obstacle-free area navigation finished with status: {result.status}')
                    except Exception as e:
                        self.get_logger().warn(f'Error getting obstacle-free area navigation result: {e}')
                    finally:
                        self.navigation_goal_handle = None
                        self.current_target = None
                        self.navigating_to_clear_area = False
                
                result_future.add_done_callback(result_callback)
            except Exception as e:
                self.get_logger().warn(f'Error in obstacle-free area goal response: {e}')
                self.navigating_to_clear_area = False
                self.current_target = None
        
        send_goal_future.add_done_callback(goal_response_callback)
    
    def select_best_frontier(self, clusters):
        """Select the best frontier to explore, prioritizing those near unmapped areas"""
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            # If we can't get pose, just pick the largest cluster that's reachable and not visited
            for cx, cy, size in clusters:
                # Round to avoid floating point precision issues
                key = (round(cx, 1), round(cy, 1))
                if self.is_reachable(cx, cy) and key not in self.visited_frontiers:
                    return (cx, cy, size, key)
            return None
        
        rx, ry = robot_pose
        
        # Score frontiers by size, distance, and proximity to unknown space
        scored = []
        for cx, cy, size in clusters:
            # Check if reachable
            if not self.is_reachable(cx, cy):
                continue
            
            # Round to avoid floating point precision issues when checking visited
            key = (round(cx, 1), round(cy, 1))
            
            # Skip if we've already visited this frontier
            if key in self.visited_frontiers:
                continue
            
            # Skip if we've already determined this frontier is unreachable
            if key in self.unreachable_frontiers:
                continue
            
            dist = math.sqrt((rx - cx)**2 + (ry - cy)**2)
            
            # Skip if too close (already explored)
            if dist < self.min_distance_to_goal:
                continue
            
            # Prefer larger clusters that are not too far
            if dist > self.exploration_radius:
                continue
            
            # Calculate proximity to unknown space (higher is better)
            unknown_proximity = self.calculate_unknown_proximity(cx, cy)

            # Calculate size of unknown region near this frontier (prioritize larger unexplored areas)
            unknown_region_size = self.calculate_unknown_region_size(cx, cy)
            # Normalize region size (assume max reasonable size is 100 m², scale to 0-1)
            normalized_region_size = min(unknown_region_size / 100.0, 1.0)

            # Calculate path clearance (how open is the route to this frontier)
            # Returns (avg_clearance, min_clearance) - we care most about min_clearance
            avg_clearance, min_clearance = self.calculate_path_clearance_detailed(rx, ry, cx, cy)

            # REJECT only truly impassable paths (very low threshold)
            # Allow narrow passages but heavily penalize them in scoring below
            MIN_PATH_CLEARANCE = 0.08  # Only block paths with < 8% clearance (truly impassable)
            if min_clearance < MIN_PATH_CLEARANCE:
                self.get_logger().debug(
                    f'Rejecting frontier at ({cx:.1f}, {cy:.1f}) - path impassable '
                    f'(min_clearance={min_clearance:.2f} < {MIN_PATH_CLEARANCE})'
                )
                continue

            # Calculate temporal revisit factor (prioritize areas not visited recently)
            temporal_factor = self.calculate_temporal_revisit_factor(cx, cy)
            
            # Score: PRIORITIZE LARGE OPEN SPACES for person detection!
            # Significantly boost the value of open areas and large unknown regions
            # People are more likely to be in open spaces, not narrow corridors
            openness_score = (normalized_region_size ** 0.7) * 10.0  # Heavily favor large open areas
            # Heavily penalize narrow paths - use exponential scaling to strongly discourage narrow passages
            # Clearance 0.2 (narrow) gets ~2.24, clearance 0.5 (medium) gets ~3.54, clearance 0.8 (wide) gets ~4.47
            clearance_score = (min_clearance ** 0.4) * 20.0  # Strongly reward wide paths, heavily penalize narrow ones
            
            base_score = (
                size * 1.0 +  # Frontier size (baseline)
                unknown_proximity * 2.0 +  # Near unexplored areas
                openness_score +  # MAJOR: Large open spaces where people are likely!
                clearance_score +  # Wide open paths
                temporal_factor * 3.0  # Revisit areas not seen recently
            )
            
            # For patrol mode, reduce distance penalty to encourage wider exploration
            if self.patrol_mode:
                distance_penalty = (dist ** 1.0) + 0.1  # Linear penalty for patrol
            else:
                # STRONG distance penalty for progressive navigation
                distance_penalty = (dist ** 1.5) + 0.1
            
            score = base_score / distance_penalty
            scored.append((score, cx, cy, size, key))
        
        if not scored:
            # If no suitable frontier found, try expanding radius more aggressively
            for cx, cy, size in clusters:
                if not self.is_reachable(cx, cy):
                    continue
                key = (round(cx, 1), round(cy, 1))
                if key in self.visited_frontiers:
                    continue
                if key in self.unreachable_frontiers:
                    continue
                dist = math.sqrt((rx - cx)**2 + (ry - cy)**2)
                if dist < self.min_distance_to_goal:
                    continue
                # Expand to 3x the radius to find more distant unknown areas
                if dist <= self.exploration_radius * 3.0:
                    unknown_proximity = self.calculate_unknown_proximity(cx, cy)
                    unknown_region_size = self.calculate_unknown_region_size(cx, cy)
                    normalized_region_size = min(unknown_region_size / 100.0, 1.0)
                    avg_clearance, min_clearance = self.calculate_path_clearance_detailed(rx, ry, cx, cy)
                    # Still reject only truly impassable paths in fallback mode
                    MIN_PATH_CLEARANCE = 0.08  # Only block paths with < 8% clearance (truly impassable)
                    if min_clearance < MIN_PATH_CLEARANCE:
                        continue
                    # Calculate temporal revisit factor (prioritize areas not visited recently)
                    temporal_factor = self.calculate_temporal_revisit_factor(cx, cy)
                    
                    # Same scoring as primary mode: prioritize large open spaces
                    openness_score = (normalized_region_size ** 0.7) * 10.0
                    # Heavily penalize narrow paths - use exponential scaling to strongly discourage narrow passages
                    clearance_score = (min_clearance ** 0.4) * 20.0  # Strongly reward wide paths, heavily penalize narrow ones
                    
                    base_score = (
                        size * 1.0 +
                        unknown_proximity * 2.0 +
                        openness_score +
                        clearance_score +
                        temporal_factor * 3.0
                    )
                    
                    # For patrol mode, reduce distance penalty
                    if self.patrol_mode:
                        distance_penalty = (dist ** 1.0) + 0.1
                    else:
                        distance_penalty = (dist ** 1.5) + 0.1
                    
                    score = base_score / distance_penalty
                    scored.append((score, cx, cy, size, key))
        
        if not scored:
            return None
        
        # Return highest scoring frontier
        scored.sort(key=lambda x: x[0], reverse=True)
        _, cx, cy, size, key = scored[0]
        return (cx, cy, size, key)
    
    def calculate_unknown_proximity(self, x, y):
        """Calculate how close a point is to unknown space (0-1 scale, weighted by distance)"""
        if self.current_map is None or self.map_metadata is None:
            return 0.0
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Convert to map coordinates
        map_x = int((x - self.map_metadata.origin.position.x) / self.map_metadata.resolution)
        map_y = int((y - self.map_metadata.origin.position.y) / self.map_metadata.resolution)
        
        # Check a larger radius around the point for unknown cells (increased from 3 to 8)
        radius = 8  # cells (larger radius to better assess unknown area density)
        unknown_weighted_sum = 0.0
        total_weight = 0.0
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if (0 <= check_x < self.current_map.info.width and 
                    0 <= check_y < self.current_map.info.height):
                    # Calculate distance from center
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Weight by inverse distance (closer unknown cells count more)
                    weight = 1.0 / (dist + 1.0)
                    total_weight += weight
                    
                    if map_data[check_y, check_x] == -1:
                        unknown_weighted_sum += weight
        
        if total_weight == 0:
            return 0.0
        
        return unknown_weighted_sum / total_weight

    def calculate_clearance(self, x, y):
        """
        Calculate the clearance (free space) around a point.
        Returns a score from 0-1 where higher means more open space.
        This helps prioritize frontiers that are easier to reach.
        """
        if self.current_map is None or self.map_metadata is None:
            return 0.5  # Default neutral score

        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )

        resolution = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y

        # Convert to map coordinates
        map_x = int((x - origin_x) / resolution)
        map_y = int((y - origin_y) / resolution)

        # Check bounds
        if not (0 <= map_x < self.current_map.info.width and
                0 <= map_y < self.current_map.info.height):
            return 0.0

        # Search radius in cells (about 2.0 meters at 0.05m resolution)
        # Increased from 1.5m to be more conservative about detecting nearby obstacles
        search_radius = int(2.0 / resolution)

        # Count free cells vs obstacle cells in the area
        free_count = 0
        obstacle_count = 0
        total_checked = 0

        # Also track minimum distance to nearest obstacle
        min_obstacle_dist = float('inf')

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = map_x + dx
                check_y = map_y + dy

                # Circular search area
                dist_sq = dx*dx + dy*dy
                if dist_sq > search_radius * search_radius:
                    continue

                if (0 <= check_x < self.current_map.info.width and
                    0 <= check_y < self.current_map.info.height):
                    cell_value = map_data[check_y, check_x]
                    total_checked += 1

                    if cell_value == 0:  # Free space
                        free_count += 1
                    elif cell_value > 50:  # Obstacle
                        obstacle_count += 1
                        dist = math.sqrt(dist_sq) * resolution
                        if dist < min_obstacle_dist:
                            min_obstacle_dist = dist

        if total_checked == 0:
            return 0.5

        # Calculate clearance score
        # Factor 1: Ratio of free space (0-1)
        free_ratio = free_count / total_checked

        # Factor 2: Distance to nearest obstacle (normalized, 0-1)
        # More distance = better, max at 2.0m (matches search radius)
        if min_obstacle_dist == float('inf'):
            obstacle_dist_score = 1.0  # No obstacles nearby
        else:
            obstacle_dist_score = min(min_obstacle_dist / 2.0, 1.0)

        # Combined clearance score (weighted average)
        # Weight free ratio more heavily to avoid cluttered areas
        clearance_score = 0.7 * free_ratio + 0.3 * obstacle_dist_score

        return clearance_score
    
    def get_min_distance_to_obstacle(self, x, y):
        """
        Calculate the minimum distance to the nearest obstacle from a point.
        Returns the distance in meters, or float('inf') if no obstacles found.
        """
        if self.current_map is None or self.map_metadata is None:
            return float('inf')
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        resolution = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y
        
        # Convert to map coordinates
        map_x = int((x - origin_x) / resolution)
        map_y = int((y - origin_y) / resolution)
        
        # Check bounds
        if not (0 <= map_x < self.current_map.info.width and
                0 <= map_y < self.current_map.info.height):
            return 0.0  # Out of bounds, consider as obstacle
        
        # Search radius in cells (about 2.0 meters at 0.05m resolution)
        search_radius = int(2.0 / resolution)
        
        min_obstacle_dist = float('inf')
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                # Circular search area
                dist_sq = dx*dx + dy*dy
                if dist_sq > search_radius * search_radius:
                    continue
                
                if (0 <= check_x < self.current_map.info.width and
                    0 <= check_y < self.current_map.info.height):
                    cell_value = map_data[check_y, check_x]
                    
                    if cell_value > 50:  # Obstacle
                        dist = math.sqrt(dist_sq) * resolution
                        if dist < min_obstacle_dist:
                            min_obstacle_dist = dist
        
        return min_obstacle_dist if min_obstacle_dist != float('inf') else float('inf')

    def calculate_path_clearance(self, start_x, start_y, end_x, end_y):
        """
        Calculate the average clearance along a straight-line path between two points.
        This helps prioritize paths that go through more open areas.
        """
        avg, _ = self.calculate_path_clearance_detailed(start_x, start_y, end_x, end_y)
        return avg

    def calculate_path_clearance_detailed(self, start_x, start_y, end_x, end_y):
        """
        Calculate both average and minimum clearance along a path.
        Returns (avg_clearance, min_clearance).
        The minimum clearance is critical for detecting narrow passages.
        """
        if self.current_map is None or self.map_metadata is None:
            return 0.5, 0.5

        # Sample points along the path
        dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if dist < 0.1:
            clearance = self.calculate_clearance(end_x, end_y)
            return clearance, clearance

        # Sample every 0.2 meters along the path for very detailed narrow passage detection
        # This ensures we don't miss thin obstacles like table legs
        num_samples = max(int(dist / 0.2), 5)
        clearances = []

        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)
            clearances.append(self.calculate_clearance(sample_x, sample_y))

        if not clearances:
            return 0.5, 0.5

        avg_clearance = sum(clearances) / len(clearances)
        min_clearance = min(clearances)

        return avg_clearance, min_clearance

    def calculate_unknown_region_size(self, x, y):
        """Calculate the size of the unknown region near a frontier point using flood-fill"""
        if self.current_map is None or self.map_metadata is None:
            return 0.0
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Convert to map coordinates
        map_x = int((x - self.map_metadata.origin.position.x) / self.map_metadata.resolution)
        map_y = int((y - self.map_metadata.origin.position.y) / self.map_metadata.resolution)
        
        # Check bounds
        if (map_x < 0 or map_x >= self.current_map.info.width or
            map_y < 0 or map_y >= self.current_map.info.height):
            return 0.0
        
        # Find a nearby unknown cell to start flood-fill from
        # Search in a small radius for an unknown cell
        search_radius = 5
        start_x, start_y = None, None
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if (0 <= check_x < self.current_map.info.width and 
                    0 <= check_y < self.current_map.info.height):
                    if map_data[check_y, check_x] == -1:  # Unknown cell
                        start_x, start_y = check_x, check_y
                        break
            if start_x is not None:
                break
        
        if start_x is None:
            return 0.0  # No unknown cell found nearby
        
        # Flood-fill to count connected unknown cells
        # Use a set to track visited cells
        visited = set()
        stack = [(start_x, start_y)]
        unknown_count = 0
        max_count = 10000  # Limit to prevent infinite loops on very large maps
        
        while stack and unknown_count < max_count:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            
            # Check bounds
            if (cx < 0 or cx >= self.current_map.info.width or
                cy < 0 or cy >= self.current_map.info.height):
                continue
            
            # Check if this is an unknown cell
            if map_data[cy, cx] != -1:
                continue
            
            visited.add((cx, cy))
            unknown_count += 1
            
            # Add neighbors (4-connected)
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
        
        # Convert cell count to area in square meters
        cell_area = self.map_metadata.resolution * self.map_metadata.resolution
        return unknown_count * cell_area
    
    def navigate_to_frontier(self, x, y, frontier_key=None):
        """Navigate to a frontier point (or nearby reachable point)"""
        # Check if we're already at this location
        robot_pose = self.get_robot_pose()
        if robot_pose:
            rx, ry = robot_pose
            dist = math.sqrt((rx - x)**2 + (ry - y)**2)
            if dist < self.min_distance_to_goal:
                self.get_logger().info(f'Already at or very close to ({x:.2f}, {y:.2f}), marking as visited')
                if frontier_key:
                    self.visited_frontiers.add(frontier_key)
                return False
        
        # Find a reachable point near the frontier (goal doesn't need to be exact)
        goal_x, goal_y = x, y
        if not self.is_reachable(x, y):
            # Try to find a nearby reachable point
            nearby = self.find_nearby_reachable_point(x, y, max_search_radius=2.0)
            if nearby:
                goal_x, goal_y = nearby
                self.get_logger().info(f'Frontier at ({x:.2f}, {y:.2f}) not directly reachable, using nearby point ({goal_x:.2f}, {goal_y:.2f})')
            else:
                self.get_logger().warn(f'Cannot find reachable point near frontier ({x:.2f}, {y:.2f}), marking as unreachable')
                if frontier_key:
                    self.unreachable_frontiers.add(frontier_key)
                return False
        
        # Safety check: Verify the goal is in free space and has reasonable clearance
        # Use a more lenient requirement for hallways (0.15m = ~6 inches minimum)
        # This allows navigation in narrow spaces while still avoiding goals right next to walls
        min_obstacle_dist = self.get_min_distance_to_obstacle(goal_x, goal_y)
        MIN_GOAL_DISTANCE = 0.15  # 6 inches minimum distance from obstacles (reduced from 1 ft for hallway navigation)
        if min_obstacle_dist < MIN_GOAL_DISTANCE:
            self.get_logger().info(
                f'Goal at ({goal_x:.2f}, {goal_y:.2f}) rejected - too close to obstacle '
                f'(distance: {min_obstacle_dist:.3f}m < {MIN_GOAL_DISTANCE:.3f}m / 6 inches)'
            )
            if frontier_key:
                self.unreachable_frontiers.add(frontier_key)
            return False
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_x)
        goal_msg.pose.pose.position.y = float(goal_y)
        goal_msg.pose.pose.position.z = 0.0
        
        # Calculate orientation towards the original frontier (not the adjusted goal)
        if robot_pose:
            rx, ry = robot_pose
            yaw = math.atan2(y - ry, x - rx)
            goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(
            f'🎯 SENDING NAV GOAL: Frontier at ({x:.2f}, {y:.2f}), goal at ({goal_x:.2f}, {goal_y:.2f}). '
            f'Obstacle too close: {self.obstacle_too_close}, E-stop: {self.estop_active}'
        )
        
        # Update target for visualization (show original frontier location)
        self.current_target = (x, y)
        # Store actual goal location sent to Nav2 for completion checking
        self.actual_goal_location = (goal_x, goal_y)
        self.current_frontier_key = frontier_key  # Store key for marking visited when close enough
        
        # Don't mark as visited yet - wait until we successfully reach it or get within 2 feet
        
        # Send goal asynchronously
        send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=None
        )
        
        # Use a callback to handle the result
        def goal_response_callback(future):
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn('Goal was rejected! Marking frontier as unreachable')
                    self.current_target = None
                    self.actual_goal_location = None
                    self.current_frontier_key = None
                    if frontier_key:
                        self.unreachable_frontiers.add(frontier_key)
                        self.visited_frontiers.discard(frontier_key)  # Remove if rejected
                    return
                
                self.navigation_goal_handle = goal_handle
                self.last_goal_time = time.time()
                
                # Set up result callback to clear goal handle when done
                result_future = goal_handle.get_result_async()
                
                def result_callback(result_future):
                    try:
                        result = result_future.result()
                        if result.status == 4:  # SUCCEEDED
                            self.get_logger().info('Navigation goal reached (nav2 confirmed)!')
                            # Mark frontier as visited only after successful navigation
                            if frontier_key:
                                self.visited_frontiers.add(frontier_key)
                        elif result.status == 6:  # ABORTED (usually means path planning failed)
                            self.get_logger().warn(f'Navigation goal aborted (likely unreachable), marking frontier as unreachable')
                            if frontier_key:
                                self.unreachable_frontiers.add(frontier_key)
                                self.visited_frontiers.discard(frontier_key)  # Remove from visited if it was there
                        else:
                            self.get_logger().info(f'Navigation goal finished with status: {result.status}')
                            # For other failures, don't mark as unreachable yet (might be temporary)
                    except Exception as e:
                        self.get_logger().warn(f'Error getting navigation result: {e}')
                    finally:
                        self.navigation_goal_handle = None
                        self.current_target = None
                        self.actual_goal_location = None
                        self.current_frontier_key = None
                
                result_future.add_done_callback(result_callback)
            except Exception as e:
                self.get_logger().warn(f'Error in goal response: {e}')
                self.current_target = None
                self.actual_goal_location = None
                self.current_frontier_key = None
                if frontier_key:
                    self.unreachable_frontiers.add(frontier_key)
                    self.visited_frontiers.discard(frontier_key)
        
        send_goal_future.add_done_callback(goal_response_callback)
        
        return True
    
    def publish_visualization(self):
        """Publish visualization markers for target, frontiers, and detected people"""
        # Clean up old person locations
        self.cleanup_old_person_locations()
        
        # Check map frame availability
        self.check_map_frame_available()
        
        # Only publish if map frame is available
        if not self.map_frame_available:
            return
        
        # Publish person location markers (always, regardless of robot state)
        self.publish_person_location_markers()
        
        # Publish target marker
        if self.current_target:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'exploration_target'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(self.current_target[0])
            marker.pose.position.y = float(self.current_target[1])
            marker.pose.position.z = 0.2
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()  # Never expire
            self.target_marker_pub.publish(marker)
        
        # Publish frontier markers (show all detected frontiers)
        if self.current_map is not None:
            frontiers = self.find_frontiers()
            if frontiers:
                marker_array = MarkerArray()
                for i, (fx, fy) in enumerate(frontiers[:50]):  # Limit to 50 for performance
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'frontiers'
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose.position.x = float(fx)
                    marker.pose.position.y = float(fy)
                    marker.pose.position.z = 0.1
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 0.5
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
                    marker_array.markers.append(marker)
                
                # Delete old markers
                delete_marker = Marker()
                delete_marker.header.frame_id = 'map'
                delete_marker.header.stamp = self.get_clock().now().to_msg()
                delete_marker.action = Marker.DELETEALL
                marker_array.markers.insert(0, delete_marker)
                
                self.frontier_markers_pub.publish(marker_array)
    
    def publish_person_location_markers(self):
        """Publish markers for all detected people - human-shaped with waving text"""
        try:
            marker_array = MarkerArray()
            current_time = self.get_clock().now()
            
            # Delete all previous markers first (for all namespaces we use)
            for ns in ['person_body', 'person_head', 'person_text', 'waving_alert']:
                delete_marker = Marker()
                delete_marker.header.frame_id = 'map'
                delete_marker.header.stamp = current_time.to_msg()
                delete_marker.ns = ns
                delete_marker.action = Marker.DELETEALL
                marker_array.markers.append(delete_marker)
            
            # Track if anyone is waving
            anyone_waving = False
            
            # Add markers for all detected people
            for i, (person_x, person_y, timestamp, is_waving) in enumerate(self.detected_people):
                if is_waving:
                    anyone_waving = True
                
                # Colors: Bright magenta/pink for waving, green for not waving
                if is_waving:
                    body_color = (1.0, 0.0, 0.8, 1.0)  # Magenta/pink
                    head_color = (1.0, 0.4, 0.9, 1.0)  # Lighter pink
                else:
                    body_color = (0.2, 0.8, 0.2, 1.0)  # Green
                    head_color = (0.9, 0.8, 0.6, 1.0)  # Skin tone
                
                # --- Body (Cylinder) ---
                body_marker = Marker()
                body_marker.header.frame_id = 'map'
                body_marker.header.stamp = current_time.to_msg()
                body_marker.ns = 'person_body'
                body_marker.id = i
                body_marker.type = Marker.CYLINDER
                body_marker.action = Marker.ADD
                body_marker.pose.position.x = float(person_x)
                body_marker.pose.position.y = float(person_y)
                body_marker.pose.position.z = 0.5  # Center of cylinder (1m tall, bottom at ground)
                body_marker.pose.orientation.x = 0.0
                body_marker.pose.orientation.y = 0.0
                body_marker.pose.orientation.z = 0.0
                body_marker.pose.orientation.w = 1.0
                body_marker.scale.x = 0.4  # Diameter
                body_marker.scale.y = 0.4  # Diameter
                body_marker.scale.z = 1.0  # Height (1 meter)
                body_marker.color.r = body_color[0]
                body_marker.color.g = body_color[1]
                body_marker.color.b = body_color[2]
                body_marker.color.a = body_color[3]
                body_marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
                marker_array.markers.append(body_marker)
                
                # --- Head (Sphere) ---
                head_marker = Marker()
                head_marker.header.frame_id = 'map'
                head_marker.header.stamp = current_time.to_msg()
                head_marker.ns = 'person_head'
                head_marker.id = i
                head_marker.type = Marker.SPHERE
                head_marker.action = Marker.ADD
                head_marker.pose.position.x = float(person_x)
                head_marker.pose.position.y = float(person_y)
                head_marker.pose.position.z = 1.2  # Above body
                head_marker.pose.orientation.x = 0.0
                head_marker.pose.orientation.y = 0.0
                head_marker.pose.orientation.z = 0.0
                head_marker.pose.orientation.w = 1.0
                head_marker.scale.x = 0.25  # Head diameter
                head_marker.scale.y = 0.25
                head_marker.scale.z = 0.25
                head_marker.color.r = head_color[0]
                head_marker.color.g = head_color[1]
                head_marker.color.b = head_color[2]
                head_marker.color.a = head_color[3]
                head_marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
                marker_array.markers.append(head_marker)
                
                # --- "WAVING!" Text above waving people ---
                if is_waving:
                    text_marker = Marker()
                    text_marker.header.frame_id = 'map'
                    text_marker.header.stamp = current_time.to_msg()
                    text_marker.ns = 'person_text'
                    text_marker.id = i
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.pose.position.x = float(person_x)
                    text_marker.pose.position.y = float(person_y)
                    text_marker.pose.position.z = 1.7  # Above head
                    text_marker.pose.orientation.w = 1.0
                    text_marker.scale.z = 0.3  # Text height
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 0.0  # Yellow text
                    text_marker.color.a = 1.0
                    text_marker.text = "WAVING!"
                    text_marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
                    marker_array.markers.append(text_marker)
            
            # --- Global "WAVING DETECTED" alert at top of RViz view ---
            if anyone_waving:
                alert_marker = Marker()
                alert_marker.header.frame_id = 'base_link'  # Relative to robot
                alert_marker.header.stamp = current_time.to_msg()
                alert_marker.ns = 'waving_alert'
                alert_marker.id = 0
                alert_marker.type = Marker.TEXT_VIEW_FACING
                alert_marker.action = Marker.ADD
                alert_marker.pose.position.x = 2.0  # In front of robot
                alert_marker.pose.position.y = 0.0
                alert_marker.pose.position.z = 2.0  # Above robot
                alert_marker.pose.orientation.w = 1.0
                alert_marker.scale.z = 0.5  # Large text
                alert_marker.color.r = 1.0
                alert_marker.color.g = 0.2
                alert_marker.color.b = 0.2  # Red alert color
                alert_marker.color.a = 1.0
                alert_marker.text = "⚠ WAVING DETECTED! ⚠"
                alert_marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
                marker_array.markers.append(alert_marker)
                self.waving_text_visible = True
            else:
                self.waving_text_visible = False
            
            # Always publish (even if no people) to keep topic active and clear old markers
            if len(self.detected_people) > 0:
                self.get_logger().debug(f'Publishing {len(self.detected_people)} person location markers')
            self.person_location_marker_pub.publish(marker_array)
            
            # Also publish as PointCloud2 (alternative visualization)
            self.publish_person_location_cloud()
        except Exception as e:
            self.get_logger().error(f'Error publishing person location markers: {e}', exc_info=True)
    
    def publish_person_location_cloud(self):
        """Publish person locations as PointCloud2 (alternative to MarkerArray)"""
        try:
            cloud_msg = PointCloud2()
            cloud_msg.header.frame_id = 'map'
            cloud_msg.header.stamp = self.get_clock().now().to_msg()
            
            if len(self.detected_people) == 0:
                # Publish empty cloud
                cloud_msg.height = 1
                cloud_msg.width = 0
                cloud_msg.is_dense = True
                self.person_location_cloud_pub.publish(cloud_msg)
                return
            
            # Create point cloud data
            points = np.array([
                [float(x), float(y), 0.5]  # x, y, z (z = 0.5m above ground)
                for x, y, _, _ in self.detected_people
            ], dtype=np.float32)
            
            # Define point fields (x, y, z, rgb)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]
            
            # Add RGB color - magenta for waving, green for not waving
            def pack_rgb(r, g, b):
                """Pack RGB values into uint32 for PointCloud2"""
                return (int(r) | (int(g) << 8) | (int(b) << 16))
            
            # Create colors based on waving state
            colors = []
            for x, y, t, is_waving in self.detected_people:
                if is_waving:
                    colors.append(pack_rgb(255, 0, 200))  # Magenta for waving
                else:
                    colors.append(pack_rgb(50, 200, 50))  # Green for not waving
            colors = np.array(colors, dtype=np.uint32)
            
            # Combine points and colors
            points_with_color = np.zeros(len(points), dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
            ])
            points_with_color['x'] = points[:, 0]
            points_with_color['y'] = points[:, 1]
            points_with_color['z'] = points[:, 2]
            points_with_color['rgb'] = colors
            
            # Set cloud properties
            cloud_msg.height = 1
            cloud_msg.width = len(points)
            cloud_msg.fields = fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = 16  # 4 bytes per field * 4 fields
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True
            cloud_msg.data = points_with_color.tobytes()
            
            self.person_location_cloud_pub.publish(cloud_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing person location cloud: {e}', exc_info=True)
    
    def explore_callback(self):
        """Main exploration callback"""
        if not self.exploring:
            return

        # Don't explore if doing a pivot turn
        if self.pivot_in_progress:
            return

        # Don't explore if in wall hugging mode
        if self.wall_hugging_mode:
            return

        # Don't explore if we're approaching a person
        if self.approaching_person:
            return

        # Don't explore if we're navigating to an obstacle-free area (stuck recovery)
        if self.navigating_to_clear_area:
            return
        
        if self.current_map is None:
            self.get_logger().debug('No map received yet...')
            return
        
        # Check if map frame is available
        self.check_map_frame_available()
        if not self.map_frame_available:
            self.get_logger().debug('Map frame not available yet, waiting for SLAM to initialize...')
            return
        
        # If we're currently navigating, don't interrupt
        # (timeout handler will clear it if needed)
        if self.navigation_goal_handle is not None:
            # Periodically log that we're still navigating
            if not hasattr(self, '_last_nav_active_log'):
                self._last_nav_active_log = 0
            current_time = time.time()
            if current_time - self._last_nav_active_log > 5.0:
                self.get_logger().info(f'🚀 Navigation active - goal handle exists (obstacle_close={self.obstacle_too_close})')
                self._last_nav_active_log = current_time
            return

        # Update visited areas
        self.update_visit_timestamps()
        
        # Enclosure check is now done via timer (every 5 seconds) - see check_enclosure_callback()
        # Check if we should switch to wall hugging or patrol mode based on previous enclosure check
        if hasattr(self, '_last_enclosure_check_result'):
            is_enclosed, exploration_pct, should_stop = self._last_enclosure_check_result
        else:
            is_enclosed, exploration_pct, should_stop = False, 0.0, False
            
        if should_stop and not self.wall_hug_complete:
            self.get_logger().info(
                f'Enclosed area is {exploration_pct:.1f}% explored (>=80%). '
                'Starting wall-hugging mode to complete border mapping.'
            )
            # Cancel any current navigation
            self.cancel_current_goal()
            # Start wall hugging mode
            self.wall_hugging_mode = True
            self.wall_hug_start_pose = self.get_robot_pose()
            self.last_wall_hug_pose = self.wall_hug_start_pose
            self.wall_hug_distance_traveled = 0.0
            return
        elif should_stop and self.wall_hug_complete:
            # Wall hugging done, switch to patrol mode for person detection
            if not self.patrol_mode:
                self.get_logger().info(
                    '🚨 Initial exploration complete! Switching to PATROL mode for person detection.'
                )
                self.patrol_mode = True
                # Save map if not already saved
                if not self.map_saved:
                    self.save_map()
            # Continue to patrol mode exploration below

        # Find frontiers
        frontiers = self.find_frontiers()
        if not frontiers:
            self.get_logger().info('No frontiers found. Map might be complete!')
            return
        
        self.get_logger().info(f'Found {len(frontiers)} frontier points')
        
        # Cluster frontiers
        clusters = self.cluster_frontiers(frontiers)
        if not clusters:
            self.get_logger().info('No frontier clusters found')
            return
        
        self.get_logger().info(f'Found {len(clusters)} frontier clusters')
        
        # Select best frontier
        best = self.select_best_frontier(clusters)
        if best is None:
            self.get_logger().info('No suitable frontier found (all may be visited or unreachable)')
            # Clear some visited/unreachable frontiers if we've explored a lot
            if len(self.visited_frontiers) > 50 or len(self.unreachable_frontiers) > 50:
                self.get_logger().info('Clearing old visited/unreachable frontiers to allow re-exploration')
                self.visited_frontiers.clear()
                self.unreachable_frontiers.clear()  # Clear unreachable too - map might have changed
            return
        
        # Handle both old format (3 values) and new format (4 values with key)
        if len(best) == 4:
            x, y, size, key = best
        else:
            x, y, size = best
            key = (round(x, 1), round(y, 1))
        
        # Calculate and log clearance for the selected frontier
        robot_pose = self.get_robot_pose()
        if robot_pose:
            rx, ry = robot_pose
            avg_clear, min_clear = self.calculate_path_clearance_detailed(rx, ry, x, y)
            self.get_logger().info(
                f'Selected frontier at ({x:.2f}, {y:.2f}) with {size} points - '
                f'Path clearance: avg={avg_clear:.2f}, min={min_clear:.2f} (wider paths prioritized)'
            )
        else:
            self.get_logger().info(f'Selected frontier at ({x:.2f}, {y:.2f}) with {size} points')
        
        # Navigate to it
        self.navigate_to_frontier(x, y, key)

    def pivot_turn_callback(self):
        """Handle periodic 360-degree pivot turns for better LiDAR coverage"""
        # DISABLED: Commented out to prevent regular spinning
        return
        
        # Don't pivot if e-stop active
        # if self.estop_active:
        #     if self.pivot_in_progress:
        #         self.pivot_in_progress = False
        #         self.stop_robot()
        #     return

        # Don't pivot if obstacle too close or backing up
        # if self.obstacle_too_close or self.is_backing_up:
        #     if self.pivot_in_progress:
        #         self.pivot_in_progress = False
        #         self.stop_robot()
        #     return

        # Don't pivot during wall hugging
        # if self.wall_hugging_mode:
        #     return

        # current_time = time.time()

        # Check if we're currently doing a pivot
        # if self.pivot_in_progress:
        #     elapsed = current_time - self.pivot_start_time
        #     if elapsed < self.pivot_duration:
        #         # Continue rotating
        #         twist = Twist()
        #         twist.angular.z = self.pivot_angular_speed
        #         self.cmd_vel_pub.publish(twist)
        #     else:
        #         # Pivot complete
        #         self.pivot_in_progress = False
        #         self.stop_robot()
        #         self.last_pivot_time = current_time
        #         self.get_logger().info('360-degree pivot turn complete')
        #     return

        # Check if it's time for a new pivot
        # if current_time - self.last_pivot_time >= self.pivot_interval:
        #     # Only start pivot if not navigating
        #     if self.navigation_goal_handle is None and not self.approaching_person:
        #         # Check clearance before pivoting (need 2 feet = 0.6096m minimum)
        #         if self.check_pivot_clearance():
        #             self.get_logger().info('✨ Starting 360° pivot turn for person detection (2ft clearance confirmed)')
        #             self.pivot_in_progress = True
        #             self.pivot_start_time = current_time
        #             # Cancel any current navigation
        #             self.cancel_current_goal()
        #         else:
        #             self.get_logger().debug('Skipping pivot - insufficient clearance (need 2ft radius)')
        #             # Try again in 10 seconds
        #             self.last_pivot_time = current_time - self.pivot_interval + 10.0

    def wall_hug_callback(self):
        """Wall hugging behavior - follow the wall in a loop"""
        if not self.wall_hugging_mode:
            return

        # Don't wall hug if e-stop active
        if self.estop_active:
            self.stop_robot()
            return

        # Don't wall hug if obstacle too close or backing up
        if self.obstacle_too_close or self.is_backing_up:
            self.stop_robot()
            return

        if self.current_map is None:
            return

        # Get current robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        # Track distance traveled
        if self.last_wall_hug_pose is not None:
            rx, ry = robot_pose
            lx, ly = self.last_wall_hug_pose
            dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)
            self.wall_hug_distance_traveled += dist
        self.last_wall_hug_pose = robot_pose

        # Check if we've completed a loop (returned to start after traveling significant distance)
        if self.wall_hug_start_pose is not None and self.wall_hug_distance_traveled > 5.0:
            sx, sy = self.wall_hug_start_pose
            rx, ry = robot_pose
            dist_to_start = math.sqrt((rx - sx)**2 + (ry - sy)**2)
            if dist_to_start < 1.0:  # Within 1 meter of start
                self.get_logger().info(
                    f'Wall hug loop complete! Traveled {self.wall_hug_distance_traveled:.1f}m'
                )
                self.wall_hugging_mode = False
                self.wall_hug_complete = True
                self.stop_robot()
                # Save the map
                self.save_map()
                return

        # Simple wall-following behavior using LiDAR data from map
        # Strategy: Keep wall on the right, move forward
        twist = Twist()

        # Get map data for obstacle detection
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        resolution = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y

        # Convert robot position to grid
        rx, ry = robot_pose
        grid_x = int((rx - origin_x) / resolution)
        grid_y = int((ry - origin_y) / resolution)

        # Check for obstacles in different directions (simplified)
        # Look ahead, right, and left in grid cells
        look_distance = int(0.5 / resolution)  # 0.5 meters

        def check_obstacle(dx, dy, distance):
            """Check if obstacle exists in direction"""
            for d in range(1, distance + 1):
                cx = grid_x + dx * d
                cy = grid_y + dy * d
                if 0 <= cx < self.current_map.info.width and 0 <= cy < self.current_map.info.height:
                    if map_data[cy, cx] > 50:  # Obstacle
                        return True, d
            return False, distance

        # Check front, right, left (simplified - would need robot orientation for accuracy)
        front_blocked, front_dist = check_obstacle(1, 0, look_distance)
        right_blocked, right_dist = check_obstacle(0, -1, look_distance)
        left_blocked, left_dist = check_obstacle(0, 1, look_distance)

        # Wall following logic: keep wall on right
        linear_speed = 0.15  # m/s
        angular_speed = 0.3  # rad/s

        if front_blocked and front_dist < look_distance // 2:
            # Obstacle ahead, turn left
            twist.linear.x = 0.0
            twist.angular.z = angular_speed
        elif not right_blocked:
            # No wall on right, turn right to find wall
            twist.linear.x = linear_speed * 0.5
            twist.angular.z = -angular_speed * 0.5
        elif right_blocked and right_dist < look_distance // 3:
            # Too close to wall, turn slightly left
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed * 0.3
        else:
            # Wall on right at good distance, go forward
            twist.linear.x = linear_speed
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        """Send zero velocity command to stop the robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def save_map(self):
        """Save the current map using map_saver_cli"""
        import subprocess
        import os

        # Create save directory if it doesn't exist
        os.makedirs(self.map_save_path, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        map_name = f'map_{timestamp}'
        map_file = os.path.join(self.map_save_path, map_name)

        # Also save as 'latest' for easy loading
        latest_file = os.path.join(self.map_save_path, 'latest')

        try:
            self.get_logger().info(f'Saving map to {map_file}...')
            # Use ros2 run map_server map_saver_cli
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.get_logger().info(f'Map saved successfully: {map_file}.yaml')
                # Create symlink to latest
                latest_yaml = f'{latest_file}.yaml'
                latest_pgm = f'{latest_file}.pgm'
                # Remove old symlinks
                if os.path.exists(latest_yaml):
                    os.remove(latest_yaml)
                if os.path.exists(latest_pgm):
                    os.remove(latest_pgm)
                # Create new symlinks
                os.symlink(f'{map_file}.yaml', latest_yaml)
                os.symlink(f'{map_file}.pgm', latest_pgm)
                self.get_logger().info(f'Latest map symlinked to {latest_yaml}')
                self.map_saved = True
            else:
                self.get_logger().error(f'Map save failed: {result.stderr}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('Map save timed out')
        except Exception as e:
            self.get_logger().error(f'Error saving map: {e}')

    def update_display(self):
        """Update the display window with RGB, pose overlay, and info"""
        if not self.display_enabled:
            return
        
        if self.rgb_image is None:
            return
        
        try:
            # Create a copy for display
            display_image = self.rgb_image.copy()
            
            # Draw pose landmarks if available (simple visualization)
            if (self.current_pose_results is not None and 
                self.current_pose_results.pose_landmarks):
                landmarks = self.current_pose_results.pose_landmarks.landmark
                h, w = display_image.shape[:2]
                
                # Draw keypoints (shoulders, wrists for waving detection)
                keypoint_indices = [5, 6, 9, 10]  # left_shoulder, right_shoulder, left_wrist, right_wrist
                for idx in keypoint_indices:
                    if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                
                # Draw connections (shoulder to wrist)
                if (len(landmarks) > 10 and landmarks[5].visibility > 0.5 and landmarks[9].visibility > 0.5):
                    x1 = int(landmarks[5].x * w)
                    y1 = int(landmarks[5].y * h)
                    x2 = int(landmarks[9].x * w)
                    y2 = int(landmarks[9].y * h)
                    cv2.line(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if (len(landmarks) > 10 and landmarks[6].visibility > 0.5 and landmarks[10].visibility > 0.5):
                    x1 = int(landmarks[6].x * w)
                    y1 = int(landmarks[6].y * h)
                    x2 = int(landmarks[10].x * w)
                    y2 = int(landmarks[10].y * h)
                    cv2.line(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text overlay with pose and location info
            y_offset = 30
            line_height = 25
            h, w = display_image.shape[:2]
            
            # EMERGENCY STOP - BIG RED BANNER AT TOP
            if self.estop_active:
                # Draw red banner
                cv2.rectangle(display_image, (0, 0), (w, 60), (0, 0, 200), -1)
                cv2.putText(display_image, "!!! EMERGENCY STOP ACTIVE !!!", (w//2 - 200, 42),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                y_offset = 80
            
            # FPS and Latency display (top right)
            fps_text = f"FPS: {self.actual_fps:.1f}"
            latency_text = f"Latency: {self.frame_latency_ms:.0f}ms"
            slam_text = f"SLAM: {'OK' if self.slam_working else 'FAIL'}"
            cv2.putText(display_image, fps_text, (w - 120, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.actual_fps >= 4 else (0, 0, 255), 2)
            cv2.putText(display_image, latency_text, (w - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.frame_latency_ms < 200 else (0, 0, 255), 2)
            cv2.putText(display_image, slam_text, (w - 150, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.slam_working else (0, 0, 255), 2)
            
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
            if self.person_location_camera_frame:
                x, y, z = self.person_location_camera_frame
                location_text = f"Location: X={x:.2f}m Y={y:.2f}m Z={z:.2f}m"
                cv2.putText(display_image, location_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_image, "Location: N/A", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            y_offset += line_height
            
            # Exploration status
            if self.estop_active:
                cv2.putText(display_image, "Status: E-STOP ACTIVE", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.approaching_person:
                cv2.putText(display_image, "Status: APPROACHING PERSON", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            elif self.navigation_goal_handle is not None:
                cv2.putText(display_image, "Status: NAVIGATING", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                cv2.putText(display_image, "Status: EXPLORING", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow('Frontier Explorer - BlazePose Detection', display_image)
            
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
                if hasattr(self, 'display_timer'):
                    self.display_timer.cancel()
            else:
                self.get_logger().error(f'OpenCV error: {e}')
    
    def check_slam_health(self):
        """Periodically check SLAM health and warn if transforms are stale"""
        current_time = time.time()
        
        # Check if we've successfully looked up a transform recently
        time_since_last_tf = current_time - self.last_tf_lookup_time
        
        if self.map_frame_available and not self.slam_working:
            self.get_logger().warn(
                'SLAM WARNING: map->base_link transform not available! '
                'Lidar scans will appear to move with robot. '
                'Check that SLAM Toolbox is running and converging.'
            )
        elif self.slam_working and time_since_last_tf > 2.0:
            self.get_logger().warn(
                f'SLAM WARNING: No successful TF lookup in {time_since_last_tf:.1f}s. '
                'Lidar scans may drift.'
            )
        elif self.slam_working:
            # Periodically log that SLAM is healthy (every 10 seconds)
            if not hasattr(self, '_last_slam_ok_log'):
                self._last_slam_ok_log = 0
            if current_time - self._last_slam_ok_log > 10.0:
                self.get_logger().info('SLAM is healthy - transforms updating correctly')
                self._last_slam_ok_log = current_time
    
    def _set_high_priority(self):
        """Set high CPU priority for this process (especially for pose detection)"""
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
            self.get_logger().debug(f"Could not set process priority: {e}")
        
        # Set CPU affinity to high-performance cores (if available)
        self._set_cpu_affinity()
    
    def _set_torch_thread_priority(self):
        """Set high priority for PyTorch threads"""
        try:
            import torch
            # Set PyTorch to use fewer threads but higher priority
            torch.set_num_threads(2)  # Use 2 threads for inference
            
            # Set thread priority using threading module
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
    
    def shutdown(self):
        """Clean shutdown - saves map before exiting"""
        self.get_logger().info('Shutting down - saving final map...')
        self.exploring = False
        self.cancel_current_goal()
        self.current_target = None
        self.actual_goal_location = None

        # Save the map one last time before shutdown
        try:
            self.save_map()
        except Exception as e:
            self.get_logger().error(f'Failed to save map during shutdown: {e}')

        cv2.destroyAllWindows()


def main():
    rclpy.init()

    # Create explorer node
    explorer = FrontierExplorer()

    # Signal handler for graceful shutdown with map saving
    def signal_handler(sig, frame):
        explorer.get_logger().info(f'Received signal {sig}, shutting down gracefully...')
        explorer.shutdown()
        cv2.destroyAllWindows()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Spin the explorer
        explorer.get_logger().info('Starting frontier exploration...')
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        explorer.get_logger().info('Keyboard interrupt - shutting down...')
    except Exception as e:
        explorer.get_logger().error(f'Error in exploration: {e}')
    finally:
        explorer.shutdown()
        cv2.destroyAllWindows()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ignore shutdown errors


if __name__ == '__main__':
    main()

