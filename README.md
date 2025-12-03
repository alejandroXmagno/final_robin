# Install Python dependencies for RealSense viewer (BlazePose detection):
pip3 install -r src/roverrobotics_ros2/roverrobotics_driver/requirements.txt

ros2 launch roverrobotics_driver mini_2wd.launch.py 

ros2 launch roverrobotics_driver slam_launch.py

ros2 launch roverrobotics_driver exploration_launch.py
# Exploration now includes person detection AND multi-layer collision prevention:
# - Detects waving people using BlazePose at 5Hz
# - When waving detected, navigates to person (stops 1 foot away)
# - Waits 10 seconds near person, then resumes exploration
# - RealSense camera is automatically started with exploration
#
# PROGRESSIVE NAVIGATION & SAFETY:
# - Emergency LiDAR monitoring at 10Hz with 240° coverage
# - 3-tier obstacle detection: Slowdown (1.0m), Stop (0.7m), Backup (0.45m)
# - Automatic emergency backup when obstacles < 45cm
# - Path clearance analysis rejects very narrow passages (< 20% clearance)
# - Nav2 costmap inflation (35cm) allows indoor navigation
# - Progressive path planning: Prefers closer frontiers (3.5m radius)
# - Takes small steps instead of long paths - reduces Nav2 failures
# - Balanced velocity (25cm/s) for smooth navigation

ros2 launch roverrobotics_driver realsense_viewer.launch.py
# RealSense viewer now includes BlazePose detection at 5Hz:
# - Detects human poses
# - Detects waving (hand above shoulder)
# - Calculates person location relative to camera
# - Displays RGB with pose overlay and location info
# Alternative: ros2 launch roverrobotics_driver realsense_simple.launch.py
# Then use: rqt_image_view /camera/realsense/color/image_raw (for RGB)
# And: rqt_image_view /camera/realsense/depth/image_rect_raw (for depth)

rviz2 -d /home/stickykeys/final_robin/src/roverrobotics_ros2/roverrobotics_driver/config/rviz_configs/slam_rviz_layout.rviz