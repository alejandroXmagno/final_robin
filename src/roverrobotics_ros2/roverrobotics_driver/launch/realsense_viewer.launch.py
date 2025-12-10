#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rover_dir = get_package_share_directory('roverrobotics_driver')
    realsense_config = os.path.join(rover_dir, 'config', 'realsense_config.yaml')
    
    # RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        name='realsense',
        executable='realsense2_camera_node',
        parameters=[
            realsense_config,
            {'rgb_camera.power_line_frequency': 0}  # Explicitly override to disable (0=Disabled, 1=50Hz, 2=60Hz)
        ],
        output='screen'
    )
    
    # Static transform publisher for camera mount position
    # Camera is mounted 3 inches (0.0762m) forward, 13 inches (0.3302m) up from base_link
    # Camera is tilted UP by 20 degrees (pitch = -0.349066 radians)
    # Arguments: x y z yaw pitch roll frame_id child_frame_id
    camera_x = 0.0762  # 3 inches forward
    camera_y = 0.0
    camera_z = 0.3302  # 13 inches up
    camera_yaw = 0.0
    camera_pitch = -0.349066  # -20 degrees (tilted up)
    camera_roll = 0.0
    
    camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_link_tf',
        arguments=[
            str(camera_x), str(camera_y), str(camera_z),
            str(camera_yaw), str(camera_pitch), str(camera_roll),
            'base_link', 'camera_link'
        ]
    )
    
    # Image viewer script
    # Get the source directory path
    # The install directory structure is: install/roverrobotics_driver/share/roverrobotics_driver
    # We need to go up to workspace root, then to src
    package_share_dir = get_package_share_directory("roverrobotics_driver")
    # From install/roverrobotics_driver/share/roverrobotics_driver go to workspace root
    # install/roverrobotics_driver/share/roverrobotics_driver -> .. -> .. -> .. -> .. -> workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(package_share_dir))))  # workspace root (final_robin)
    scripts_dir = os.path.join(workspace_root, 'src', 'roverrobotics_ros2', 'roverrobotics_driver', 'scripts')
    viewer_script = os.path.join(scripts_dir, 'realsense_viewer.py')
    
    viewer_node = ExecuteProcess(
        cmd=['python3', viewer_script],
        output='screen',
        name='realsense_viewer'
    )
    
    ld = LaunchDescription()
    ld.add_action(camera_tf_node)  # Publish camera mount transform (with 20 degree tilt)
    ld.add_action(realsense_node)
    ld.add_action(viewer_node)
    
    return ld

