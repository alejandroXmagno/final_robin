#!/usr/bin/env python3
"""
Simple RealSense launch file - just starts the camera
Use rqt_image_view to view the images manually
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rover_dir = get_package_share_directory('roverrobotics_driver')
    realsense_config = os.path.join(rover_dir, 'config', 'realsense_config.yaml')
    
    # RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        name='realsense',
        executable='realsense2_camera_node',
        parameters=[realsense_config],
        output='screen'
    )
    
    ld = LaunchDescription()
    ld.add_action(realsense_node)
    
    return ld

