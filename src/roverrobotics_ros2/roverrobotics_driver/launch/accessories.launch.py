#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from math import pi
import yaml

def generate_launch_description():
    ld = LaunchDescription()

    accessories_config_path = Path(get_package_share_directory(
        'roverrobotics_driver'), 'config/accessories.yaml')

     # Read the config file
    with open(accessories_config_path, 'r') as f:
        accessories_config = yaml.load(f, Loader=yaml.FullLoader)

    
    # RP Lidar Setup
    if accessories_config.get('rplidar', {}).get('ros__parameters', {}).get('active', False):
        lidar_node = Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar',
            parameters=[accessories_config_path],
            remappings=[('scan', 'scan_raw')],  # Remap to scan_raw so filter can process it
            output='screen')
    
        # Add RPLidar S2 to launch description
        ld.add_action(lidar_node)
        
        # Laser footprint filter - filters out robot footprint and standoffs
        # The filter subscribes to scan_raw and publishes to scan
        filter_script_path = os.path.join(
            get_package_share_directory('roverrobotics_driver'),
            '..', '..', 'lib', 'roverrobotics_driver', 'laser_footprint_filter'
        )
        laser_filter_node = ExecuteProcess(
            cmd=['python3', filter_script_path],
            output='screen',
            name='laser_footprint_filter'
        )
        ld.add_action(laser_filter_node)
    
    # BNO055 IMU Setup
    if accessories_config.get('bno055', {}).get('ros__parameters', {}).get('active', False):
        bno055_node = Node(
            package = 'bno055',
            name = 'bno055',
            executable = 'bno055',
            parameters = [accessories_config_path],
            remappings=[
                ('/imu', '/imu/data')
            ])
        
        # Add BNO055 IMU to launch description
        ld.add_action(bno055_node)

    # Realsense Node
    if accessories_config.get('realsense', {}).get('ros__parameters', {}).get('active', False):
        realsense_node = Node(
            package='realsense2_camera',
            name="realsense",
            executable='realsense2_camera_node',
            parameters=[accessories_config_path],
            output='screen')

        # Add Realsense d435i to launch description
        ld.add_action(realsense_node)

    return ld

