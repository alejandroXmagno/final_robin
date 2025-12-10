from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    rover_path = get_package_share_path('roverrobotics_description')
    default_model_path = rover_path / 'urdf/mini_2wd.urdf'
    
    model_arg = DeclareLaunchArgument(name='model', default_value=str(default_model_path),
                                      description='Absolute path to robot urdf file')
    record_standoff_arg = DeclareLaunchArgument(
        name='record_standoff',
        default_value='false',
        description='If true, record standoff positions on startup. If false, use existing config file.'
    )
    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)
   
    hardware_config = Path(get_package_share_directory(
        'roverrobotics_driver'), 'config', 'mini_2wd_config.yaml')
    assert hardware_config.is_file()

    ld = LaunchDescription()

    robot_driver = Node(
        package = 'roverrobotics_driver',
        name = 'roverrobotics_driver',
        executable = 'roverrobotics_driver',
        parameters = [hardware_config],
        output='screen',
        respawn=True,
        respawn_delay=1
    )

    accessories_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_package_share_directory('roverrobotics_driver'), '/launch/accessories.launch.py']),
    )
   
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher'
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )
    
    # Get the source directory path
    # The install directory structure is: install/roverrobotics_driver/share/roverrobotics_driver
    # We need to go up to workspace root, then to src
    package_share_dir = get_package_share_directory("roverrobotics_driver")
    # From install/roverrobotics_driver/share/roverrobotics_driver go to workspace root
    # install/roverrobotics_driver/share/roverrobotics_driver -> .. -> .. -> .. -> .. -> workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(package_share_dir))))  # workspace root (final_robin)
    scripts_dir = os.path.join(workspace_root, 'src', 'roverrobotics_ros2', 'roverrobotics_driver', 'scripts')
    
    # Lidar marker publisher - shows a sphere at lidar_link position in RViz
    lidar_marker_script = os.path.join(scripts_dir, 'lidar_marker.py')
    lidar_marker_node = ExecuteProcess(
        cmd=['python3', lidar_marker_script],
        output='screen',
        name='lidar_marker'
    )
    
    # Standoff detector - records lidar readings < 1 foot and clusters into 4 points
    # Only launch if record_standoff argument is true
    standoff_detector_script = os.path.join(scripts_dir, 'standoff_detector.py')
    standoff_detector_node = ExecuteProcess(
        cmd=['python3', standoff_detector_script],
        output='screen',
        name='standoff_detector',
        condition=IfCondition(LaunchConfiguration('record_standoff'))
    )
    
    ld.add_action(model_arg)
    ld.add_action(record_standoff_arg)
    ld.add_action(robot_driver)
    ld.add_action(accessories_launch)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(lidar_marker_node)
    ld.add_action(standoff_detector_node)
   
    return ld
