#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess,
                            OpaqueFunction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    rover_dir = get_package_share_directory('roverrobotics_driver')
    bringup_dir = get_package_share_directory('nav2_bringup')
    launch_dir = os.path.join(bringup_dir, 'launch')

    # Create the launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')
    load_map = LaunchConfiguration('load_map')
    map_file = LaunchConfiguration('map_file')

    # Default map file path (latest saved map)
    default_map_path = '/home/stickykeys/final_robin/saved_maps/latest.yaml'

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace',
        default_value='false',
        description='Whether to apply a namespace to the navigation stack')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(rover_dir, 'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', default_value='False',
        description='Whether to use composed bringup')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='log level')

    declare_load_map_cmd = DeclareLaunchArgument(
        'load_map', default_value='false',
        description='Whether to load a previously saved map')

    declare_map_file_cmd = DeclareLaunchArgument(
        'map_file', default_value=default_map_path,
        description='Full path to map yaml file to load (default: latest saved map)')

    # Robot localizer launch
    rl_launch_path = os.path.join(get_package_share_directory("roverrobotics_driver"), 'launch', 'robot_localizer.launch.py')
    robot_localizer_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(rl_launch_path),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # Nav2 backend launch
    nav2_backend_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(rover_dir, 'launch', 'nav2_backend.py')),
        launch_arguments={'namespace': namespace,
                          'use_sim_time': use_sim_time,
                          'autostart': autostart,
                          'params_file': params_file,
                          'use_composition': use_composition,
                          'use_respawn': use_respawn,
                          'container_name': 'nav2_container'}.items())

    # RealSense camera node (needed for person detection during exploration)
    realsense_config = os.path.join(rover_dir, 'config', 'realsense_config.yaml')
    realsense_node = Node(
        package='realsense2_camera',
        name='realsense',
        executable='realsense2_camera_node',
        parameters=[
            realsense_config,
            {'rgb_camera.power_line_frequency': 0}  # Explicitly override to disable
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
    
    # Exploration node
    # Get the source directory path
    # The install directory structure is: install/roverrobotics_driver/share/roverrobotics_driver
    # We need to go up to workspace root, then to src
    package_share_dir = get_package_share_directory("roverrobotics_driver")
    # From install/roverrobotics_driver/share/roverrobotics_driver go to workspace root
    # install/roverrobotics_driver/share/roverrobotics_driver -> .. -> .. -> .. -> .. -> workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(package_share_dir))))  # workspace root (final_robin)
    scripts_dir = os.path.join(workspace_root, 'src', 'roverrobotics_ros2', 'roverrobotics_driver', 'scripts')
    exploration_script = os.path.join(scripts_dir, 'frontier_explorer.py')
    
    exploration_node = ExecuteProcess(
        cmd=['python3', exploration_script],
        output='screen',
        name='frontier_explorer'
    )

    # SLAM Toolbox - always runs in mapping mode for exploration
    # When load_map:=true, use RViz SLAM Toolbox panel to deserialize a saved map
    slam_params_mapping = os.path.join(rover_dir, 'config', 'slam_configs', 'mapper_params_online_async.yaml')
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_mapping,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Map server node (only launched when load_map:=true)
    map_server_node = Node(
        condition=IfCondition(load_map),
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_file
        }]
    )

    # Lifecycle manager for map server (only when loading a map)
    map_lifecycle_manager = Node(
        condition=IfCondition(load_map),
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_load_map_cmd)
    ld.add_action(declare_map_file_cmd)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(robot_localizer_launch)
    ld.add_action(nav2_backend_launch)
    ld.add_action(slam_toolbox_node)  # SLAM Toolbox for mapping (when load_map:=false)
    ld.add_action(camera_tf_node)  # Publish camera mount transform (with 20 degree tilt)
    ld.add_action(realsense_node)  # Start camera for person detection
    ld.add_action(map_server_node)  # Map server (only when load_map:=true)
    ld.add_action(map_lifecycle_manager)  # Lifecycle manager for map server
    ld.add_action(exploration_node)

    return ld

