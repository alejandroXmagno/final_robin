#!/usr/bin/env python3
"""
Standalone exploration script that launches nav2 nodes internally
and makes the robot wander around to expand the map.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np
import math
import subprocess
import signal
import sys
import os
from threading import Thread
import time

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # State
        self.current_map = None
        self.map_metadata = None
        self.navigation_goal_handle = None
        self.exploring = True
        self.last_goal_time = time.time()
        self.goal_timeout = 60.0  # seconds to wait before giving up on a goal
        
        # Exploration parameters
        self.min_frontier_size = 20  # minimum number of frontier cells
        self.exploration_radius = 3.0  # max distance to explore from current position
        
        # Wait for nav2 action server
        self.get_logger().info('Waiting for navigate_to_pose action server...')
        if not self.nav_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('NavigateToPose action server not available!')
            return
        
        self.get_logger().info('NavigateToPose action server available!')
        
        # Start exploration timer
        self.explore_timer = self.create_timer(5.0, self.explore_callback)
        
        # Start goal timeout checker
        self.timeout_timer = self.create_timer(1.0, self.check_goal_timeout)
        
    def map_callback(self, msg):
        """Store the latest map"""
        self.current_map = msg
        self.map_metadata = msg.info
        
    def check_goal_timeout(self):
        """Check if current goal has timed out"""
        if self.navigation_goal_handle is not None:
            # Check timeout
            elapsed = time.time() - self.last_goal_time
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f'Goal timed out after {elapsed:.1f}s, canceling...')
                self.cancel_current_goal()
                self.last_goal_time = time.time()
    
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
        """Find frontier points in the current map"""
        if self.current_map is None or self.map_metadata is None:
            return []
        
        map_data = np.array(self.current_map.data, dtype=np.int8).reshape(
            (self.current_map.info.height, self.current_map.info.width)
        )
        
        # Find unknown cells (value = -1) adjacent to free cells (value = 0)
        frontiers = []
        height, width = map_data.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Check if this is an unknown cell
                if map_data[y, x] == -1:
                    # Check neighbors for free space
                    neighbors = [
                        map_data[y-1, x], map_data[y+1, x],
                        map_data[y, x-1], map_data[y, x+1],
                        map_data[y-1, x-1], map_data[y-1, x+1],
                        map_data[y+1, x-1], map_data[y+1, x+1]
                    ]
                    
                    # If any neighbor is free space, this is a frontier
                    if 0 in neighbors:
                        # Convert to world coordinates
                        world_x = self.map_metadata.origin.position.x + x * self.map_metadata.resolution
                        world_y = self.map_metadata.origin.position.y + y * self.map_metadata.resolution
                        frontiers.append((world_x, world_y))
        
        return frontiers
    
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
                if dist < 0.5:  # cluster threshold
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
    
    def get_robot_pose(self):
        """Get current robot pose from tf"""
        try:
            from tf2_ros import Buffer, TransformListener
            
            if not hasattr(self, 'tf_buffer'):
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)
            
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            return (
                transform.transform.translation.x,
                transform.transform.translation.y
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get robot pose: {e}')
            return None
    
    def select_best_frontier(self, clusters):
        """Select the best frontier to explore"""
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            # If we can't get pose, just pick the largest cluster
            if clusters:
                return clusters[0]
            return None
        
        rx, ry = robot_pose
        
        # Score frontiers by size and distance
        scored = []
        for cx, cy, size in clusters:
            dist = math.sqrt((rx - cx)**2 + (ry - cy)**2)
            
            # Prefer larger clusters that are not too far
            if dist > self.exploration_radius:
                continue
            
            # Score: size / distance (higher is better)
            score = size / (dist + 0.1)
            scored.append((score, cx, cy, size))
        
        if not scored:
            return None
        
        # Return highest scoring frontier
        scored.sort(key=lambda x: x[0], reverse=True)
        _, cx, cy, size = scored[0]
        return (cx, cy, size)
    
    def navigate_to_frontier(self, x, y):
        """Navigate to a frontier point"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0  # Face forward
        
        self.get_logger().info(f'Navigating to frontier at ({x:.2f}, {y:.2f})')
        
        send_goal_future = self.nav_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected!')
            return False
        
        self.navigation_goal_handle = goal_handle
        self.last_goal_time = time.time()
        
        # Set up result callback to clear goal handle when done
        result_future = goal_handle.get_result_async()
        
        def result_callback(future):
            try:
                result = future.result()
                if result.status == 4:  # SUCCEEDED
                    self.get_logger().info('Navigation goal reached!')
                else:
                    self.get_logger().info(f'Navigation goal finished with status: {result.status}')
                self.navigation_goal_handle = None
            except Exception as e:
                self.get_logger().warn(f'Error getting navigation result: {e}')
                self.navigation_goal_handle = None
        
        result_future.add_done_callback(result_callback)
        
        return True
    
    def explore_callback(self):
        """Main exploration callback"""
        if not self.exploring:
            return
        
        if self.current_map is None:
            self.get_logger().warn('No map received yet...')
            return
        
        # If we're currently navigating, don't interrupt
        # (timeout handler will clear it if needed)
        if self.navigation_goal_handle is not None:
            return
        
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
            self.get_logger().info('No suitable frontier found')
            return
        
        x, y, size = best
        self.get_logger().info(f'Selected frontier at ({x:.2f}, {y:.2f}) with {size} points')
        
        # Navigate to it
        self.navigate_to_frontier(x, y)
    
    def shutdown(self):
        """Clean shutdown"""
        self.exploring = False
        self.cancel_current_goal()


class Nav2Launcher:
    """Launches nav2 nodes in a subprocess"""
    
    def __init__(self):
        self.processes = []
        import logging
        self.logger = logging.getLogger('Nav2Launcher')
        
    def launch_nav2(self):
        """Launch nav2 nodes (without map_server, using SLAM map)"""
        from ament_index_python.packages import get_package_share_directory
        import os
        
        # Get nav2 params file
        params_file = os.path.join(
            get_package_share_directory('roverrobotics_driver'),
            'config', 'nav2_params.yaml'
        )
        
        # Launch nav2 backend (which doesn't include map_server)
        # We need to source the workspace first
        workspace_path = os.path.expanduser('~/final_robin')
        cmd = [
            'bash', '-c',
            f'source /opt/ros/humble/setup.bash && '
            f'source {workspace_path}/install/setup.bash && '
            f'ros2 launch roverrobotics_driver nav2_backend.py '
            f'params_file:={params_file} '
            f'use_sim_time:=false '
            f'autostart:=true '
            f'use_composition:=False '
            f'use_respawn:=False'
        ]
        
        self.logger.info('Launching nav2 backend...')
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        self.processes.append(process)
        
        return process
    
    def shutdown(self):
        """Kill all processes"""
        for proc in self.processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass


def main():
    rclpy.init()
    
    # Launch nav2 nodes in background
    launcher = Nav2Launcher()
    nav2_process = launcher.launch_nav2()
    
    # Wait a bit for nav2 to start
    print('Waiting for nav2 to initialize...')
    time.sleep(8.0)
    
    # Create explorer node
    explorer = FrontierExplorer()
    
    def signal_handler(sig, frame):
        """Handle shutdown signals"""
        print('\nShutting down...')
        explorer.shutdown()
        launcher.shutdown()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Spin the explorer
        print('Starting exploration...')
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    finally:
        explorer.shutdown()
        launcher.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

