#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
from collections import defaultdict
import time
import json
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

class StandoffDetector(Node):
    def __init__(self):
        super().__init__('standoff_detector')
        
        # 1 foot in meters
        self.max_distance = 0.3048  # 1 foot = 0.3048 meters
        self.recording_duration = 5.0  # 5 seconds
        self.start_time = None
        self.recording = True
        self.points = []  # List of (x, y) points
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Standoff detector started. Waiting for lidar data...')
        self.start_time = None
        self.first_scan_received = False
        
        # Set up a timer to process after recording
        self.process_timer = self.create_timer(0.1, self.check_and_process)
        self.processed = False

    def scan_callback(self, msg):
        # Start recording on first scan
        if not self.first_scan_received:
            self.first_scan_received = True
            self.start_time = time.time()
            self.get_logger().info('First lidar scan received. Recording lidar data for 5 seconds...')
        
        if not self.recording:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= self.recording_duration:
            self.recording = False
            self.get_logger().info(f'Recording complete. Collected {len(self.points)} points.')
            return
        
        # Convert polar to cartesian and filter by distance
        for i, range_val in enumerate(msg.ranges):
            # Skip invalid readings
            if math.isnan(range_val) or math.isinf(range_val) or range_val < msg.range_min or range_val > msg.range_max:
                continue
            
            # Filter points less than 1 foot
            if range_val < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                self.points.append((x, y))

    def check_and_process(self):
        # Only process if we've stopped recording and haven't processed yet
        if self.start_time is not None and not self.recording and not self.processed:
            self.processed = True
            if len(self.points) > 0:
                self.cluster_and_print()
            else:
                self.get_logger().warn('No points collected! Make sure lidar is working and standoffs are visible.')
            # Shutdown after processing
            self.destroy_node()
            rclpy.shutdown()

    def cluster_and_print(self):
        if len(self.points) == 0:
            self.get_logger().warn('No points collected!')
            return
        
        points_array = np.array(self.points)
        
        # Use simple k-means clustering with k=4
        # Initialize 4 cluster centers using farthest point sampling
        n_clusters = 4
        if len(points_array) < n_clusters:
            self.get_logger().warn(f'Not enough points ({len(points_array)}) for 4 clusters. Found points:')
            for i, (x, y) in enumerate(self.points):
                self.get_logger().info(f'  Point {i+1}: x={x:.4f}m, y={y:.4f}m, distance={math.sqrt(x*x+y*y):.4f}m')
            return
        
        # Initialize cluster centers using farthest point sampling
        centers = []
        # First center: point closest to origin
        distances_to_origin = np.linalg.norm(points_array, axis=1)
        first_idx = np.argmin(distances_to_origin)
        centers.append(points_array[first_idx])
        
        # Find 3 more centers that are farthest from existing centers
        used_indices = [first_idx]
        for _ in range(n_clusters - 1):
            max_min_dist = -1
            best_idx = -1
            for i, point in enumerate(points_array):
                if i in used_indices:
                    continue
                min_dist_to_centers = min([np.linalg.norm(point - c) for c in centers])
                if min_dist_to_centers > max_min_dist:
                    max_min_dist = min_dist_to_centers
                    best_idx = i
            if best_idx >= 0:
                centers.append(points_array[best_idx])
                used_indices.append(best_idx)
        
        # Simple k-means iteration
        for iteration in range(10):  # Max 10 iterations
            # Assign points to nearest center
            clusters = [[] for _ in range(n_clusters)]
            for point in points_array:
                distances = [np.linalg.norm(point - center) for center in centers]
                nearest_cluster = np.argmin(distances)
                clusters[nearest_cluster].append(point)
            
            # Update centers
            new_centers = []
            for i, cluster_points in enumerate(clusters):
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    # Keep old center if cluster is empty
                    new_centers.append(centers[i])
            
            # Check for convergence
            if all(np.linalg.norm(new_centers[i] - centers[i]) < 0.01 for i in range(n_clusters)):
                break
            centers = new_centers
        
        # Print the 4 cluster centers
        self.get_logger().info('=' * 60)
        self.get_logger().info('STANDOFF DETECTION RESULTS (4 points):')
        self.get_logger().info('=' * 60)
        for i, center in enumerate(centers):
            x, y = center[0], center[1]
            distance = math.sqrt(x*x + y*y)
            angle_deg = math.degrees(math.atan2(y, x))
            self.get_logger().info(f'Standoff {i+1}:')
            self.get_logger().info(f'  Position: x={x:.4f}m, y={y:.4f}m')
            self.get_logger().info(f'  Distance from lidar: {distance:.4f}m ({distance*3.28084:.4f} feet)')
            self.get_logger().info(f'  Angle: {angle_deg:.2f} degrees')
            self.get_logger().info('')
        self.get_logger().info('=' * 60)
        
        # Calculate bounding box with 2 inch margin
        self.calculate_and_save_bounding_box(centers)
    
    def calculate_and_save_bounding_box(self, standoff_centers):
        """Calculate bounding box from standoff points and save to config file"""
        # Extract x and y coordinates
        x_coords = [center[0] for center in standoff_centers]
        y_coords = [center[1] for center in standoff_centers]
        
        # Find min/max
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Add 2 inch margin (0.0508 meters)
        margin = 0.0508  # 2 inches in meters
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Save to config file
        package_share = get_package_share_directory('roverrobotics_driver')
        config_file = Path(package_share) / 'config' / 'standoff_bounding_box.json'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'standoff_bounding_box': {
                'x_min': float(x_min),
                'x_max': float(x_max),
                'y_min': float(y_min),
                'y_max': float(y_max),
                'margin_meters': float(margin)
            },
            'standoff_points': [
                {'x': float(center[0]), 'y': float(center[1])} 
                for center in standoff_centers
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('STANDOFF BOUNDING BOX (with 2 inch margin):')
        self.get_logger().info(f'  x_min: {x_min:.4f}m, x_max: {x_max:.4f}m')
        self.get_logger().info(f'  y_min: {y_min:.4f}m, y_max: {y_max:.4f}m')
        self.get_logger().info(f'  Saved to: {config_file}')
        self.get_logger().info('=' * 60)

def main(args=None):
    rclpy.init(args=args)
    node = StandoffDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

