#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class PosePrinter(Node):
    def __init__(self):
        super().__init__('pose_printer')
        
        # Subscribe to odometry topic
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/wheels',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('Pose printer started. Monitoring robot position and orientation...')
        self.get_logger().info('Waiting for odometry data...')
    
    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to yaw angle in radians"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw, pitch, roll
    
    def odom_callback(self, msg):
        """Callback function that prints position and orientation"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Extract orientation (quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw, pitch, roll = self.quaternion_to_yaw(qx, qy, qz, qw)
        
        # Convert yaw from radians to degrees
        yaw_deg = math.degrees(yaw)
        
        # Extract linear and angular velocities
        linear_x = msg.twist.twist.linear.x
        angular_z = msg.twist.twist.angular.z
        
        # Print formatted output
        print(f"\rPosition: x={x:7.3f}m, y={y:7.3f}m, z={z:7.3f}m | "
              f"Orientation: yaw={yaw_deg:7.2f}° ({yaw:7.4f} rad) | "
              f"Velocity: linear={linear_x:6.3f}m/s, angular={angular_z:6.3f}rad/s", end='', flush=True)

def main(args=None):
    rclpy.init(args=args)
    pose_printer = PosePrinter()
    
    try:
        rclpy.spin(pose_printer)
    except KeyboardInterrupt:
        print("\n")  # New line after Ctrl+C
        pass
    finally:
        pose_printer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

