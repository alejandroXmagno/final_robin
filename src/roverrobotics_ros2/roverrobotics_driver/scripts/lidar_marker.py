#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class LidarMarker(Node):
    def __init__(self):
        super().__init__('lidar_marker')
        
        # Create publisher for visualization markers
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        
        # Create a timer to publish the marker periodically
        self.timer = self.create_timer(0.1, self.publish_marker)  # 10 Hz
        
        self.get_logger().info('Lidar marker publisher started. Publishing sphere at lidar_link position.')
    
    def publish_marker(self):
        """Publish a sphere marker at the lidar_link position"""
        marker = Marker()
        marker.header.frame_id = "lidar_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lidar_position"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position (at origin of lidar_link frame, so 0,0,0)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Size of the sphere (in meters)
        marker.scale.x = 0.05  # 5cm radius
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Color (red with some transparency)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Slightly transparent
        
        # Marker lifetime (0 = never delete)
        marker.lifetime.sec = 0
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    lidar_marker = LidarMarker()
    
    try:
        rclpy.spin(lidar_marker)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        try:
            lidar_marker.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()


