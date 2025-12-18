#!/usr/bin/env python3
"""
Basic Robot Simulation Example for Gazebo

This script demonstrates fundamental concepts for controlling a simulated robot in Gazebo.
It includes topics for sensor data, services for robot control, and basic navigation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import math


class BasicRobotController(Node):
    """
    Basic robot controller for Gazebo simulation
    Demonstrates ROS 2 communication with simulated sensors and actuators
    """

    def __init__(self):
        super().__init__('basic_robot_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Services
        self.enable_robot_srv = self.create_service(
            SetBool,
            '/enable_robot',
            self.enable_robot_callback
        )

        # Robot state
        self.robot_enabled = False
        self.current_pose = None
        self.laser_data = None
        self.obstacle_distance = float('inf')

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Basic Robot Controller initialized')

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        # Get the minimum distance from the front-facing range (±30 degrees)
        front_range_start = int(len(msg.ranges) * 0.45)  # 45% into the scan
        front_range_end = int(len(msg.ranges) * 0.55)    # 55% into the scan

        front_ranges = [
            r for r in msg.ranges[front_range_start:front_range_end]
            if not math.isnan(r) and r > 0
        ]

        if front_ranges:
            self.obstacle_distance = min(front_ranges)
        else:
            self.obstacle_distance = float('inf')

        self.laser_data = msg

    def odom_callback(self, msg):
        """Process odometry data to track robot position"""
        self.current_pose = msg.pose.pose

    def image_callback(self, msg):
        """Process camera image data (placeholder for image processing)"""
        # In a real implementation, you would process the image data
        # For this example, we just acknowledge receiving the image
        pass

    def enable_robot_callback(self, request, response):
        """Service callback to enable/disable robot movement"""
        self.robot_enabled = request.data
        response.success = True
        response.message = f"Robot {'enabled' if self.robot_enabled else 'disabled'}"
        self.get_logger().info(response.message)
        return response

    def control_loop(self):
        """Main control loop for robot behavior"""
        if not self.robot_enabled:
            # Stop the robot if disabled
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            return

        # Simple obstacle avoidance behavior
        cmd_vel = Twist()

        if self.obstacle_distance < 1.0:  # Obstacle within 1 meter
            # Turn right to avoid obstacle
            cmd_vel.linear.x = 0.2  # Move forward slowly
            cmd_vel.angular.z = -0.5  # Turn right
        else:
            # Move forward
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)

    robot_controller = BasicRobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down Basic Robot Controller')
    finally:
        # Cleanup
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()