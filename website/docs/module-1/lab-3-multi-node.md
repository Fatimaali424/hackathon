---
sidebar_position: 7
---

# Lab 3: Multi-node System Integration

## Overview

In this lab, you will implement a complex multi-node ROS 2 system that demonstrates how different nodes work together to achieve sophisticated robotic behaviors. This lab integrates concepts from previous labs (publishers/subscribers, services, and actions) into a coordinated system that simulates a simple robot performing navigation and manipulation tasks.

## Learning Objectives

After completing this lab, you will be able to:
- Design and implement a multi-node ROS 2 system
- Coordinate communication between multiple nodes
- Handle complex task execution using services and actions
- Debug multi-node system interactions
- Implement system-level error handling and recovery

## Prerequisites

- Completion of Lab 1 (Publisher/Subscriber) and Lab 2 (Services/Actions)
- ROS 2 Humble Hawksbill installed
- Basic knowledge of Python or C++
- Understanding of ROS 2 communication patterns

## Lab Setup

Create a new ROS 2 package for this multi-node system:

```bash
# Navigate to your workspace
cd ~/ros2_labs/src

# Create the multi-node lab package
ros2 pkg create --build-type ament_python ros2_lab3 --dependencies rclpy std_msgs geometry_msgs sensor_msgs action_msgs example_interfaces
```

## System Architecture

The multi-node system will consist of the following nodes:

1. **Navigator Node**: Handles navigation tasks using action-based movement
2. **Perceptor Node**: Processes sensor data and detects objects
3. **Manipulator Node**: Controls robotic arm movements
4. **Coordinator Node**: Orchestrates the overall task execution
5. **Task Planner Node**: Plans sequences of actions based on goals

### Service Definition

First, let's create a custom service definition for object detection. Create `ros2_lab3/srv/ObjectDetection.srv`:

```
string sensor_source
float32 confidence_threshold
string[] object_classes
---
bool success
string message
Object[] objects
float32 detection_time

# Nested message definition
msg Object:
  string class
  float32 confidence
  float64[4] bbox  # x, y, width, height
  float64[3] center_3d  # x, y, z
```

### Coordinator Node Implementation

Create the coordinator node in `ros2_lab3/ros2_lab3/coordinator_node.py`:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from example_interfaces.action import Fibonacci
from example_interfaces.srv import AddTwoInts  # Using example service for demonstration


class TaskCoordinator(Node):

    def __init__(self):
        super().__init__('task_coordinator')

        # Publishers for task status
        self.status_publisher = self.create_publisher(String, 'task_status', 10)

        # Service client for object detection (using example service for this lab)
        self.object_detection_client = self.create_client(
            AddTwoInts, 'object_detection_service')

        # Action client for navigation
        self.nav_action_client = ActionClient(
            self, Fibonacci, 'navigate_to_location')

        # Timer for task orchestration
        self.task_timer = self.create_timer(1.0, self.task_execution_callback)

        self.current_task = "idle"
        self.task_sequence = [
            "navigate_to_object",
            "detect_object",
            "manipulate_object",
            "return_to_base"
        ]
        self.task_index = 0

    def task_execution_callback(self):
        """Execute tasks in sequence"""
        if self.current_task == "idle" and self.task_index < len(self.task_sequence):
            task = self.task_sequence[self.task_index]
            self.get_logger().info(f'Executing task: {task}')

            if task == "navigate_to_object":
                self.execute_navigation_task()
            elif task == "detect_object":
                self.execute_detection_task()
            elif task == "manipulate_object":
                self.execute_manipulation_task()
            elif task == "return_to_base":
                self.execute_return_task()

            self.task_index += 1
        elif self.task_index >= len(self.task_sequence):
            self.get_logger().info('All tasks completed')
            self.current_task = "completed"
            self.publish_status("completed")

    def execute_navigation_task(self):
        """Execute navigation task using action client"""
        self.current_task = "navigating"
        self.publish_status("navigating to object location")

        # Send navigation goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = 5  # Simplified navigation goal

        self.nav_action_client.wait_for_server()
        future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback)

        future.add_done_callback(self.navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        self.get_logger().info(f'Navigation progress: {feedback_msg.feedback.sequence}')

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Navigation goal accepted')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """Handle navigation completion"""
        result = future.result().result
        self.get_logger().info(f'Navigation completed: {result.sequence}')
        self.current_task = "idle"
        self.publish_status("navigation completed")

    def execute_detection_task(self):
        """Execute object detection task using service client"""
        self.current_task = "detecting"
        self.publish_status("detecting objects")

        while not self.object_detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Object detection service not available, waiting...')

        # Using AddTwoInts service as an example - in a real system, you'd use a custom object detection service
        request = AddTwoInts.Request()
        request.a = 10
        request.b = 20

        future = self.object_detection_client.call_async(request)
        future.add_done_callback(self.detection_result_callback)

    def detection_result_callback(self, future):
        """Handle detection result"""
        try:
            response = future.result()
            self.get_logger().info(f'Detection result: {response.sum}')
            self.current_task = "idle"
            self.publish_status("detection completed")
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            self.current_task = "idle"

    def execute_manipulation_task(self):
        """Execute manipulation task"""
        self.current_task = "manipulating"
        self.publish_status("manipulating object")
        # In a real system, this would connect to a manipulation action server
        # For this lab, we'll simulate the completion after a delay
        self.create_timer(2.0, self.manipulation_complete_callback)

    def manipulation_complete_callback(self):
        """Handle manipulation completion"""
        self.get_logger().info('Manipulation completed')
        self.current_task = "idle"
        self.publish_status("manipulation completed")

    def execute_return_task(self):
        """Execute return to base task"""
        self.current_task = "returning"
        self.publish_status("returning to base")
        # In a real system, this would connect to navigation action server
        # For this lab, we'll simulate the completion after a delay
        self.create_timer(2.0, self.return_complete_callback)

    def return_complete_callback(self):
        """Handle return completion"""
        self.get_logger().info('Return to base completed')
        self.current_task = "idle"
        self.publish_status("return completed")

    def publish_status(self, status):
        """Publish task status"""
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    coordinator = TaskCoordinator()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Navigator Node Implementation

Create the navigator node in `ros2_lab3/ros2_lab3/navigator_node.py`:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class NavigatorNode(Node):

    def __init__(self):
        super().__init__('navigator_node')

        # Action server for navigation
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'navigate_to_location',
            self.execute_navigation_callback)

        # Publisher for movement commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscriber for odometry
        self.odom_subscriber = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        self.current_pose = None
        self.get_logger().info('Navigator node initialized')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def execute_navigation_callback(self, goal_handle):
        """Execute navigation action"""
        self.get_logger().info('Executing navigation goal...')

        # Create feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Simulate navigation by sending movement commands
        twist_msg = Twist()
        twist_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        twist_msg.angular.z = 0.0  # No rotation

        # Send movement commands for the duration of the goal
        for i in range(goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation goal canceled')
                result_msg.sequence = feedback_msg.sequence
                return result_msg

            # Publish movement command
            self.cmd_vel_publisher.publish(twist_msg)

            # Update feedback
            feedback_msg.sequence.append(i)
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to simulate movement
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1.0))

        # Stop the robot
        twist_msg.linear.x = 0.0
        self.cmd_vel_publisher.publish(twist_msg)

        # Complete the goal
        goal_handle.succeed()
        result_msg.sequence = feedback_msg.sequence
        self.get_logger().info(f'Navigation completed: {result_msg.sequence}')

        return result_msg


def main(args=None):
    rclpy.init(args=args)
    navigator = NavigatorNode()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Perceptor Node Implementation

Create the perceptor node in `ros2_lab3/ros2_lab3/perceptor_node.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts  # Using example service for demonstration


class PerceptorNode(Node):

    def __init__(self):
        super().__init__('perceptor_node')

        # Publishers for detected objects
        self.object_publisher = self.create_publisher(String, 'detected_objects', 10)

        # Subscribers for sensor data
        self.image_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Service server for object detection requests
        self.detection_service = self.create_service(
            AddTwoInts, 'object_detection_service', self.detection_service_callback)

        self.get_logger().info('Perceptor node initialized')

    def image_callback(self, msg):
        """Process image data and detect objects"""
        # Simulate object detection from image
        # In a real system, this would use computer vision algorithms
        if len(msg.data) > 0:  # If image has data
            # Simulate detection of an object
            detected_obj_msg = String()
            detected_obj_msg.data = f"object_detected_at_{self.get_clock().now().nanoseconds}"
            self.object_publisher.publish(detected_obj_msg)
            self.get_logger().info(f'Detected object: {detected_obj_msg.data}')

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Simulate obstacle detection from laser scan
        if len(msg.ranges) > 0:
            min_range = min(msg.ranges)
            if min_range < 1.0:  # If obstacle is within 1 meter
                self.get_logger().info(f'Obstacle detected at {min_range:.2f}m')

    def detection_service_callback(self, request, response):
        """Handle object detection service requests"""
        self.get_logger().info(f'Detection service called with: {request.a}, {request.b}')

        # Simulate detection process
        # In a real system, this would process sensor data and detect objects
        response.sum = request.a + request.b  # Simple sum for demonstration

        self.get_logger().info(f'Detection service returning: {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)
    perceptor = PerceptorNode()

    try:
        rclpy.spin(perceptor)
    except KeyboardInterrupt:
        pass
    finally:
        perceptor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Setup for Execution

Update the `setup.py` file to make the nodes executable:

```python
from setuptools import find_packages, setup

package_name = 'ros2_lab3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 multi-node system integration lab',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coordinator = ros2_lab3.coordinator_node:main',
            'navigator = ros2_lab3.navigator_node:main',
            'perceptor = ros2_lab3.perceptor_node:main',
        ],
    },
)
```

## Running the Multi-Node System

### 1. Build the Package

```bash
# Navigate to your workspace
cd ~/ros2_labs

# Build the package
colcon build --packages-select ros2_lab3

# Source the workspace
source install/setup.bash
```

### 2. Run the Nodes

Run each node in a separate terminal:

**Terminal 1 - Perceptor Node:**
```bash
ros2 run ros2_lab3 perceptor
```

**Terminal 2 - Navigator Node:**
```bash
ros2 run ros2_lab3 navigator
```

**Terminal 3 - Coordinator Node:**
```bash
ros2 run ros2_lab3 coordinator
```

### 3. Monitor the System

Monitor the communication between nodes:

```bash
# List all active topics
ros2 topic list

# Monitor task status
ros2 topic echo /task_status std_msgs/msg/String

# Monitor detected objects
ros2 topic echo /detected_objects std_msgs/msg/String

# Monitor navigation feedback
ros2 action list
ros2 action info /navigate_to_location
```

## Multi-Node Communication Patterns

### Publisher-Subscriber Pattern Across Nodes

The perceptor node publishes detected objects that can be subscribed to by other nodes:

```python
# Example of another node subscribing to detected objects
class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String, 'detected_objects', self.object_callback, 10)

    def object_callback(self, msg):
        self.get_logger().info(f'Received detected object: {msg.data}')
```

### Service-Based Communication

Nodes can communicate through services for synchronous requests:

```python
# Example service client in another node
class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client_node')
        self.cli = self.create_client(AddTwoInts, 'object_detection_service')

    def call_detection_service(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.cli.call_async(request)
        return future
```

### Action-Based Communication

For long-running tasks, nodes use actions:

```python
# Example action client in another node
class ActionClientNode(Node):
    def __init__(self):
        super().__init__('action_client_node')
        self._action_client = ActionClient(self, Fibonacci, 'navigate_to_location')

    def send_navigation_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        send_goal_future.add_done_callback(self.goal_response_callback)
        return send_goal_future
```

## Lab Exercises

### Exercise 1: Add Error Handling
1. Implement error handling in the coordinator node when services or actions are not available
2. Add retry logic for failed communications
3. Implement graceful degradation when nodes fail

### Exercise 2: Extend the System
1. Add a new node for battery monitoring that publishes battery status
2. Modify the coordinator to check battery status before executing tasks
3. Implement task prioritization based on battery level

### Exercise 3: Add Logging and Diagnostics
1. Add comprehensive logging to track system state
2. Implement a diagnostics node that monitors all other nodes
3. Create a dashboard that visualizes the system status

### Exercise 4: Performance Optimization
1. Analyze communication patterns and identify bottlenecks
2. Optimize message frequency and QoS settings
3. Implement message compression for large data (images, point clouds)

## Debugging Multi-Node Systems

### Common Issues and Solutions

1. **Node Discovery Issues**
   - Check that all nodes are on the same ROS domain
   - Verify that the ROS_MASTER_URI is properly set
   - Ensure network configuration allows node discovery

2. **Communication Delays**
   - Check QoS settings for appropriate reliability
   - Verify network bandwidth for high-frequency topics
   - Consider using intra-process communication where possible

3. **Resource Conflicts**
   - Ensure nodes have unique names
   - Check for topic name collisions
   - Verify action and service names are unique

### Debugging Commands

```bash
# Check node connectivity
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_py listener

# Monitor system resources
ros2 run top ros2_top

# Check message statistics
ros2 topic hz /topic_name

# Monitor system status
ros2 doctor
```

## Summary

In this lab, you've implemented a complex multi-node ROS 2 system that demonstrates:

1. **System Architecture**: Designing multiple nodes with specific responsibilities
2. **Communication Patterns**: Using topics, services, and actions for inter-node communication
3. **Task Coordination**: Orchestrating complex behaviors across multiple nodes
4. **Error Handling**: Implementing robust error handling in distributed systems
5. **System Integration**: Combining different ROS 2 concepts into a cohesive system

This multi-node approach is essential for real-world robotic systems where different components need to work together to achieve complex goals. The coordinator pattern demonstrated here is commonly used in robotics for task planning and execution management.