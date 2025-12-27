---
sidebar_position: 7
---

# Lab 6: Unity Integration with Simulation
## Overview
In this lab, you will integrate Unity with your ROS 2 simulation environment to create advanced visualization and Human-Robot Interaction capabilities. This lab demonstrates how to connect Unity's powerful graphics engine with Gazebo's physics simulation through ROS 2, creating a comprehensive Digital Twin system with intuitive user interfaces.

## Learning Objectives
After completing this lab, you will be able to:
- Set up Unity-ROS 2 integration using the Unity Robotics Hub
- Create synchronized visualization between Gazebo simulation and Unity
- Implement Human-Robot Interaction interfaces in Unity
- Design intuitive user interfaces for robot teleoperation
- Validate simulation-visualization synchronization

## Prerequisites
- Completion of Lab 4 and Lab 5
- Unity 2021.3 LTS or later installed
- Unity Robotics Hub and ROS-TCP-Connector packages
- Gazebo simulation environment from previous labs
- ROS 2 Humble Hawksbill with required interfaces
- Understanding of Unity C# scripting

## Lab Setup
### 1. Unity Environment Setup
First, ensure you have the required Unity packages installed:

```bash
# In your Unity project, install the following packages via Package Manager:
# - Unity Robotics Hub
# - ROS-TCP-Connector
# - XR Interaction Toolkit (optional, for advanced interfaces)
```

### 2. ROS 2 Bridge Configuration
Create a ROS 2 bridge package for Unity integration:

```bash
# Navigate to your workspace
cd ~/ros2_labs/src

# Create the Unity integration package
ros2 pkg create --build-type ament_python unity_integration --dependencies rclpy std_msgs geometry_msgs sensor_msgs visualization_msgs
```

## Unity Integration Architecture
The Unity integration system will consist of:

1. **Unity Visualizer Node**: Receives robot state and renders in Unity
2. **Unity Controller Node**: Processes user input from Unity UI
3. **Synchronization Manager**: Keeps Unity and Gazebo states aligned
4. **User Interface System**: Provides teleoperation controls

### Unity Visualizer Node Implementation
Create the Unity visualizer node in `unity_integration/unity_integration/visualizer_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import socket
import json
import threading


class UnityVisualizerNode(Node):
    def __init__(self):
        super().__init__('unity_visualizer_node')

        # ROS 2 subscribers for robot state
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.laser_subscriber = self.create_subscription(
            String, '/unity_commands', self.unity_command_callback, 10)

        # TCP connection to Unity
        self.unity_socket = None
        self.connect_to_unity()

        # Timer for sending state updates to Unity
        self.update_timer = self.create_timer(0.033, self.send_state_to_unity)  # ~30 Hz

        # Robot state storage
        self.robot_pose = Pose()
        self.joint_positions = {}
        self.get_logger().info('Unity Visualizer Node initialized')

    def connect_to_unity(self):
        """Establish connection to Unity application"""
        try:
            self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.unity_socket.connect(('localhost', 10000))
            self.get_logger().info('Connected to Unity')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Unity: {e}')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        """Update joint positions from joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def unity_command_callback(self, msg):
        """Handle commands from Unity"""
        try:
            command = json.loads(msg.data)
            self.get_logger().info(f'Received Unity command: {command}')
            # Process command (e.g., send to robot controller)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON command from Unity')

    def send_state_to_unity(self):
        """Send robot state to Unity for visualization"""
        if self.unity_socket:
            try:
                state_data = {
                    'timestamp': self.get_clock().now().nanoseconds,
                    'robot_pose': {
                        'x': self.robot_pose.position.x,
                        'y': self.robot_pose.position.y,
                        'z': self.robot_pose.position.z,
                        'qx': self.robot_pose.orientation.x,
                        'qy': self.robot_pose.orientation.y,
                        'qz': self.robot_pose.orientation.z,
                        'qw': self.robot_pose.orientation.w
                    },
                    'joint_positions': self.joint_positions
                }

                serialized_data = json.dumps(state_data)
                self.unity_socket.send(serialized_data.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'Failed to send data to Unity: {e}')

    def destroy_node(self):
        """Clean up socket connection"""
        if self.unity_socket:
            self.unity_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UnityVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Unity Controller Node Implementation
Create the Unity controller node in `unity_integration/unity_integration/controller_node.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json


class UnityControllerNode(Node):
    def __init__(self):
        super().__init__('unity_controller_node')

        # Publisher for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for Unity control commands
        self.unity_control_subscriber = self.create_subscription(
            String, '/unity_control_commands', self.control_command_callback, 10)

        self.get_logger().info('Unity Controller Node initialized')

    def control_command_callback(self, msg):
        """Process control commands from Unity"""
        try:
            command = json.loads(msg.data)

            if command['type'] == 'teleop':
                twist_msg = Twist()
                twist_msg.linear.x = command.get('linear_x', 0.0)
                twist_msg.linear.y = command.get('linear_y', 0.0)
                twist_msg.linear.z = command.get('linear_z', 0.0)
                twist_msg.angular.x = command.get('angular_x', 0.0)
                twist_msg.angular.y = command.get('angular_y', 0.0)
                twist_msg.angular.z = command.get('angular_z', 0.0)

                self.cmd_vel_publisher.publish(twist_msg)
                self.get_logger().info(f'Sent command: linear={twist_msg.linear}, angular={twist_msg.angular}')

            elif command['type'] == 'navigation':
                # Handle navigation goal commands
                self.handle_navigation_command(command)

            elif command['type'] == 'manipulation':
                # Handle manipulation commands
                self.handle_manipulation_command(command)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON command from Unity')
        except Exception as e:
            self.get_logger().error(f'Error processing Unity command: {e}')

    def handle_navigation_command(self, command):
        """Handle navigation commands from Unity"""
        # Implementation would send navigation goals
        self.get_logger().info(f'Navigation command: {command}')

    def handle_manipulation_command(self, command):
        """Handle manipulation commands from Unity"""
        # Implementation would send manipulation goals
        self.get_logger().info(f'Manipulation command: {command}')


def main(args=None):
    rclpy.init(args=args)
    node = UnityControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Unity Scene Implementation
### 1. Robot Model Setup in Unity
Create a Unity C# script to handle robot visualization:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Newtonsoft.Json;

public class RobotVisualizer : MonoBehaviour
{
    [SerializeField] private GameObject robotBase;
    [SerializeField] private Dictionary<string, GameObject> jointObjects;
    [SerializeField] private ROSConnection ros;

    private string unityTopic = "unity_commands";
    private string controlTopic = "unity_control_commands";

    void Start()
    {
        ros = ROSConnection.instance;
        jointObjects = new Dictionary<string, GameObject>();

        // Initialize joint objects
        FindJointObjects();

        // Start listening for robot state updates
        StartCoroutine(ReceiveRobotState());
    }

    void FindJointObjects()
    {
        // Find all joint objects in the robot hierarchy
        Transform[] allChildren = robotBase.GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name.Contains("joint") || child.name.Contains("Joint"))
            {
                jointObjects[child.name] = child.gameObject;
            }
        }
    }

    IEnumerator ReceiveRobotState()
    {
        // This would receive state from a custom ROS topic or service
        // For this lab, we'll simulate receiving state updates
        while (true)
        {
            // In a real implementation, this would receive actual robot state
            yield return new WaitForSeconds(0.033f); // ~30 Hz
        }
    }

    public void SendCommandToRobot(string commandType, Dictionary<string, float> parameters)
    {
        var command = new Dictionary<string, object>
        {
            {"type", commandType},
            {"parameters", parameters},
            {"timestamp", Time.time}
        };

        string jsonString = JsonConvert.SerializeObject(command);
        ros.Send(unityTopic, new StringMsg { data = jsonString });
    }

    public void TeleopRobot(float linearX, float angularZ)
    {
        var teleopCommand = new Dictionary<string, float>
        {
            {"linear_x", linearX},
            {"angular_z", angularZ}
        };

        SendCommandToRobot("teleop", teleopCommand);
    }
}
```

### 2. Unity User Interface
Create a Unity C# script for the user interface:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;

public class UnityRobotInterface : MonoBehaviour
{
    [SerializeField] private RobotVisualizer robotVisualizer;
    [SerializeField] private Slider linearSpeedSlider;
    [SerializeField] private Slider angularSpeedSlider;
    [SerializeField] private Button forwardButton;
    [SerializeField] private Button backwardButton;
    [SerializeField] private Button leftButton;
    [SerializeField] private Button rightButton;
    [SerializeField] private Button stopButton;

    private float linearSpeed = 0.5f;
    private float angularSpeed = 0.5f;

    void Start()
    {
        SetupUIEvents();
    }

    void SetupUIEvents()
    {
        linearSpeedSlider.onValueChanged.AddListener(OnLinearSpeedChanged);
        angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);

        forwardButton.onClick.AddListener(() => MoveRobot(1, 0));
        backwardButton.onClick.AddListener(() => MoveRobot(-1, 0));
        leftButton.onClick.AddListener(() => MoveRobot(0, 1));
        rightButton.onClick.AddListener(() => MoveRobot(0, -1));
        stopButton.onClick.AddListener(() => MoveRobot(0, 0));
    }

    void OnLinearSpeedChanged(float value)
    {
        linearSpeed = value;
    }

    void OnAngularSpeedChanged(float value)
    {
        angularSpeed = value;
    }

    void MoveRobot(float linearDirection, float angularDirection)
    {
        float linearX = linearDirection * linearSpeed;
        float angularZ = angularDirection * angularSpeed;

        robotVisualizer.TeleopRobot(linearX, angularZ);
    }

    void Update()
    {
        // Handle keyboard input for teleoperation
        float linear = 0, angular = 0;

        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
            linear = 1;
        else if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            linear = -1;

        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            angular = 1;
        else if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            angular = -1;

        if (linear != 0 || angular != 0)
        {
            robotVisualizer.TeleopRobot(linear * linearSpeed, angular * angularSpeed);
        }
    }
}
```

## Integration Testing
### 1. Setup Integration Test Environment
Create a test script to verify Unity-ROS integration:

```python
#!/usr/bin/env python3
# test_unity_integration.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time
import json


class UnityIntegrationTester(Node):
    def __init__(self):
        super().__init__('unity_integration_tester')

        # Publisher for test commands
        self.test_publisher = self.create_publisher(String, '/unity_control_commands', 10)

        # Timer for test sequence
        self.test_timer = self.create_timer(2.0, self.run_test_sequence)
        self.test_step = 0

        self.get_logger().info('Unity Integration Tester initialized')

    def run_test_sequence(self):
        """Run a sequence of tests to verify Unity integration"""
        if self.test_step == 0:
            self.get_logger().info('Test 1: Sending teleoperation command')
            command = {
                'type': 'teleop',
                'linear_x': 0.5,
                'angular_z': 0.0
            }
            self.send_command(command)

        elif self.test_step == 1:
            self.get_logger().info('Test 2: Sending rotation command')
            command = {
                'type': 'teleop',
                'linear_x': 0.0,
                'angular_z': 0.5
            }
            self.send_command(command)

        elif self.test_step == 2:
            self.get_logger().info('Test 3: Stopping robot')
            command = {
                'type': 'teleop',
                'linear_x': 0.0,
                'angular_z': 0.0
            }
            self.send_command(command)

        elif self.test_step == 3:
            self.get_logger().info('Test 4: Sending navigation command')
            command = {
                'type': 'navigation',
                'target_x': 1.0,
                'target_y': 1.0,
                'target_theta': 0.0
            }
            self.send_command(command)

        self.test_step += 1
        if self.test_step > 3:
            self.test_timer.cancel()
            self.get_logger().info('All tests completed')

    def send_command(self, command):
        """Send command to Unity controller"""
        msg = String()
        msg.data = json.dumps(command)
        self.test_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    tester = UnityIntegrationTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Running the Unity Integration System
### 1. Launch Gazebo Simulation
```bash
# Terminal 1: Start Gazebo simulation
ros2 launch your_robot_gazebo your_robot_world.launch.py
```

### 2. Launch ROS 2 Bridge Nodes
```bash
# Terminal 2: Start Unity integration nodes
cd ~/ros2_labs
source install/setup.bash
ros2 run unity_integration visualizer_node
```

```bash
# Terminal 3: Start Unity controller node
cd ~/ros2_labs
source install/setup.bash
ros2 run unity_integration controller_node
```

### 3. Run Unity Application
1. Open your Unity project
2. Load the robot visualization scene
3. Configure the ROS TCP Connector to connect to localhost:10000
4. Press Play to start the Unity application

### 4. Test the Integration
```bash
# Terminal 4: Run integration tests
cd ~/ros2_labs
source install/setup.bash
python3 test_unity_integration.py
```

## Lab Exercises
### Exercise 1: Enhanced Visualization1. Add sensor visualization (LIDAR rays, camera feed) to the Unity scene
2. Implement real-time sensor data visualization
3. Add visual effects for robot status (battery level, system health)

### Exercise 2: Advanced User Interface1. Create a joystick interface for more intuitive teleoperation
2. Add a map view showing robot position and navigation goals
3. Implement a command history and playback system

### Exercise 3: Multi-Robot Visualization1. Extend the system to visualize multiple robots simultaneously
2. Implement robot identification and labeling
3. Add inter-robot communication visualization

### Exercise 4: Mixed Reality Integration1. Research and implement AR/VR integration with the system
2. Create immersive teleoperation interfaces
3. Implement gesture-based controls

## Troubleshooting Common Issues
### Connection Issues- **Problem**: Unity cannot connect to ROS 2
- **Solution**: Verify ROS TCP Connector settings, check firewall settings, ensure both systems are on the same network

### Synchronization Issues- **Problem**: Unity visualization lags behind Gazebo simulation
- **Solution**: Increase update frequency, optimize data serialization, check network latency

### Performance Issues- **Problem**: Low frame rate in Unity application
- **Solution**: Reduce visualization complexity, optimize robot models, use Level of Detail (LOD) systems

## Best Practices
### 1. Data Optimization- Minimize data transmission between ROS 2 and Unity
- Use efficient serialization formats
- Implement data compression for large datasets

### 2. Visualization Quality- Match visual fidelity to physics fidelity
- Use consistent coordinate systems
- Implement proper scaling and units

### 3. User Experience- Provide intuitive control interfaces
- Include visual feedback for all actions
- Implement error handling and graceful degradation

## Summary
This lab demonstrated the integration of Unity with ROS 2 simulation systems, creating a powerful Digital Twin environment with advanced visualization and user interaction capabilities. You've learned to:

1. **Set up Unity-ROS integration**: Configure the communication bridge between Unity and ROS 2
2. **Create synchronized visualization**: Ensure Unity visualization matches Gazebo physics simulation
3. **Implement user interfaces**: Design intuitive interfaces for robot teleoperation
4. **Validate integration**: Test and verify the Unity-ROS connection

The Unity integration provides a professional-grade visualization and interaction layer that enhances the development and debugging experience for Robotic Systems, making complex robot behaviors more accessible and understandable.