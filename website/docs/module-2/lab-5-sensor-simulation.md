---
sidebar_position: 8
---

# Lab 5: Advanced Sensor Simulation

## Overview

This lab focuses on implementing and simulating advanced sensors for robotic systems in Gazebo. You will learn to configure various sensor types including LIDAR, IMU, GPS, and force/torque sensors, understand their simulation characteristics, and integrate them with ROS 2 for realistic perception and state estimation.

## Learning Objectives

After completing this lab, you will be able to:
- Configure and simulate various sensor types in Gazebo
- Understand the characteristics and limitations of simulated sensors
- Implement sensor fusion techniques for state estimation
- Analyze sensor noise and error models in simulation
- Integrate multiple sensors into a coherent perception system

## Prerequisites

- Completion of Lab 4: Basic Robot Model and Simulation
- Understanding of ROS 2 concepts including message types and topics
- Basic knowledge of coordinate frames and transformations
- Familiarity with Gazebo simulation from Module 2

## Hardware/Software Requirements

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo Fortress (Ignition)
- Basic text editor or IDE
- Minimum 8GB RAM recommended

## Lab Duration

Estimated completion time: 3-4 hours

## Part 1: Advanced Sensor Configuration

### Step 1: Project Setup

We'll extend the robot model from Lab 4 to include advanced sensors. First, let's create a new URDF file with additional sensors:

```xml
<!-- File: ~/robotics_ws/src/robot_description/urdf/advanced_robot.xacro -->
<?xml version="1.0"?>
<robot name="advanced_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include the base robot model -->
  <xacro:include filename="$(find robot_description)/urdf/robot.xacro"/>

  <!-- Properties for sensors -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- IMU Sensor -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- GPS Sensor -->
  <link name="gps_link">
    <visual>
      <geometry>
        <cylinder radius="0.01" length="0.01"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.01" length="0.01"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <joint name="gps_joint" type="fixed">
    <parent link="base_link"/>
    <child link="gps_link"/>
    <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- 360 LIDAR Sensor -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.02" length="0.03"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.02" length="0.03"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00008"/>
    </inertial>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <!-- Force/Torque Sensor -->
  <link name="ft_sensor_link">
    <visual>
      <geometry>
        <box size="0.03 0.03 0.01"/>
      </geometry>
      <material name="purple">
        <color rgba="0.8 0 0.8 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.03 0.03 0.01"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="ft_sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="ft_sensor_link"/>
    <origin xyz="-0.2 0 0.05" rpy="0 0 3.14159"/>
  </joint>

  <!-- Include advanced Gazebo configurations -->
  <xacro:include filename="$(find robot_description)/urdf/advanced_robot.gazebo.xacro"/>

</robot>
```

### Step 2: Advanced Gazebo Sensor Plugins

Create the Gazebo-specific configuration file for the advanced sensors:

```xml
<!-- File: ~/robotics_ws/src/robot_description/urdf/advanced_robot.gazebo.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include the base Gazebo configuration -->
  <xacro:include filename="$(find robot_description)/urdf/robot.gazebo.xacro"/>

  <!-- IMU Sensor Plugin -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.01</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- GPS Sensor Plugin -->
  <gazebo reference="gps_link">
    <sensor name="gps_sensor" type="navsat">
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <navsat>
        <position_sensing>
          <horizontal>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.01</stddev>
            </noise>
          </horizontal>
          <vertical>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.02</stddev>
            </noise>
          </vertical>
        </position_sensing>
      </navsat>
    </sensor>
  </gazebo>

  <!-- 360 LIDAR Sensor Plugin -->
  <gazebo reference="lidar_link">
    <sensor name="lidar_sensor" type="ray">
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>lidar</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_link</frame_name>
      </plugin>
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

  <!-- Force/Torque Sensor Plugin -->
  <gazebo reference="ft_sensor_link">
    <sensor name="ft_sensor" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>sensor</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
      <plugin name="ft_sensor_controller" filename="libgazebo_ros_ft_sensor.so">
        <ros>
          <namespace>ft_sensor</namespace>
          <remapping>~/wrench:=wrench</remapping>
        </ros>
        <frame_name>ft_sensor_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Color materials for sensors -->
  <gazebo reference="imu_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="gps_link">
    <material>Gazebo/Yellow</material>
  </gazebo>

  <gazebo reference="lidar_link">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="ft_sensor_link">
    <material>Gazebo/Purple</material>
  </gazebo>

</robot>
```

## Part 2: Sensor Fusion Implementation

### Step 3: Robot Localization with Sensor Fusion

Create a configuration file for the robot_localization package to fuse IMU and odometry data:

```yaml
# File: ~/robotics_ws/src/robot_description/config/robot_localization.yaml
# The frequency, in Hz, at which the filter will output a position estimate
frequency: 50

# The period, in seconds, after which we consider a sensor to have timed out
sensor_timeout: 0.1

# Whether to two-dimensionalize the robot's pose
two_d_mode: true

# Whether to print diagnostic messages
print_diagnostics: true

# Settings for each IMU input
imu0: /imu/data
imu0_config: [false, false, false,  # x, y, z
              false, false, false,  # roll, pitch, yaw
              true, true, true,    # x_vel, y_vel, z_vel
              false, false, false,  # roll_vel, pitch_vel, yaw_vel
              true, true, true]    # x_acc, y_acc, z_acc
imu0_differential: false
imu0_relative: true
imu0_queue_size: 10

# Settings for odometry input
odom0: /odom
odom0_config: [true, true, false,  # x, y, z
               false, false, true,  # roll, pitch, yaw
               false, false, false,  # x_vel, y_vel, z_vel
               false, false, false,  # roll_vel, pitch_vel, yaw_vel
               false, false, false]  # x_acc, y_acc, z_acc
odom0_differential: false
odom0_relative: true
odom0_queue_size: 10

# Process noise for the filter
process_noise_covariance: [0.05, 0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0.05, 0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0.06, 0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0.03, 0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0.03, 0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0.06, 0,     0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0.025, 0,     0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0.025, 0,    0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0.04, 0,    0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0.01, 0,    0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0.01, 0,    0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0.02, 0,    0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0.01, 0,    0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0.01, 0,
                          0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0.015]

# Initial estimate error covariance
initial_estimate_covariance: [1e-9, 0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    1e-9, 0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    1e-9, 0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    1e-9, 0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    1e-9, 0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    1e-9, 0,    0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    1e-9, 0,    0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    1e-9, 0,    0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    1e-9, 0,     0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    1e-9,  0,     0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     1e-9,  0,     0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     1e-9,  0,    0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     1e-9, 0,    0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    1e-9, 0,
                             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    1e-9]
```

### Step 4: Sensor Fusion Node Implementation

Create a Python node to demonstrate sensor fusion techniques:

```python
# File: ~/robotics_ws/src/robot_description/sensor_fusion/sensor_fusion_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, NavSatFix
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Initialize state variables
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.linear_acceleration = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Time tracking
        self.last_time = self.get_clock().now()

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, 10)

        # Publishers
        self.position_pub = self.create_publisher(
            Vector3, '/estimated_position', 10)
        self.velocity_pub = self.create_publisher(
            Vector3, '/estimated_velocity', 10)
        self.status_pub = self.create_publisher(
            Float64, '/sensor_fusion_status', 10)

        # Timer for publishing estimated state
        self.timer = self.create_timer(0.02, self.publish_state)  # 50 Hz

        self.get_logger().info('Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU data for state estimation"""
        # Extract linear acceleration (remove gravity)
        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Extract angular velocity
        ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Extract orientation
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Update internal state
        self.linear_acceleration = linear_acc
        self.angular_velocity = ang_vel
        self.orientation = orientation

        # Integrate acceleration to get velocity (simple integration)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt > 0:
            self.velocity += linear_acc * dt
            # Update position based on velocity
            self.position += self.velocity * dt

    def lidar_callback(self, msg):
        """Process LIDAR data for obstacle detection and mapping"""
        # Calculate distances to nearest obstacles
        ranges = np.array(msg.ranges)
        # Remove invalid measurements (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().debug(f'Min obstacle distance: {min_distance:.2f}m')

        # Calculate front, left, right distances
        if len(ranges) >= 360:
            front_idx = 0
            left_idx = 90
            right_idx = 270

            front_distance = ranges[front_idx] if np.isfinite(ranges[front_idx]) else float('inf')
            left_distance = ranges[left_idx] if np.isfinite(ranges[left_idx]) else float('inf')
            right_distance = ranges[right_idx] if np.isfinite(ranges[right_idx]) else float('inf')

            self.get_logger().debug(f'Front: {front_distance:.2f}, Left: {left_distance:.2f}, Right: {right_distance:.2f}')

    def gps_callback(self, msg):
        """Process GPS data for absolute position"""
        # For simulation, we'll use GPS as ground truth for position
        # In real applications, we would fuse GPS with other sensors
        self.position[0] = msg.longitude  # X position in local frame
        self.position[1] = msg.latitude   # Y position in local frame
        self.position[2] = msg.altitude   # Z position in local frame

    def publish_state(self):
        """Publish the estimated state"""
        # Create and publish position
        pos_msg = Vector3()
        pos_msg.x = float(self.position[0])
        pos_msg.y = float(self.position[1])
        pos_msg.z = float(self.position[2])
        self.position_pub.publish(pos_msg)

        # Create and publish velocity
        vel_msg = Vector3()
        vel_msg.x = float(self.velocity[0])
        vel_msg.y = float(self.velocity[1])
        vel_msg.z = float(self.velocity[2])
        self.velocity_pub.publish(vel_msg)

        # Publish fusion status (simple indicator)
        status_msg = Float64()
        status_msg.data = 1.0  # Indicate fusion is active
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Part 3: Launch Files for Sensor Simulation

### Step 5: Advanced Robot Launch File

Create a launch file for the advanced robot with all sensors:

```python
# File: ~/robotics_ws/src/robot_description/launch/advanced_robot.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='empty.sdf')

    # Paths
    pkg_gazebo_ros = FindPackageShare('ros_gz_sim')
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('robot_description'),
        'urdf',
        'advanced_robot.xacro'
    ])

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': ['-r', world]
        }.items()
    )

    # Read robot description
    with open(os.path.join(
        FindPackageShare('robot_description').perform({}),
        'urdf',
        'advanced_robot.xacro'
    ), 'r') as infp:
        robot_description = infp.read()

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description
        }]
    )

    # Spawn entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'advanced_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # Robot localization node
    robot_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('robot_description'),
                'config',
                'robot_localization.yaml'
            ])
        ]
    )

    # Sensor fusion node
    sensor_fusion = Node(
        package='robot_description',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from `/usr/share/gazebo/worlds`'),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        robot_localization,
        sensor_fusion
    ])
```

## Part 4: Testing and Analysis

### Step 6: Sensor Analysis Tools

Create a script to analyze sensor data quality:

```python
# File: ~/robotics_ws/src/robot_description/scripts/sensor_analysis.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, NavSatFix
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class SensorAnalysisNode(Node):
    def __init__(self):
        super().__init__('sensor_analysis_node')

        # Data storage for analysis
        self.imu_linear_acc_x = deque(maxlen=1000)
        self.imu_linear_acc_y = deque(maxlen=1000)
        self.imu_linear_acc_z = deque(maxlen=1000)

        self.lidar_ranges = deque(maxlen=100)

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # Publishers for analysis results
        self.noise_pub = self.create_publisher(
            Float64MultiArray, '/sensor_noise_analysis', 10)

        # Timer for analysis
        self.timer = self.create_timer(1.0, self.analyze_data)

        self.get_logger().info('Sensor Analysis Node initialized')

    def imu_callback(self, msg):
        """Store IMU data for analysis"""
        self.imu_linear_acc_x.append(msg.linear_acceleration.x)
        self.imu_linear_acc_y.append(msg.linear_acceleration.y)
        self.imu_linear_acc_z.append(msg.linear_acceleration.z)

    def lidar_callback(self, msg):
        """Store LIDAR data for analysis"""
        self.lidar_ranges.append(np.array(msg.ranges))

    def analyze_data(self):
        """Analyze collected sensor data"""
        if len(self.imu_linear_acc_x) > 10:
            # Calculate statistics for IMU data
            acc_x_mean = np.mean(self.imu_linear_acc_x)
            acc_x_std = np.std(self.imu_linear_acc_x)
            acc_y_mean = np.mean(self.imu_linear_acc_y)
            acc_y_std = np.std(self.imu_linear_acc_y)
            acc_z_mean = np.mean(self.imu_linear_acc_z)
            acc_z_std = np.std(self.imu_linear_acc_z)

            # Publish analysis results
            analysis_msg = Float64MultiArray()
            analysis_msg.data = [
                acc_x_mean, acc_x_std,  # X acceleration mean and std
                acc_y_mean, acc_y_std,  # Y acceleration mean and std
                acc_z_mean, acc_z_std   # Z acceleration mean and std
            ]
            self.noise_pub.publish(analysis_msg)

            self.get_logger().info(
                f'IMU Noise Analysis - X: μ={acc_x_mean:.4f}, σ={acc_x_std:.4f} | '
                f'Y: μ={acc_y_mean:.4f}, σ={acc_y_std:.4f} | '
                f'Z: μ={acc_z_mean:.4f}, σ={acc_z_std:.4f}'
            )

def main(args=None):
    rclpy.init(args=args)
    sensor_analysis_node = SensorAnalysisNode()

    try:
        rclpy.spin(sensor_analysis_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_analysis_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Building the Package

Update the CMakeLists.txt to include the new Python nodes:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  urdf
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS
  scripts/sensor_analysis.py
  sensor_fusion/sensor_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

And update package.xml to include new dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_description</name>
  <version>0.0.0</version>
  <description>Robot description package for simple robot model</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>urdf</depend>
  <depend>xacro</depend>
  <depend>robot_state_publisher</depend>
  <depend>ros_gz_sim</depend>
  <depend>robot_localization</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_msgs</depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Step 8: Building and Running the Simulation

Build the updated package:

```bash
cd ~/robotics_ws
colcon build --packages-select robot_description
source install/setup.bash
```

Launch the advanced robot simulation:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# Launch the robot with all sensors
ros2 launch robot_description advanced_robot.launch.py
```

In another terminal, check the sensor topics:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# List all topics
ros2 topic list

# Echo IMU data
ros2 topic echo /imu/data

# Echo LIDAR data
ros2 topic echo /scan

# Echo GPS data
ros2 topic echo /gps/fix
```

## Part 5: Lab Exercises

### Exercise 1: Sensor Noise Analysis
1. Run the sensor analysis node to observe noise characteristics
2. Compare the noise levels with the values defined in the URDF
3. Document the differences and explain potential causes

### Exercise 2: Sensor Fusion Implementation
1. Implement a simple Kalman filter to fuse IMU and odometry data
2. Compare the fused position estimate with individual sensor estimates
3. Analyze the improvement in accuracy

### Exercise 3: Multi-Sensor Mapping
1. Create a simple occupancy grid using LIDAR data
2. Overlay the robot's position (from fusion) on the grid
3. Visualize the resulting map in RViz

### Exercise 4: Sensor Failure Simulation
1. Modify the URDF to simulate sensor failures (e.g., set update_rate to 0)
2. Observe how the sensor fusion algorithm adapts
3. Implement a basic sensor health monitoring system

## Lab Assignment

### Task 1: Add a 3D LIDAR
- Add a 3D LIDAR sensor (Velodyne-style) to the robot
- Configure appropriate parameters for realistic simulation
- Visualize the 3D point cloud in RViz

### Task 2: Implement Adaptive Sensor Fusion
- Create a sensor fusion algorithm that adapts based on sensor reliability
- Implement a method to detect when individual sensors become unreliable
- Adjust the fusion algorithm to give less weight to unreliable sensors

### Task 3: Create a Custom Sensor Model
- Design a custom sensor model (e.g., thermal camera, ultrasonic array)
- Implement the sensor plugin for Gazebo
- Integrate the sensor into the robot model and simulation

## Summary

In this lab, you've learned to configure and simulate various types of advanced sensors in Gazebo, implemented sensor fusion techniques for state estimation, and analyzed sensor noise characteristics. You've gained experience with IMU, GPS, LIDAR, and force/torque sensors, and understand how to integrate them into a coherent perception system.

## Additional Resources

- [Gazebo Sensor Documentation](http://gazebosim.org/tutorials?tut=ros_gz_sensors)
- [Robot Localization Package](http://docs.ros.org/en/noetic/api/robot_localization/html/)
- [Sensor Messages in ROS 2](https://docs.ros.org/en/rolling/Releases/Release-Galactic-Geochelone.html)
- [ROS 2 Sensor Integration Guide](https://navigation.ros.org/setup_guides/index.html)

## Next Steps

After completing this lab, you should:
- Understand the characteristics of various robot sensors
- Be able to configure and simulate advanced sensors in Gazebo
- Have experience with sensor fusion techniques
- Be ready to proceed to Lab 6: Unity Integration with Simulation