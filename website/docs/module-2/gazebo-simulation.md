---
sidebar_position: 4
---

# Gazebo Simulation & Physics Modeling

## Overview

Gazebo is a powerful 3D simulation environment that enables the development, testing, and validation of robotic systems in a virtual environment before deployment to real hardware. This chapter covers the fundamentals of physics simulation, modeling, and creating realistic virtual environments for robotic applications.

## Learning Objectives

After completing this chapter, you will be able to:
- Set up and configure Gazebo simulation environments
- Create and import robot models with accurate physics properties
- Configure sensors and actuators in simulation
- Implement realistic physics parameters for accurate simulation
- Understand the differences between simulation and real-world behavior

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. While Gazebo is first and foremost a simulator, it provides several key features that make it particularly valuable for robotics development:

- **Physics engines**: Support for ODE, Bullet, Simbody, and DART physics engines
- **Sensor simulation**: Support for various sensors including cameras, LIDAR, IMU, and more
- **Realistic rendering**: High-quality graphics rendering for visual perception tasks
- **ROS integration**: Seamless integration with ROS and ROS 2 through Gazebo ROS packages
- **Extensible architecture**: Plugin system for custom sensors, controllers, and environments

### Gazebo vs. Gazebo Classic

With the transition to Gazebo Fortress (also known as Ignition Gazebo), the simulator has been rearchitected for better performance and maintainability. Gazebo Fortress offers:

- **Modular architecture**: Component-based design with better separation of concerns
- **Improved performance**: Better multi-threading and resource management
- **Enhanced rendering**: More realistic lighting and materials
- **Better plugin system**: More intuitive and flexible plugin architecture

## Setting Up Gazebo Environment

### Installation

For Ubuntu 22.04 with ROS 2 Humble Hawksbill, install Gazebo Fortress:

```bash
# Add Gazebo repository
sudo apt update && sudo apt install wget
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update

# Install Gazebo Fortress
sudo apt install gz-fortress
```

### Basic Gazebo Launch

```bash
# Launch Gazebo with an empty world
gz sim -r empty.sdf

# Launch with a specific world file
gz sim -r my_world.sdf
```

## Robot Modeling in Gazebo

### URDF to SDF Conversion

Gazebo uses SDF (Simulation Description Format) for describing robot models and environments. However, most ROS robots are described using URDF (Unified Robot Description Format). The Gazebo ROS packages provide tools for seamless integration between URDF and SDF.

### Creating Robot Models

A robot model in Gazebo consists of several key elements:

1. **Links**: Rigid bodies with physical properties
2. **Joints**: Connections between links with specific degrees of freedom
3. **Inertial properties**: Mass, center of mass, and inertia tensor
4. **Visual properties**: How the robot appears in simulation
5. **Collision properties**: How the robot interacts with the environment

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="simple_robot">
    <!-- Base link -->
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </visual>

      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Wheel joint and link -->
    <joint name="wheel_joint" type="continuous">
      <parent>base_link</parent>
      <child>wheel_link</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>

    <link name="wheel_link">
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>

      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </visual>

      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

## Physics Simulation

### Physics Engines

Gazebo supports multiple physics engines, each with different characteristics:

- **ODE (Open Dynamics Engine)**: Fast, suitable for most applications
- **Bullet**: Good balance of speed and accuracy
- **Simbody**: High accuracy for complex articulated systems
- **DART**: Advanced contact handling and stability

The choice of physics engine can significantly impact simulation accuracy and performance. For humanoid robotics applications, Bullet or DART are often preferred due to their superior contact handling capabilities.

### Physics Parameters

Physics parameters in Gazebo control how objects behave in simulation:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
</physics>
```

- **max_step_size**: Smaller values increase accuracy but decrease performance
- **real_time_factor**: Target simulation speed relative to real time
- **real_time_update_rate**: Updates per second

## Sensor Simulation

Gazebo provides realistic simulation of various sensors:

### Camera Sensors

```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LIDAR Sensors

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
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
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensors

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Environment Modeling

Creating realistic environments is crucial for effective simulation. Gazebo allows for the creation of complex indoor and outdoor environments:

### World Files

World files in SDF format define the entire simulation environment:

```xml
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models -->
    <model name="table">
      <pose>1 0 0 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Integration with ROS 2

Gazebo integrates seamlessly with ROS 2 through the Gazebo ROS packages:

### Launch Files

```xml
<launch>
  <!-- Start Gazebo server -->
  <node name="gzserver" pkg="ros_gz_sim" exec="gzserver" args="-r empty.sdf"/>

  <!-- Start Gazebo client -->
  <node name="gzclient" pkg="ros_gz_sim" exec="gzclient" output="screen"/>

  <!-- Spawn robot model -->
  <node name="spawn_entity" pkg="ros_gz_sim" exec="spawn_entity.py"
        args="-topic robot_description -entity simple_robot"/>
</launch>
```

### Topics and Services

Gazebo provides various ROS 2 topics and services for controlling the simulation:

- `/clock`: Simulation time
- `/model_states`: State of all models
- `/joint_states`: Joint positions, velocities, efforts
- `/gazebo/reset_simulation`: Service to reset simulation
- `/gazebo/set_model_state`: Service to set model state

## Best Practices for Simulation

### Accuracy vs. Performance

When setting up simulation parameters, there's always a trade-off between accuracy and performance:

1. **Physics step size**: Smaller steps increase accuracy but decrease performance
2. **Update rates**: Higher rates improve responsiveness but increase computational load
3. **Sensor noise**: Realistic noise models improve training transfer but add complexity

### Sim-to-Real Transfer

To maximize the effectiveness of simulation for real robot development:

1. **Model calibration**: Accurately model robot physical properties
2. **Sensor noise**: Include realistic sensor noise and limitations
3. **Environment complexity**: Gradually increase environment complexity
4. **Domain randomization**: Randomize parameters to improve robustness

## Summary

This chapter covered the fundamentals of Gazebo simulation and physics modeling. We explored the architecture of Gazebo Fortress, how to create robot models with accurate physics properties, and how to integrate with ROS 2. Understanding these concepts is crucial for effective robotic development using digital twin technology.

In the next chapter, we'll explore Unity integration for advanced visualization and how it complements Gazebo simulation for comprehensive digital twin implementations.