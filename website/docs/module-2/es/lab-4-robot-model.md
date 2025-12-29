---
sidebar_position: 7
---

# Lab 4: Basic Robot Model and Simulation
## Overview
This lab provides hands-on experience in creating a basic robot model and setting up a simulation environment in Gazebo. You will learn to define robot geometry, configure physical properties, and create a functional simulation that can be controlled through ROS 2.

## Learning Objectives
After completing this lab, you will be able to:
- Create a URDF robot model with proper kinematic and dynamic properties
- Configure Gazebo simulation parameters for the robot model
- Launch and visualize the robot in Gazebo simulation environment
- Control the robot using ROS 2 commands and visualize sensor data
- Debug common issues in robot simulation setup

## Prerequisites
- Completion of Module 1: The Robotic Nervous System (ROS 2)
- Understanding of ROS 2 concepts including topics, nodes, and launch files
- Basic understanding of coordinate systems and transformations
- Familiarity with Gazebo concepts from Chapter 1 (Gazebo Simulation & Physics Modeling)

## Hardware/Software Requirements
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo Fortress (Ignition)
- Basic text editor or IDE
- Minimum 8GB RAM recommended

## Lab Duration
Estimated completion time: 2-3 hours

## Part 1: Creating the Robot URDF Model
### Step 1: Project Setup
First, let's create a workspace for our robot model:

```bash
# Create workspace directory
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws/src

# Create robot description package
ros2 pkg create --build-type ament_cmake robot_description --dependencies urdf xacro
```

### Step 2: Basic Robot Structure
Create the main URDF file for our simple differential drive robot:

```xml
<!-- File: ~/robotics_ws/src/robot_description/urdf/simple_robot.urdf -->
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0125" ixy="0" ixz="0" iyy="0.0208" iyz="0" izz="0.0333"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera Link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

</robot>
```

### Step 3: Gazebo Integration
Create a Gazebo-specific version of the robot model with plugins:

```xml
<!-- File: ~/robotics_ws/src/robot_description/urdf/simple_robot.gazebo.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Gazebo plugin for differential drive -->
  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <publish_odom_tf>true</publish_odom_tf>
      <odometry_source>world</odometry_source>
    </plugin>
  </gazebo>

  <!-- Camera plugin -->
  <gazebo reference="camera_link">
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
      <topic>camera/image_raw</topic>
    </sensor>
  </gazebo>

  <!-- Color materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="camera_link">
    <material>Gazebo/Red</material>
  </gazebo>

</robot>
```

### Step 4: Combined URDF with Xacro
Create a combined URDF using Xacro for cleaner organization:

```xml
<!-- File: ~/robotics_ws/src/robot_description/urdf/robot.xacro -->
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Properties -->
  <xacro:property name="base_width" value="0.5"/>
  <xacro:property name="base_length" value="0.3"/>
  <xacro:property name="base_height" value="0.15"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_separation" value="0.4"/>
  <xacro:property name="wheel_offset" value="0.05"/>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="1.0"/>
      <inertia
        ixx="0.0125" ixy="0" ixz="0"
        iyy="0.0208" iyz="0" izz="0.0333"/>
    </inertial>
  </link>

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix *joint_origin">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>

      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="0.2"/>
        <inertia
          ixx="0.0005" ixy="0" ixz="0"
          iyy="0.0005" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="joint_origin"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Wheels -->
  <xacro:wheel prefix="left">
    <origin xyz="0 ${wheel_separation/2} -${wheel_offset}" rpy="1.5708 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="right">
    <origin xyz="0 -${wheel_separation/2} -${wheel_offset}" rpy="1.5708 0 0"/>
  </xacro:wheel>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.05"/>
      <inertia
        ixx="0.00001" ixy="0" ixz="0"
        iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_width/2-0.025} 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- Include Gazebo-specific configurations -->
  <xacro:include filename="$(find robot_description)/urdf/robot.gazebo.xacro"/>

</robot>
```

## Part 2: Creating Launch Files
### Step 5: Robot State Publisher Launch
Create a launch file to publish robot state:

```python
# File: ~/robotics_ws/src/robot_description/launch/robot_state_publisher.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robot.xacro'
    )

    # Read robot description from file
    with open(robot_description_path, 'r') as infp:
        robot_description = infp.read()

    # Robot state publisher node
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

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        robot_state_publisher
    ])
```

### Step 6: Gazebo Launch File
Create a launch file to spawn the robot in Gazebo:

```python
# File: ~/robotics_ws/src/robot_description/launch/spawn_robot.launch.py
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

    # Paths
    pkg_gazebo_ros = FindPackageShare('ros_gz_sim')
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('robot_description'),
        'urdf',
        'robot.xacro'
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
            'gz_args': '-r empty.sdf'  # Launch with empty world
        }.items()
    )

    # Read robot description
    with open(os.path.join(
        FindPackageShare('robot_description').perform({}),
        'urdf',
        'robot.xacro'
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
            '-entity', 'simple_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Part 3: Building and Testing the Robot Model
### Step 7: Package.xml Configuration
Update the package.xml file to include necessary dependencies:

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

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Step 8: CMakeLists.txt Configuration
Update the CMakeLists.txt file:

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
  DESTINATION share/${PROJECT_NAME}
)

# Install other files
install(PROGRAMS
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

### Step 9: Building the Package
Build the robot description package:

```bash
cd ~/robotics_ws
colcon build --packages-select robot_description
source install/setup.bash
```

### Step 10: Testing the Robot Model
Test the URDF model in RViz:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# Launch robot state publisher
ros2 launch robot_description robot_state_publisher.launch.py
```

In another terminal, visualize the robot:

```bash
# Install and run RViz if not already installed
sudo apt install ros-humble-rviz2

# Run RViz
rviz2
```

In RViz, add a RobotModel display and set the Robot Description to `/robot_description`.

## Part 4: Running the Simulation
### Step 11: Launch Gazebo Simulation
Launch the robot in Gazebo:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# Launch the robot in Gazebo
ros2 launch robot_description spawn_robot.launch.py
```

### Step 12: Controlling the Robot
In a new terminal, send velocity commands to the robot:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# Send a velocity command to move the robot forward
ros2 topic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"
```

Or use the teleop_twist_keyboard package for interactive control:

```bash
# Install teleop package if not already installed
sudo apt install ros-humble-teleop-twist-keyboard

# Run teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Step 13: Viewing Sensor Data
Check the camera feed from the robot:

```bash
# Source the workspace
source ~/robotics_ws/install/setup.bash

# View camera image
rqt_image_view /camera/image_raw
```

## Part 5: Debugging and Troubleshooting
### Common Issues and Solutions
1. **Robot not showing in Gazebo**: Check that the URDF is valid and all links/joints are properly defined.

2. **Robot falls through the ground**: Verify inertial properties and collision geometries.

3. **No response to velocity commands**: Check that the diff_drive plugin is properly configured and topics are connected.

4. **TF errors**: Ensure all frames are properly connected in the URDF.

### Debugging Commands
```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Echo TF transforms
ros2 topic echo /tf

# Check robot description
ros2 param get /robot_state_publisher robot_description

# List all topics
ros2 topic list
```

## Lab Assignment
### Task 1: Modify the Robot Model- Add a simple gripper to the front of the robot
- Adjust the mass and inertia properties to account for the new component
- Verify that the simulation still works properly

### Task 2: Create a Custom Environment- Create a simple world file with obstacles
- Modify the launch file to use your custom world
- Test the robot's navigation in the new environment

### Task 3: Add Additional Sensors- Add an IMU sensor to the robot
- Add a LIDAR sensor to the robot
- Verify that sensor data is published correctly

## Summary
In this lab, you created a complete robot model with proper URDF definition, configured Gazebo simulation with appropriate plugins, and successfully launched the robot in simulation. You learned how to control the robot using ROS 2 commands and visualize its sensor data. This foundation will be essential for more advanced robotics applications in subsequent modules.

## Additional Resources
- [ROS URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [Gazebo Documentation](http://gazebosim.org/tutorials)
- [Xacro Documentation](http://wiki.ros.org/xacro)
- [ROS 2 Control Documentation](https://control.ros.org/)

## Next Steps
After completing this lab, you should:
- Understand the structure of robot models in ROS 2
- Be able to create and simulate custom robot models
- Have experience with Gazebo simulation setup
- Be ready to proceed to Lab 5: Advanced Sensor Simulation