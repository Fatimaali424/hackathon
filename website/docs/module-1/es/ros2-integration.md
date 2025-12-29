---
sidebar_position: 4
---

# ROS 2 Integration
## Overview
This chapter focuses on integrating ROS 2 with real Robotic Systems, covering practical implementation techniques, hardware interfaces, and deployment strategies. We'll explore how to connect ROS 2 with various sensors, actuators, and control systems to create complete robotic applications.

## Learning Objectives
After completing this chapter, you will be able to:
- Integrate ROS 2 with hardware sensors and actuators
- Implement robot drivers and hardware interfaces
- Configure and use ROS 2 control frameworks
- Deploy ROS 2 applications on embedded systems
- Integrate with simulation environments
- Handle real-time constraints in Robotic Systems

## Hardware Integration Overview
### ROS 2 Hardware Abstraction Layer
ROS 2 provides a standardized interface for hardware integration through the `ros2_control` framework. This framework abstracts hardware-specific details and provides a consistent interface for controlling robots.

```cpp
// Example of a hardware interface
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"

class MyRobotHardware : public hardware_interface::SystemInterface
{
public:
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override
    {
        if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS)
        {
            return CallbackReturn::ERROR;
        }

        // Initialize hardware components
        // Parse configuration from URDF
        return CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override
    {
        std::vector<hardware_interface::StateInterface> state_interfaces;

        // Export state interfaces for joints
        for (auto i = 0u; i < info_.joints.size(); i++)
        {
            state_interfaces.emplace_back(
                hardware_interface::StateInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            state_interfaces.emplace_back(
                hardware_interface::StateInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
        }

        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override
    {
        std::vector<hardware_interface::CommandInterface> command_interfaces;

        // Export command interfaces for joints
        for (auto i = 0u; i < info_.joints.size(); i++)
        {
            command_interfaces.emplace_back(
                hardware_interface::CommandInterface(
                    info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]));
        }

        return command_interfaces;
    }

    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override
    {
        // Activate hardware
        return SystemInterface::on_activate(previous_state);
    }

    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override
    {
        // Deactivate hardware
        return SystemInterface::on_deactivate(previous_state);
    }

    hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override
    {
        // Read data from hardware
        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override
    {
        // Write commands to hardware
        return hardware_interface::return_type::OK;
    }

private:
    std::vector<double> hw_commands_;
    std::vector<double> hw_positions_;
    std::vector<double> hw_velocities_;
};
```

## Sensor Integration
### Camera Integration
Integrating cameras with ROS 2 requires using the `image_transport` package and appropriate camera drivers:

```python
# Example camera publisher node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.cap = cv2.VideoCapture(0)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()
```

### LIDAR Integration
LIDAR sensors typically publish point cloud data in the `sensor_msgs/PointCloud2` format:

```cpp
// Example LIDAR interface
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

class LIDARInterface : public rclcpp::Node
{
public:
    LIDARInterface() : Node("lidar_interface")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar/points", 10);
    }

private:
    void process_lidar_data(const std::vector<float>& ranges)
    {
        auto msg = sensor_msgs::msg::PointCloud2();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "lidar_link";

        // Configure message structure
        msg.height = 1;
        msg.width = ranges.size();
        msg.is_dense = false;
        msg.is_bigendian = false;

        // Define fields
        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

        // Fill point cloud data
        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");

        for (size_t i = 0; i < ranges.size(); ++i)
        {
            *iter_x = ranges[i] * cos(angle_increment * i);
            *iter_y = ranges[i] * sin(angle_increment * i);
            *iter_z = 0.0;
            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        publisher_->publish(msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};
```

## Control System Integration
### ROS 2 Control Framework
The `ros2_control` framework provides standardized interfaces for robot control:

```yaml
# Example controller configuration (controllers.yaml)
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    forward_position_controller:
      type: position_controllers/JointGroupPositionController

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

forward_position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

joint_trajectory_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
```

### Joint State Publisher
```python
# Example joint state publisher
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

        # Initialize joint names and positions
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_positions = [0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0]

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.publisher_.publish(msg)
```

## Communication with External Systems
### Serial Communication
Many hardware devices communicate via serial protocols:

```python
# Example serial communication node
import rclpy
from rclpy.node import Node
import serial
import struct

class SerialInterface(Node):
    def __init__(self):
        super().__init__('serial_interface')
        self.publisher_ = self.create_publisher(String, 'serial_data', 10)
        self.subscription_ = self.create_subscription(
            String, 'serial_command', self.command_callback, 10)

        # Connect to hardware
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        self.timer = self.create_timer(0.01, self.read_serial)

    def read_serial(self):
        if self.serial_port.in_waiting > 0:
            data = self.serial_port.readline().decode('utf-8').strip()
            if data:
                msg = String()
                msg.data = data
                self.publisher_.publish(msg)

    def command_callback(self, msg):
        command = msg.data + '\n'
        self.serial_port.write(command.encode())
```

### CAN Bus Integration
For automotive and industrial applications, CAN bus integration is essential:

```cpp
// Example CAN interface (using socketcan)
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>

class CANInterface : public rclcpp::Node
{
public:
    CANInterface() : Node("can_interface")
    {
        publisher_ = this->create_publisher<can_msgs::msg::Frame>("can_rx", 10);
        subscription_ = this->create_subscription<can_msgs::msg::Frame>(
            "can_tx", 10, std::bind(&CANInterface::can_tx_callback, this, std::placeholders::_1));

        // Initialize CAN socket
        init_can_socket();
    }

private:
    void init_can_socket()
    {
        socket_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (socket_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create CAN socket");
            return;
        }

        struct ifreq ifr;
        strcpy(ifr.ifr_name, "can0");
        ioctl(socket_, SIOCGIFINDEX, &ifr);

        struct sockaddr_can addr;
        memset(&addr, 0, sizeof(addr));
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to bind CAN socket");
            return;
        }
    }

    void can_tx_callback(const can_msgs::msg::Frame::SharedPtr msg)
    {
        struct can_frame frame;
        frame.can_id = msg->id;
        frame.can_dlc = msg->dlc;
        memcpy(frame.data, msg->data.data(), frame.can_dlc);

        write(socket_, &frame, sizeof(frame));
    }

    int socket_;
    rclcpp::Publisher<can_msgs::msg::Frame>::SharedPtr publisher_;
    rclcpp::Subscription<can_msgs::msg::Frame>::SharedPtr subscription_;
};
```

## Simulation Integration
### Gazebo Integration
ROS 2 integrates with Gazebo through the `ros_gz` bridge:

```xml
<!-- Example URDF with Gazebo plugins -->
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="diff_drive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.15</wheel_diameter>
    </plugin>
  </gazebo>
</robot>
```

## Real-time Considerations
### Real-time Setup
For time-critical applications, ROS 2 can be configured for real-time performance:

```bash
# Configure kernel for real-time
echo 'kernel.sched_rt_runtime_us = -1' | sudo tee -a /etc/security/limits.conf
echo 'kernel.sched_rt_period_us = 1000000' | sudo tee -a /etc/security/limits.conf

# Add user to real-time group
sudo usermod -a -G realtime $USER
```

### Real-time Scheduling
```cpp
// Example of setting real-time priority
#include <sched.h>
#include <sys/mman.h>

class RealTimeNode : public rclcpp::Node
{
public:
    RealTimeNode() : Node("realtime_node")
    {
        // Lock memory to prevent page faults
        mlockall(MCL_CURRENT | MCL_FUTURE);

        // Set real-time scheduling policy
        struct sched_param param;
        param.sched_priority = 80;  // High priority
        if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
            RCLCPP_WARN(this->get_logger(), "Failed to set real-time priority");
        }

        // Create real-time timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&RealTimeNode::realtime_callback, this));
    }

private:
    void realtime_callback()
    {
        // Time-critical callback code
        // This will run with real-time priority
    }

    rclcpp::TimerBase::SharedPtr timer_;
};
```

## Deployment on Embedded Systems
### Resource Optimization
When deploying on embedded systems, consider resource constraints:

```bash
# Cross-compilation for embedded systems
colcon build --merge-install --packages-select my_robot_package \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
```

### Docker for Deployment
```dockerfile
# Example Dockerfile for ROS 2 application
FROM ros:humble-ros-base-jammy

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Build application
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select my_robot_package

# Source ROS and run application
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.sh && source install/setup.sh && ros2 run my_robot_package my_robot_node"]
```

## Debugging and Monitoring
### System Monitoring
```python
# Example system monitoring node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
import json

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')
        self.publisher_ = self.create_publisher(String, 'system_status', 10)
        self.timer = self.create_timer(1.0, self.monitor_system)

    def monitor_system(self):
        status = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self.get_temperature(),
            'timestamp': self.get_clock().now().to_msg()
        }

        msg = String()
        msg.data = json.dumps(status)
        self.publisher_.publish(msg)

    def get_temperature(self):
        # Implementation depends on hardware
        # This is a placeholder
        return 35.0  # degrees Celsius
```

## Best Practices for Integration
### Error Handling
```cpp
// Robust error handling for hardware interfaces
class RobustHardwareInterface : public rclcpp::Node
{
public:
    RobustHardwareInterface() : Node("robust_hardware_interface")
    {
        // Initialize with error handling
        if (!initialize_hardware()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize hardware, shutting down");
            rclcpp::shutdown();
            return;
        }

        // Setup error monitoring
        error_timer_ = this->create_timer(
            std::chrono::seconds(1),
            std::bind(&RobustHardwareInterface::check_hardware_status, this));
    }

private:
    bool initialize_hardware()
    {
        try {
            // Hardware initialization code
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Hardware initialization failed: %s", e.what());
            return false;
        }
    }

    void check_hardware_status()
    {
        if (!is_hardware_connected()) {
            RCLCPP_WARN(this->get_logger(), "Hardware connection lost");
            // Implement recovery strategy
        }
    }

    rclcpp::TimerBase::SharedPtr error_timer_;
};
```

## Summary
This chapter covered the practical aspects of integrating ROS 2 with real Robotic Systems, including hardware interfaces, control systems, and deployment strategies. We explored various integration patterns, real-time considerations, and best practices for robust robotic applications.

The next phase will involve implementing lab exercises that apply these concepts in practical scenarios, starting with basic publisher/subscriber patterns and progressing to more complex multi-node systems.