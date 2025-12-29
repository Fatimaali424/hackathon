---
sidebar_position: 5
---

# Lab 1: Basic ROS 2 Publisher/Subscriber
## Overview
In this lab, you will implement a basic publisher/subscriber system in ROS 2. This is the fundamental communication pattern in ROS 2 and forms the basis for most robotic applications. You'll create a publisher node that sends messages and a subscriber node that receives and processes those messages.

## Learning Objectives
After completing this lab, you will be able to:
- Create ROS 2 publisher and subscriber nodes
- Implement message passing between nodes
- Use ROS 2 command-line tools to monitor communication
- Debug basic ROS 2 communication issues
- Understand Quality of Service (QoS) settings

## Prerequisites
- ROS 2 Humble Hawksbill installed
- Basic knowledge of Python or C++
- Understanding of ROS 2 concepts from previous chapters

## Lab Setup
First, create a new ROS 2 package for this lab:

```bash
# Create a new workspace if you don't have one
mkdir -p ~/ros2_labs/src
cd ~/ros2_labs/src

# Create the lab package
ros2 pkg create --build-type ament_python ros2_lab1 --dependencies rclpy std_msgs
```

## Python Implementation
### Publisher Node
Create the publisher node in `ros2_lab1/ros2_lab1/publisher_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Subscriber Node
Create the subscriber node in `ros2_lab1/ros2_lab1/subscriber_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Setup for Execution
Update the `setup.py` file to make the nodes executable:

```python
from setuptools import find_packages, setup

package_name = 'ros2_lab1'

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
    description='Basic ROS 2 publisher/subscriber lab',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_lab1.publisher_member_function:main',
            'listener = ros2_lab1.subscriber_member_function:main',
        ],
    },
)
```

## C++ Implementation (Alternative)
For those who prefer C++, here's the equivalent implementation:

### Publisher Node (C++)
Create `ros2_lab1/src/talker.cpp`:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

### Subscriber Node (C++)
Create `ros2_lab1/src/listener.cpp`:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
            });
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

For C++, also update the `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(ros2_lab1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Create executables
add_executable(talker src/talker.cpp)
add_executable(listener src/listener.cpp)

# Link libraries
ament_target_dependencies(talker rclcpp std_msgs)
ament_target_dependencies(listener rclcpp std_msgs)

install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Building and Running the Lab
### For Python Implementation:
```bash
# Navigate to your workspace
cd ~/ros2_labs

# Build the package
colcon build --packages-select ros2_lab1

# Source the workspace
source install/setup.bash

# Run the publisher in one terminal
ros2 run ros2_lab1 talker

# In another terminal, run the subscriber
ros2 run ros2_lab1 listener
```

### For C++ Implementation:
```bash
# Navigate to your workspace
cd ~/ros2_labs

# Build the package
colcon build --packages-select ros2_lab1

# Source the workspace
source install/setup.bash

# Run the publisher in one terminal
ros2 run ros2_lab1 talker

# In another terminal, run the subscriber
ros2 run ros2_lab1 listener
```

## Monitoring Communication
### Using ROS 2 Command Line Tools
Monitor the communication between nodes:

```bash
# List all active topics
ros2 topic list

# Echo messages on the topic
ros2 topic echo /topic std_msgs/msg/String

# Get information about the topic
ros2 topic info /topic

# Show bandwidth usage
ros2 topic hz /topic
```

### Using Quality of Service Settings
Modify the publisher to use different QoS settings:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE
)

# Use the custom QoS profile
self.publisher_ = self.create_publisher(String, 'topic', qos_profile)
```

## Lab Exercises
### Exercise 1: Modify Message Content1. Change the publisher to send different types of messages (e.g., include timestamps, sequence numbers)
2. Modify the message format to include more structured data

### Exercise 2: Multiple Publishers and Subscribers1. Create multiple publisher nodes with different message patterns
2. Create multiple subscriber nodes that process messages differently
3. Observe how messages are distributed among subscribers

### Exercise 3: Quality of Service Experimentation1. Try different QoS settings (reliable vs best effort, different history depths)
2. Observe the impact on message delivery and performance
3. Document when to use different QoS policies

### Exercise 4: Error Handling1. Add error handling to your publisher and subscriber nodes
2. Implement graceful degradation when communication fails
3. Add logging for debugging purposes

## Troubleshooting Common Issues
### Nodes Can't Communicate- Check that both nodes are on the same ROS domain ID
- Verify that topic names match exactly
- Ensure both nodes are using compatible QoS settings

### Performance Issues- Check system resources (CPU, memory, network)
- Adjust QoS settings for better performance
- Consider message size and frequency

### Discovery Issues- Verify network configuration
- Check for firewall blocking ROS communication
- Ensure RMW implementation is properly configured

## Summary
In this lab, you've implemented a basic publisher/subscriber system in ROS 2, which is the fundamental communication pattern in ROS. You've learned how to:

1. Create publisher and subscriber nodes in both Python and C++
2. Use ROS 2 command-line tools to monitor communication
3. Configure Quality of Service settings
4. Troubleshoot common communication issues

This foundation will be essential as you progress to more complex ROS 2 applications involving services, actions, and multi-node systems.