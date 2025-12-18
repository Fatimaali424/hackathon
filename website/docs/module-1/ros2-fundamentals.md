---
sidebar_position: 2
---

# ROS 2 Fundamentals

## Overview

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms. This chapter introduces the fundamental concepts of ROS 2 that form the foundation for all robotic applications.

## Learning Objectives

After completing this chapter, you will be able to:
- Explain the core concepts and architecture of ROS 2
- Create and run basic ROS 2 nodes
- Understand and implement publisher-subscriber communication
- Use services and actions for synchronous and asynchronous communication
- Configure ROS 2 environments and manage packages

## What is ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to address the limitations of ROS 1 while maintaining its core strengths. Unlike ROS 1 which was built around a centralized master architecture, ROS 2 is built on DDS (Data Distribution Service) which provides a decentralized, peer-to-peer communication model.

### Key Improvements in ROS 2

1. **Real-time support**: ROS 2 provides better real-time capabilities for time-critical applications
2. **Multi-robot systems**: Native support for multiple robots without a central master
3. **Security**: Built-in security features for protecting robotic systems
4. **Cross-platform support**: Improved support for Windows, macOS, and various Linux distributions
5. **Production deployment**: Better tools for deploying ROS 2 applications in production environments

## Core Concepts

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system. Each node typically performs a specific function and communicates with other nodes to achieve complex robot behaviors.

```python
# Example of a simple ROS 2 node
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
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
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic. Topics enable asynchronous communication between nodes.

Messages are the data structures that are passed between nodes. ROS 2 provides many built-in message types and allows users to define custom message types.

### Services

Services provide synchronous request-response communication between nodes. When a client sends a request to a service, it waits for the service to process the request and return a response.

### Actions

Actions are for long-running tasks that require status, feedback, and goals. They are built on top of services and topics to provide a more sophisticated communication pattern for complex operations.

## ROS 2 Architecture

### DDS (Data Distribution Service)

DDS is the middleware that underlies ROS 2's communication. It handles the discovery of nodes, message routing, and quality of service settings. DDS provides:

- **Discovery**: Automatic discovery of nodes and topics
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, and other communication characteristics
- **Transport**: Multiple transport options including UDP, TCP, shared memory, and more

### Quality of Service (QoS)

QoS profiles allow you to configure how messages are handled based on the requirements of your application:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local storage
- **History**: Keep all messages or just the last few
- **Depth**: How many messages to keep in history

## Setting Up Your ROS 2 Environment

### Installation

ROS 2 Humble Hawksbill is the recommended version for this book. Installation instructions vary by platform:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup

After installation, source the ROS 2 setup script:

```bash
source /opt/ros/humble/setup.bash
```

For convenience, add this line to your `~/.bashrc` file to automatically source ROS 2 on terminal startup.

## ROS 2 Packages

Packages are the fundamental building blocks of ROS 2. A package contains nodes, libraries, data, and configuration files. Packages are organized in a workspace.

### Creating a Package

```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create a new package
ros2 pkg create --build-type ament_python my_robot_package
```

## Summary

This chapter introduced the fundamental concepts of ROS 2, including nodes, topics, messages, services, and actions. We covered the architecture of ROS 2 built on DDS and how it differs from ROS 1. Finally, we looked at how to set up your development environment and create packages.

In the next chapter, we'll dive deeper into the ROS 2 architecture and explore more advanced concepts.