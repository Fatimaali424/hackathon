---
sidebar_position: 3
---

# ROS 2 Architecture

## Overview

Understanding the architecture of ROS 2 is crucial for developing robust and efficient robotic applications. This chapter explores the layered architecture of ROS 2, from the client libraries to the middleware layer and communication protocols. We'll examine how ROS 2 handles distributed computing, process management, and inter-process communication.

## Learning Objectives

After completing this chapter, you will be able to:
- Describe the layered architecture of ROS 2
- Explain the role of middleware and client libraries
- Configure Quality of Service (QoS) policies for different use cases
- Understand the execution model and process management in ROS 2
- Design efficient communication patterns for robotic systems

## Layered Architecture

ROS 2 follows a layered architecture that separates concerns and provides flexibility for different use cases. The architecture consists of several layers:

### Client Libraries Layer

The client libraries provide the API that developers use to create ROS 2 applications. The most common client libraries are:

- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: C client library (used by other client libraries)

### ROS Client Library (rcl) Layer

The rcl layer provides a common C-based API that all other client libraries build upon. This ensures consistency across different languages and reduces code duplication.

### ROS Middleware Interface (rmw) Layer

The ROS Middleware Interface provides an abstraction layer between ROS 2 and the underlying middleware implementation. This allows ROS 2 to work with different middleware implementations like DDS providers.

### Middleware Layer

The middleware layer handles the actual communication between nodes. ROS 2 uses DDS (Data Distribution Service) as its default middleware, with implementations from various vendors like Fast DDS, Cyclone DDS, and RTI Connext DDS.

## Quality of Service (QoS) in Depth

QoS policies are a fundamental aspect of ROS 2 architecture that allow you to fine-tune communication behavior based on your application's requirements.

### Reliability Policy

```cpp
// C++ example of setting reliability policy
rclcpp::QoS qos(10);
qos.reliable();  // For critical data that must be delivered
// or
qos.best_effort();  // For data where occasional loss is acceptable
```

```python
# Python example of setting reliability policy
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos_profile = QoSProfile(depth=10)
qos_profile.reliability = ReliabilityPolicy.RELIABLE
```

**Use cases:**
- Reliable: Sensor data, control commands, navigation goals
- Best Effort: Camera images, LIDAR scans, status updates

### Durability Policy

```cpp
// C++ example of setting durability policy
qos.transient_local();  // Keep messages for late-joining nodes
// or
qos.volatile();  // Don't keep historical messages
```

**Use cases:**
- Transient Local: Parameter updates, map data, static transforms
- Volatile: Real-time sensor data, live video feeds

### History Policy

```cpp
// C++ example of setting history policy
qos.keep_all();  // Keep all messages (use with caution)
// or
qos.keep_last(10);  // Keep only the last N messages
```

## Execution Model

### Single-threaded vs Multi-threaded Executors

ROS 2 provides different executor types to handle callback execution:

#### Single-threaded Executor

```python
# Python example of single-threaded executor
import rclpy
from rclpy.executors import SingleThreadedExecutor

rclpy.init()
node = MyNode()
executor = SingleThreadedExecutor()
executor.add_node(node)
executor.spin()
```

#### Multi-threaded Executor

```python
# Python example of multi-threaded executor
import rclpy
from rclpy.executors import MultiThreadedExecutor

rclpy.init()
node1 = MyNode1()
node2 = MyNode2()
executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(node1)
executor.add_node(node2)
executor.spin()
```

### Callback Groups

Callback groups allow you to control which callbacks can be executed concurrently:

```python
# Python example of callback groups
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

# Mutually exclusive - only one callback runs at a time
exclusive_group = MutuallyExclusiveCallbackGroup()

# Reentrant - multiple callbacks can run simultaneously
reentrant_group = ReentrantCallbackGroup()

# Assign subscriptions to different groups
sub1 = self.create_subscription(
    String, 'topic1', callback1, 10, callback_group=exclusive_group)

sub2 = self.create_subscription(
    String, 'topic2', callback2, 10, callback_group=reentrant_group)
```

## Communication Patterns

### Publisher-Subscriber Pattern

The publisher-subscriber pattern enables asynchronous, decoupled communication:

```cpp
// C++ example of publisher-subscriber pattern
class Talker : public rclcpp::Node
{
public:
    Talker() : Node("talker")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&Talker::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World";
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

### Service-Client Pattern

The service-client pattern provides synchronous request-response communication:

```cpp
// C++ example of service-client pattern
class AddTwoIntsService : public rclcpp::Node
{
public:
    AddTwoIntsService() : Node("add_two_ints_service")
    {
        service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            std::bind(&AddTwoIntsService::add, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    void add(const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
             const example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
    {
        response->sum = request->a + request->b;
        RCLCPP_INFO(this->get_logger(), "Incoming request: %ld + %ld = %ld",
                    request->a, request->b, response->sum);
    }
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

### Action Pattern

Actions provide a way to handle long-running tasks with feedback:

```python
# Python example of action server
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Process Management and Lifecycle

### Lifecycle Nodes

Lifecycle nodes provide a way to manage the state of nodes in a more controlled manner:

```cpp
// C++ example of lifecycle node
#include "rclcpp_lifecycle/lifecycle_node.hpp"

class LifecycleTalker : public rclcpp_lifecycle::LifecycleNode
{
public:
    LifecycleTalker() : rclcpp_lifecycle::LifecycleNode("lifecycle_talker") {}

private:
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr pub_;

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &)
    {
        pub_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
        RCLCPP_INFO(get_logger(), "Configured lifecycle publisher");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &)
    {
        pub_->on_activate();
        RCLCPP_INFO(get_logger(), "Lifecycle publisher activated");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};
```

## Performance Considerations

### Memory Management

ROS 2 provides mechanisms to optimize memory usage:

```cpp
// Using Intrusive List for zero-copy message passing
#include "rclcpp/strategies/allocator_memory_strategy.hpp"

// Custom allocator for message types
auto custom_allocator = std::make_shared<std::allocator<void>>();
auto publisher = node->create_publisher<MessageType>("topic", qos, custom_allocator);
```

### Inter-Process Communication

ROS 2 uses different IPC mechanisms based on the middleware:

- **Shared Memory**: For same-machine communication
- **TCP/UDP**: For network communication
- **Intra-process Communication**: For nodes within the same process

## Security Architecture

ROS 2 includes built-in security features:

### Authentication

- Identity verification using certificates
- Certificate Authority (CA) based authentication

### Encryption

- Message encryption using AES-256
- Secure transport protocols

### Authorization

- Access control lists (ACLs)
- Role-based permissions

## Summary

This chapter covered the detailed architecture of ROS 2, including its layered design, Quality of Service policies, execution models, and communication patterns. We explored how to optimize performance and implement security features. Understanding these architectural concepts is essential for building robust and efficient robotic applications.

The next chapter will focus on integrating ROS 2 with real robotic systems and implementing practical applications.