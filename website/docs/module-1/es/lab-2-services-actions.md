---
sidebar_position: 6
---

# Lab 2: Service and Action Implementation
## Overview
In this lab, you will implement service and action communication patterns in ROS 2. Services provide synchronous request-response communication, while actions handle long-running tasks with feedback. These patterns complement the publisher-subscriber pattern you learned in Lab 1 and are essential for building complete robotic applications.

## Learning Objectives
After completing this lab, you will be able to:
- Create and use ROS 2 services for synchronous communication
- Implement ROS 2 actions for long-running tasks with feedback
- Understand when to use services vs actions vs topics
- Debug service and action communication issues
- Integrate services and actions with existing publisher/subscriber systems

## Prerequisites
- Completion of Lab 1 (Publisher/Subscriber)
- ROS 2 Humble Hawksbill installed
- Basic knowledge of Python or C++
- Understanding of ROS 2 communication patterns

## Lab Setup
If you don't have the lab workspace from Lab 1, create it:

```bash
# Navigate to your workspace
cd ~/ros2_labs/src

# If you didn't create the package in Lab 1, do so now
ros2 pkg create --build-type ament_python ros2_lab2 --dependencies rclpy std_msgs example_interfaces
```

## Service Implementation
### Creating a Custom Service Message
First, let's create a custom service message. Create the service definition file in `ros2_lab2/srv/AddTwoInts.srv`:

```
int64 a
int64 b
---
int64 sum
```

### Service Server Node
Create the service server in `ros2_lab2/ros2_lab2/service_server.py`:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Client Node
Create the service client in `ros2_lab2/ros2_lab2/service_client.py`:

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Setup for Execution
Update the `setup.py` file:

```python
from setuptools import find_packages, setup

package_name = 'ros2_lab2'

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
    description='ROS 2 services and actions lab',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'server = ros2_lab2.service_server:main',
            'client = ros2_lab2.service_client:main',
            'action_server = ros2_lab2.action_server:main',
            'action_client = ros2_lab2.action_client:main',
        ],
    },
)
```

## Action Implementation
### Creating a Custom Action Message
Create an action definition file in `ros2_lab2/action/Fibonacci.action`:

```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server Node
Create the action server in `ros2_lab2/ros2_lab2/action_server.py`:

```python
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback)

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """Handle an accepted goal."""
        self.get_logger().info('Goal accepted')
        # Start executing the action
        goal_handle.execute()

    def cancel_callback(self, goal):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Get the goal order
        order = goal_handle.request.order

        # Create feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        # Simulate long-running task
        for i in range(1, order):
            # Check if there was a cancel request
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            # Update feedback
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Simulate work
            time.sleep(1.0)

        # Check if goal was canceled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
            return Fibonacci.Result()

        # Set result and succeed
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(fibonacci_action_server, executor=executor)

    fibonacci_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Action Client Node
Create the action client in `ros2_lab2/ros2_lab2/action_client.py`:

```python
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci',
            callback_group=ReentrantCallbackGroup())

    def send_goal(self, order):
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create a goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Send the goal
        self.get_logger().info(f'Sending goal with order: {order}')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        # Add a callback to handle the result
        send_goal_future.add_done_callback(self.goal_response_callback)

        return send_goal_future

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')


def main(args=None):
    rclpy.init(args=args)

    action_client = FibonacciActionClient()

    # Send goal based on command line argument
    goal_order = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    future = action_client.send_goal(goal_order)

    # Keep spinning until the result is received
    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
```

## C++ Implementation (Alternative)
For those who prefer C++, here's the equivalent implementation:

### Service Server (C++)
Create `ros2_lab2/src/service_server.cpp`:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

using AddTwoInts = example_interfaces::srv::AddTwoInts;

class MinimalService : public rclcpp::Node
{
public:
    MinimalService()
    : Node("minimal_service")
    {
        using std::placeholders::_1;
        using std::placeholders::_2;
        service_ = this->create_service<AddTwoInts>(
            "add_two_ints",
            std::bind(&MinimalService::add, this, _1, _2));
    }

private:
    void add(const AddTwoInts::Request::SharedPtr request,
             const AddTwoInts::Response::SharedPtr response)
    {
        response->sum = request->a + request->b;
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Incoming request\na: %ld b: %ld",
                    request->a, request->b);
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sending back response: [%ld]", (long int)response->sum);
    }
    rclcpp::Service<AddTwoInts>::SharedPtr service_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalService>());
    rclcpp::shutdown();
    return 0;
}
```

### Action Server (C++)
Create `ros2_lab2/src/action_server.cpp`:

```cpp
#include "example_interfaces/action/fibonacci.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include <memory>
#include <thread>

class MinimalActionServer : public rclcpp::Node
{
public:
    using Fibonacci = example_interfaces::action::Fibonacci;
    using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

    explicit MinimalActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : Node("minimal_action_server", options)
    {
        using namespace std::placeholders;

        action_server_ = rclcpp_action::create_server<Fibonacci>(
            this,
            "fibonacci",
            std::bind(&MinimalActionServer::handle_goal, this, _1, _2),
            std::bind(&MinimalActionServer::handle_cancel, this, _1),
            std::bind(&MinimalActionServer::handle_accepted, this, _1));
    }

private:
    rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const Fibonacci::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        using namespace std::placeholders;
        std::thread{std::bind(&MinimalActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Executing goal");

        rclcpp::Rate loop_rate(1);
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<Fibonacci::Feedback>();
        auto result = std::make_shared<Fibonacci::Result>();

        // Start executing the action
        auto sequence = std::vector<int32_t>{};
        sequence.push_back(0);
        sequence.push_back(1);

        auto comm_index = 0;
        for (int32_t i = 1; (i < goal->order) && (rclcpp::ok()); ++i) {
            // Check if there is a cancel request
            if (goal_handle->is_canceling()) {
                result->sequence = sequence;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal Canceled");
                return;
            }

            // Update sequence
            sequence.push_back(sequence[i] + sequence[i - 1]);

            // Publish feedback
            feedback->sequence = sequence;
            goal_handle->publish_feedback(feedback);
            RCLCPP_INFO(this->get_logger(), "Publish Feedback");

            loop_rate.sleep();
        }

        // Check if goal is done
        if (rclcpp::ok()) {
            result->sequence = sequence;
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal Succeeded");
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto action_server = std::make_shared<MinimalActionServer>();
    rclcpp::spin(action_server);
    rclcpp::shutdown();
    return 0;
}
```

## Building and Running the Lab
### For Python Implementation:
```bash
# Navigate to your workspace
cd ~/ros2_labs

# Build the package
colcon build --packages-select ros2_lab2

# Source the workspace
source install/setup.bash

# Run the service server in one terminal
ros2 run ros2_lab2 server

# In another terminal, run the service client
ros2 run ros2_lab2 client 10 20

# Run the action server in one terminal
ros2 run ros2_lab2 action_server

# In another terminal, run the action client
ros2 run ros2_lab2 action_client 10
```

## Monitoring Services and Actions
### Using ROS 2 Command Line Tools
Monitor services and actions:

```bash
# List all services
ros2 service list

# Get information about a specific service
ros2 service info /add_two_ints

# Call a service directly from command line
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 10, b: 20}"

# List all actions
ros2 action list

# Get information about a specific action
ros2 action info /fibonacci

# Send a goal to an action server
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 10}"
```

## Lab Exercises
### Exercise 1: Modify Service Behavior1. Create a service that performs more complex calculations (e.g., geometric operations)
2. Add error handling for invalid inputs
3. Implement service timeouts

### Exercise 2: Action with Multiple Feedback Types1. Modify the Fibonacci action to provide different types of feedback
2. Add intermediate goals that can be reached during execution
3. Implement action preemption (canceling current goal to start a new one)

### Exercise 3: Service-Action Integration1. Create a system that uses both services and actions together
2. Implement a service that manages multiple action goals
3. Design a coordinator node that decides when to use services vs actions

### Exercise 4: Error Handling and Recovery1. Add comprehensive error handling to your service and action implementations
2. Implement retry mechanisms for failed communications
3. Create a monitoring node that tracks service/action success rates

## When to Use Each Communication Pattern
### Topics (Publisher/Subscriber)- Use when you need asynchronous, decoupled communication
- Good for continuous data streams (sensor data, status updates)
- When multiple publishers/subscribers are needed

### Services- Use for synchronous request-response patterns
- When you need guaranteed delivery and response
- For operations that should complete quickly
- When you need a simple, direct communication

### Actions- Use for long-running operations with feedback
- When you need to track progress of a task
- For operations that might be canceled
- When you need to handle goals with intermediate results

## Troubleshooting Common Issues
### Service Issues- **Service not found**: Check that the service server is running and service names match
- **Timeout errors**: Increase timeout values or check network connectivity
- **Type mismatch**: Ensure request/response message types match

### Action Issues- **Goal not accepted**: Check that the action server is running and accepting goals
- **Feedback not received**: Verify the action server is publishing feedback
- **Cancel not working**: Ensure proper cancel handling in the action server

## Summary
In this lab, you've implemented both service and action communication patterns in ROS 2:

1. **Services**: Synchronous request-response communication for immediate operations
2. **Actions**: Long-running operations with feedback, status, and cancellation support
3. **When to use each pattern**: Understanding the appropriate use cases for topics, services, and actions

These communication patterns, combined with the publisher/subscriber pattern from Lab 1, provide a complete toolkit for building complex robotic applications with ROS 2.