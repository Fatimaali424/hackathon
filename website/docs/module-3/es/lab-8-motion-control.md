---
sidebar_position: 6
---

# Lab 8: Motion Planning and Control Implementation
## Overview
This lab focuses on implementing Motion Planning and control systems using NVIDIA Isaac's tools. You will learn to create path planners, implement control algorithms, and integrate them with perception systems to create a complete autonomous navigation solution.

## Learning Objectives
After completing this lab, you will be able to:
- Implement sampling-based motion planners (RRT, PRM)
- Design and tune PID controllers for robot motion
- Integrate Motion Planning with perception and control systems
- Evaluate path quality and control performance
- Implement obstacle avoidance and dynamic replanning

## Prerequisites
- Completion of Module 1 (ROS 2 fundamentals)
- Completion of Module 2 (Simulation concepts)
- Completion of Module 3 (Isaac perception)
- Understanding of basic control theory
- Basic knowledge of Motion Planning concepts

## Hardware and Software Requirements
### Required Hardware- Mobile robot platform (TurtleBot3, Jackal, or similar)
- Jetson platform for computation (if using physical robot)
- RGB-D camera for perception
- LIDAR for obstacle detection (if not integrated in camera)

### Required Software- ROS 2 Humble Hawksbill
- Isaac ROS Navigation packages
- Navigation2 (Nav2) packages
- Gazebo or Isaac Sim for simulation (recommended)
- Python 3.10+ with NumPy, SciPy, Matplotlib

## Lab Setup
### Environment Configuration
1. **Install Navigation packages:**
   ```bash
   sudo apt update
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   sudo apt install ros-humble-isaac-ros-nav2 ros-humble-isaac-ros-navigation
   ```

2. **Create workspace:**
   ```bash
   mkdir -p ~/motion_control_ws/src
   cd ~/motion_control_ws
   colcon build
   source install/setup.bash
   ```

3. **Verify installation:**
   ```bash
   ros2 run nav2_util lifecycle_bringup
   ```

### Simulation Environment
For this lab, we'll use a simulated environment. Create the following launch file:

```python
# motion_control_sim_launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Launch Gazebo simulation
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': '<robot name="test"><link name="base_link"/></robot>'
        }]
    )

    # Launch the motion control node
    motion_control_node = Node(
        package='motion_control_lab',
        executable='motion_control_node',
        name='motion_control_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_frame': 'base_link',
            'map_frame': 'map',
            'controller_frequency': 50.0
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gazebo_launch,
        robot_state_publisher,
        motion_control_node
    ])
```

## Implementation Steps
### Step 1: Basic Motion Planner
Create a simple RRT-based motion planner:

```python
#!/usr/bin/env python3
# rrt_planner.py

import numpy as np
import random
from scipy.spatial import KDTree
import math

class RRTPlanner:
    def __init__(self, start, goal, bounds, obstacles=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds  # [(min_x, max_x), (min_y, max_y)]
        self.obstacles = obstacles or []
        self.max_iter = 10000
        self.step_size = 0.5
        self.goal_bias = 0.2

        # RRT tree: {node_index: (configuration, parent_index)}
        self.tree = {0: (self.start, None)}
        self.nodes = [self.start]  # List of configurations
        self.node_count = 1

    def plan(self):
        """Plan path from start to goal using RRT algorithm"""
        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self.sample_free_space()

            # Find nearest node in tree
            nearest_idx = self.nearest_node(q_rand)

            # Extend tree toward random point
            q_new = self.extend_toward(nearest_idx, q_rand)

            if q_new is not None:
                # Add new node to tree
                self.tree[self.node_count] = (q_new, nearest_idx)
                self.nodes.append(q_new)

                # Check if goal is reached
                if self.distance(q_new, self.goal) < self.step_size:
                    return self.reconstruct_path(self.node_count)

                self.node_count += 1

        return None  # No path found

    def sample_free_space(self):
        """Sample a random configuration in free space"""
        while True:
            # Sample random point in bounds
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            q_rand = np.array([x, y])

            # Check if point is collision-free
            if self.is_collision_free(q_rand):
                return q_rand

    def nearest_node(self, q_rand):
        """Find nearest node in tree to q_rand using KDTree"""
        if len(self.nodes) == 1:
            return 0

        # Use KDTree for efficient nearest neighbor search
        tree = KDTree(self.nodes)
        distance, idx = tree.query(q_rand)
        return idx

    def extend_toward(self, nearest_idx, q_rand):
        """Extend tree from nearest node toward q_rand"""
        q_nearest = self.tree[nearest_idx][0]

        # Calculate direction vector
        direction = q_rand - q_nearest
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            q_new = q_rand
        else:
            # Normalize and scale direction
            direction = direction / distance * self.step_size
            q_new = q_nearest + direction

        # Check if path from nearest to new is collision-free
        if self.is_path_collision_free(q_nearest, q_new):
            return q_new

        return None

    def is_collision_free(self, q):
        """Check if configuration q is collision-free"""
        for obs in self.obstacles:
            if self.point_in_obstacle(q, obs):
                return False
        return True

    def is_path_collision_free(self, q_from, q_to):
        """Check if path from q_from to q_to is collision-free"""
        # Simple linear interpolation check
        steps = int(np.linalg.norm(q_to - q_from) / 0.1)  # Check every 0.1m
        for i in range(1, steps + 1):
            t = i / steps
            q_check = q_from + t * (q_to - q_from)
            if not self.is_collision_free(q_check):
                return False
        return True

    def point_in_obstacle(self, point, obstacle):
        """Check if point is inside obstacle (circle or rectangle)"""
        if len(obstacle) == 3:  # Circle: [x, y, radius]
            center = np.array(obstacle[:2])
            radius = obstacle[2]
            return np.linalg.norm(point - center) <= radius
        elif len(obstacle) == 4:  # Rectangle: [x, y, width, height]
            x, y, w, h = obstacle
            px, py = point
            return (x <= px <= x + w) and (y <= py <= y + h)
        return False

    def reconstruct_path(self, goal_node_idx):
        """Reconstruct path from goal to start"""
        path = []
        current_idx = goal_node_idx

        while current_idx is not None:
            config = self.tree[current_idx][0]
            path.append(config)
            current_idx = self.tree[current_idx][1]

        return path[::-1]  # Reverse to get start->goal path

    def distance(self, q1, q2):
        """Calculate Euclidean distance between configurations"""
        return np.linalg.norm(q1 - q2)
```

### Step 2: Pure Pursuit Controller
Create a path-following controller:

```python
#!/usr/bin/env python3
# pure_pursuit_controller.py

import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import math

class PurePursuitController:
    def __init__(self, lookahead_distance=1.0, max_linear_vel=1.0, max_angular_vel=1.0):
        self.lookahead_distance = lookahead_distance
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

        self.path = None
        self.current_pose = None
        self.path_index = 0

    def set_path(self, path):
        """Set the path to follow"""
        self.path = path
        self.path_index = 0

    def set_current_pose(self, pose):
        """Set the current robot pose"""
        self.current_pose = pose

    def compute_control(self):
        """Compute control command to follow path"""
        if self.path is None or self.current_pose is None:
            return Twist()

        # Find lookahead point
        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            return Twist()

        # Calculate control command
        cmd = Twist()

        # Calculate distance to lookahead point
        dx = lookahead_point[0] - self.current_pose.position.x
        dy = lookahead_point[1] - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Linear velocity (proportional to distance, capped at max)
        cmd.linear.x = min(self.max_linear_vel, distance * 0.5)

        # Calculate heading to lookahead point
        target_angle = math.atan2(dy, dx)

        # Get current robot orientation (assuming simple 2D orientation)
        current_angle = self.get_current_yaw()

        # Calculate angular error
        angle_error = self.normalize_angle(target_angle - current_angle)

        # Angular velocity (proportional to angle error)
        cmd.angular.z = max(-self.max_angular_vel,
                           min(self.max_angular_vel, angle_error * 2.0))

        return cmd

    def find_lookahead_point(self):
        """Find point on path at lookahead distance"""
        if self.path_index >= len(self.path.poses):
            return None

        # Start from current path index
        for i in range(self.path_index, len(self.path.poses)):
            pose = self.path.poses[i]
            dx = pose.pose.position.x - self.current_pose.position.x
            dy = pose.pose.position.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance >= self.lookahead_distance:
                # Update path index to improve efficiency
                self.path_index = max(0, i - 5)  # Look back a bit to avoid skipping
                return [pose.pose.position.x, pose.pose.position.y]

        # If no point is far enough, return the last point
        if self.path.poses:
            last_pose = self.path.poses[-1]
            return [last_pose.pose.position.x, last_pose.pose.position.y]

        return None

    def get_current_yaw(self):
        """Extract yaw from current pose orientation"""
        # Convert quaternion to yaw (simplified for 2D)
        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
```

### Step 3: PID Controller for Velocity Control
Implement a PID controller for more precise velocity control:

```python
#!/usr/bin/env python3
# pid_controller.py

import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits  # (min, max)

        self.reset()

    def reset(self):
        """Reset the PID controller"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        """Compute PID output given error"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = 1e-6  # Prevent division by zero

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        if self.output_limits:
            min_out, max_out = self.output_limits
            output = max(min_out, min(max_out, output))

        # Update for next iteration
        self.last_error = error
        self.last_time = current_time

        return output

class VelocityController:
    def __init__(self):
        # PID controllers for linear and angular velocities
        self.linear_pid = PIDController(
            kp=1.0, ki=0.1, kd=0.05,
            output_limits=(-1.0, 1.0)
        )
        self.angular_pid = PIDController(
            kp=2.0, ki=0.1, kd=0.1,
            output_limits=(-1.0, 1.0)
        )

    def compute_velocity_commands(self, linear_error, angular_error):
        """Compute velocity commands from position/orientation errors"""
        linear_vel = self.linear_pid.compute(linear_error)
        angular_vel = self.angular_pid.compute(angular_error)

        return linear_vel, angular_vel
```

### Step 4: Integrated Motion Control Node
Create the main motion control node that integrates planning and control:

```python
#!/usr/bin/env python3
# motion_control_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import numpy as np
from rrt_planner import RRTPlanner
from pure_pursuit_controller import PurePursuitController
from pid_controller import VelocityController

class MotionControlNode(Node):
    def __init__(self):
        super().__init__('motion_control_node')

        # Parameters
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('controller_frequency', 50.0)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/motion_control/path', 10)
        self.marker_pub = self.create_publisher(Marker, '/motion_control/waypoints', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Initialize components
        self.current_pose = None
        self.current_goal = None
        self.path = None
        self.rrt_planner = None
        self.controller = PurePursuitController(
            lookahead_distance=1.0,
            max_linear_vel=0.5,
            max_angular_vel=1.0
        )
        self.velocity_controller = VelocityController()

        # Timer for control loop
        controller_freq = self.get_parameter('controller_frequency').value
        self.control_timer = self.create_timer(1.0/controller_freq, self.control_loop)

        # Obstacles from laser scan
        self.obstacles = []

        self.get_logger().info('Motion Control Node initialized')

    def odom_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """Plan path to new goal"""
        if self.current_pose is None:
            self.get_logger().warn('Cannot plan without current pose')
            return

        # Extract goal position
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y

        # Define bounds for planning (expand from current position)
        bounds = [
            (min(start_x, goal_x) - 5, max(start_x, goal_x) + 5),
            (min(start_y, goal_y) - 5, max(start_y, goal_y) + 5)
        ]

        # Create planner with obstacles
        self.rrt_planner = RRTPlanner(
            start=[start_x, start_y],
            goal=[goal_x, goal_y],
            bounds=bounds,
            obstacles=self.obstacles
        )

        # Plan path
        path = self.rrt_planner.plan()

        if path is not None:
            self.get_logger().info(f'Path found with {len(path)} waypoints')
            self.publish_path(path)
            self.path = path
            self.controller.set_path(self.create_path_msg(path))
        else:
            self.get_logger().warn('No path found to goal')

    def scan_callback(self, msg):
        """Process laser scan to detect obstacles"""
        # Convert laser scan to obstacle points
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        self.obstacles = []
        for i, range_val in enumerate(msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)) and range_val < 2.0:
                angle = angle_min + i * angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                # Add as circular obstacle
                self.obstacles.append([x, y, 0.3])  # 0.3m radius

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or self.path is None:
            return

        # Update current pose in controller
        self.controller.set_current_pose(self.current_pose)

        # Compute control command
        cmd = self.controller.compute_control()

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Check if goal is reached
        if self.current_goal is not None:
            goal_dist = self.distance_to_goal()
            if goal_dist < 0.5:  # Within 0.5m of goal
                self.get_logger().info('Goal reached!')
                self.path = None
                self.current_goal = None

    def distance_to_goal(self):
        """Calculate distance to goal"""
        if self.current_pose is None or self.current_goal is None:
            return float('inf')

        dx = self.current_goal.position.x - self.current_pose.position.x
        dy = self.current_goal.position.y - self.current_pose.position.y
        return np.sqrt(dx*dx + dy*dy)

    def publish_path(self, path_points):
        """Publish path for visualization"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for point in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def create_path_msg(self, path_points):
        """Create Path message from path points"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'

        for point in path_points:
            pose = PoseStamped()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        return path_msg

def main(args=None):
    rclpy.init(args=args)
    motion_control_node = MotionControlNode()

    try:
        rclpy.spin(motion_control_node)
    except KeyboardInterrupt:
        motion_control_node.get_logger().info('Shutting down Motion Control Node')
    finally:
        motion_control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing and Validation
### Simulation Testing
1. **Launch the simulation:**
   ```bash
   ros2 launch motion_control_lab motion_control_sim_launch.py
   ```

2. **Send a navigation goal:**
   ```bash
   # In another terminal
   ros2 run motion_control_lab send_goal.py
   ```

3. **Monitor the robot's path following:**
   ```bash
   rqt_plot /motion_control/path
   ```

### Performance Evaluation
Create an evaluation script to measure controller performance:

```python
#!/usr/bin/env python3
# evaluate_controller.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import time

class ControllerEvaluator(Node):
    def __init__(self):
        super().__init__('controller_evaluator')

        self.path_sub = self.create_subscription(
            Path, '/motion_control/path', self.path_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot/ground_truth_pose', self.pose_callback, 10
        )

        self.reference_path = None
        self.tracking_errors = []
        self.execution_times = []

    def path_callback(self, msg):
        """Store reference path"""
        self.reference_path = msg.poses

    def pose_callback(self, msg):
        """Evaluate tracking performance"""
        if self.reference_path is not None:
            # Calculate distance to reference path
            current_pos = np.array([msg.pose.position.x, msg.pose.position.y])

            min_distance = float('inf')
            for pose in self.reference_path:
                path_pos = np.array([
                    pose.pose.position.x,
                    pose.pose.position.y
                ])
                dist = np.linalg.norm(current_pos - path_pos)
                min_distance = min(min_distance, dist)

            self.tracking_errors.append(min_distance)

            # Calculate statistics periodically
            if len(self.tracking_errors) % 50 == 0:
                avg_error = np.mean(self.tracking_errors[-50:])
                max_error = np.max(self.tracking_errors[-50:])
                self.get_logger().info(
                    f'Path tracking - Avg error: {avg_error:.3f}m, '
                    f'Max error: {max_error:.3f}m'
                )

def main(args=None):
    rclpy.init(args=args)
    evaluator = ControllerEvaluator()

    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        # Print final statistics
        if evaluator.tracking_errors:
            avg_error = np.mean(evaluator.tracking_errors)
            max_error = np.max(evaluator.tracking_errors)
            std_error = np.std(evaluator.tracking_errors)

            print(f'\nController Performance:')
            print(f'  Average tracking error: {avg_error:.3f}m')
            print(f'  Maximum tracking error: {max_error:.3f}m')
            print(f'  Standard deviation: {std_error:.3f}m')
            print(f'  Total samples: {len(evaluator.tracking_errors)}')

    evaluator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Features
### Dynamic Obstacle Avoidance
Implement dynamic obstacle avoidance by integrating with the costmap:

```python
#!/usr/bin/env python3
# dynamic_avoidance.py

import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class DynamicAvoidance:
    def __init__(self, safety_distance=0.5, avoidance_gain=2.0):
        self.safety_distance = safety_distance
        self.avoidance_gain = avoidance_gain

    def compute_avoidance_command(self, cmd_vel, scan_data):
        """Modify velocity command to avoid obstacles"""
        # Analyze laser scan for obstacles in front
        front_scan = scan_data.ranges[len(scan_data.ranges)//2-30:len(scan_data.ranges)//2+30]
        min_front_dist = min([r for r in front_scan if not np.isinf(r) and not np.isnan(r)], default=float('inf'))

        # If obstacle is too close, reduce forward velocity
        if min_front_dist < self.safety_distance:
            reduction_factor = min_front_dist / self.safety_distance
            cmd_vel.linear.x *= reduction_factor
            cmd_vel.linear.x = max(0.0, cmd_vel.linear.x)  # Don't go backward

        # Add lateral avoidance if needed
        left_scan = scan_data.ranges[:len(scan_data.ranges)//4]
        right_scan = scan_data.ranges[3*len(scan_data.ranges)//4:]

        min_left = min([r for r in left_scan if not np.isinf(r) and not np.isnan(r)], default=float('inf'))
        min_right = min([r for r in right_scan if not np.isinf(r) and not np.isnan(r)], default=float('inf'))

        # Turn away from closer obstacle
        if min_left < self.safety_distance or min_right < self.safety_distance:
            if min_left < min_right:
                cmd_vel.angular.z = -0.5  # Turn right
            else:
                cmd_vel.angular.z = 0.5   # Turn left

        return cmd_vel
```

## Troubleshooting
### Common Issues and Solutions
1. **Robot oscillates around path:**
   - Increase lookahead distance
   - Reduce linear velocity
   - Tune PID parameters

2. **Path planning fails:**
   - Check obstacle representation
   - Verify bounds are appropriate
   - Increase max iterations

3. **Controller is too slow:**
   - Increase controller frequency
   - Optimize path planning algorithm
   - Reduce path smoothing

4. **Robot collides with obstacles:**
   - Increase safety margins
   - Improve obstacle detection
   - Add dynamic avoidance

## Lab Deliverables
Complete the following tasks to finish the lab:

1. **Implement the RRT planner** with basic obstacle avoidance
2. **Create the pure pursuit controller** for path following
3. **Integrate planning and control** in a single node
4. **Evaluate performance** using the provided evaluation script
5. **Document your results** including:
   - Path planning success rate
   - Tracking accuracy statistics
   - Any challenges encountered and solutions
   - Suggestions for improvement

## Assessment Criteria
Your lab implementation will be assessed based on:
- **Functionality**: Does the motion control system work correctly?
- **Performance**: Are tracking errors and planning times acceptable?
- **Code Quality**: Is the code well-structured and documented?
- **Problem Solving**: How effectively did you troubleshoot issues?
- **Analysis**: Quality of performance evaluation and insights provided

## Extensions (Optional)
For advanced students, consider implementing:
- **D* or D* Lite** for dynamic replanning
- **MPC (Model Predictive Control)** for advanced control
- **Multi-robot coordination** algorithms
- **Learning-based Motion Planning** approaches

## Summary
This lab provided hands-on experience with Motion Planning and control systems using Isaac's tools. You learned to implement path planning algorithms, create controllers for path following, and integrate these components into a complete navigation system. These skills are essential for autonomous robotic applications.