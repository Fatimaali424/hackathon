---
sidebar_position: 3
---

# Motion Planning & Trajectory Generation
## Overview
Motion Planning is the computational process of determining a sequence of valid configurations that move a robot from an initial state to a goal state while avoiding obstacles and satisfying various constraints. This chapter explores Motion Planning algorithms and trajectory generation techniques within the NVIDIA Isaac framework, emphasizing GPU-accelerated computation for real-time applications.

Motion Planning bridges the gap between perception and action, taking environmental understanding and generating executable motion commands for Robotic Systems.

## Motion Planning Fundamentals
### Configuration Space (C-Space)
Motion Planning operates in configuration space, where each point represents a complete robot configuration:

- **Free space (C_free)**: Configurations where the robot doesn't collide with obstacles
- **Obstacle space (C_obs)**: Configurations where the robot collides with obstacles
- **Path planning**: Finding a continuous path in C_free from start to goal

For a robot with n degrees of freedom, the configuration space has n dimensions, making planning computationally challenging.

### Planning Problem Definition
A Motion Planning problem consists of:

- **Robot model**: Kinematic and dynamic properties
- **Environment**: Static and dynamic obstacles
- **Start state**: Initial configuration and velocity
- **Goal state**: Desired configuration and velocity
- **Constraints**: Kinematic, dynamic, and environmental constraints
- **Optimization criteria**: Path length, time, energy, safety

## Sampling-Based Motion Planning
### Probabilistic Roadmap (PRM)
PRM pre-computes a roadmap of possible paths:

```python
import numpy as np
from scipy.spatial import KDTree
import random

class ProbabilisticRoadmap:
    def __init__(self, robot, environment, num_samples=1000):
        self.robot = robot
        self.environment = environment
        self.samples = []
        self.graph = {}
        self.kdtree = None
        self.num_samples = num_samples

    def build_roadmap(self):
        # Sample free configurations
        for _ in range(self.num_samples):
            q = self.sample_free_configuration()
            if q is not None:
                self.samples.append(q)

        # Build k-d tree for nearest neighbor queries
        self.kdtree = KDTree(self.samples)

        # Connect nearby configurations
        for i, q in enumerate(self.samples):
            neighbors = self.get_neighbors(q, max_dist=1.0)
            for j in neighbors:
                if self.is_collision_free(q, self.samples[j]):
                    if i not in self.graph:
                        self.graph[i] = []
                    if j not in self.graph:
                        self.graph[j] = []
                    self.graph[i].append(j)
                    self.graph[j].append(i)

    def sample_free_configuration(self):
        # Sample configuration and check for collision
        for _ in range(100):  # Max attempts
            q = self.robot.sample_configuration()
            if not self.environment.in_collision(self.robot, q):
                return q
        return None

    def get_neighbors(self, q, max_dist):
        # Find nearby configurations
        indices = self.kdtree.query_ball_point(q, max_dist)
        return [i for i in indices if i != self.kdtree.query(q)[1]]
```

### Rapidly-exploring Random Trees (RRT)
RRT incrementally builds a tree of possible paths:

```python
class RRT:
    def __init__(self, start, goal, environment, robot):
        self.start = start
        self.goal = goal
        self.environment = environment
        self.robot = robot
        self.vertices = [start]
        self.edges = {}
        self.goal_bias = 0.1

    def plan(self, max_iterations=10000):
        for i in range(max_iterations):
            # Sample random configuration
            if random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self.sample_configuration()

            # Find nearest vertex in tree
            q_near = self.nearest_vertex(q_rand)

            # Extend tree toward random configuration
            q_new = self.extend(q_near, q_rand)

            if q_new is not None:
                self.add_vertex(q_new, q_near)

                # Check if goal is reached
                if self.is_goal_reached(q_new):
                    return self.reconstruct_path(q_new)

        return None  # No path found

    def extend(self, q_from, q_to):
        # Extend tree from q_from toward q_to
        direction = np.array(q_to) - np.array(q_from)
        distance = np.linalg.norm(direction)

        if distance > self.max_step_size:
            direction = direction / distance * self.max_step_size
            q_new = tuple(np.array(q_from) + direction)
        else:
            q_new = q_to

        # Check for collision along path
        if self.is_collision_free(q_from, q_new):
            return q_new
        return None
```

### GPU-Accelerated Sampling
Isaac leverages GPU acceleration for Motion Planning:

- **Parallel collision checking**: Check multiple configurations simultaneously
- **Batch nearest neighbor queries**: Process multiple queries in parallel
- **Parallel path validation**: Validate multiple potential paths concurrently

## Trajectory Generation
### Polynomial Trajectories
Generate smooth trajectories using polynomial interpolation:

```python
import numpy as np
from scipy.optimize import minimize

class PolynomialTrajectory:
    def __init__(self, degree=5):
        self.degree = degree
        self.coefficients = None

    def generate_trajectory(self, start_state, end_state, duration):
        # Define boundary conditions
        # Position: q(0) = q_start, q(T) = q_end
        # Velocity: q_dot(0) = v_start, q_dot(T) = v_end
        # Acceleration: q_ddot(0) = a_start, q_ddot(T) = a_end

        # Set up system of equations
        t0, t1 = 0, duration
        q0, q1 = start_state[0], end_state[0]
        v0, v1 = start_state[1], end_state[1]
        a0, a1 = start_state[2], end_state[2]

        # Build constraint matrix
        A = np.array([
            [1, t0, t0**2, t0**3, t0**4, t0**5],      # q(0) = q0
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4], # q_dot(0) = v0
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],     # q_ddot(0) = a0
            [1, t1, t1**2, t1**3, t1**4, t1**5],      # q(T) = q1
            [0, 1, 2*t1, 3*t1**2, 4*t1**3, 5*t1**4], # q_dot(T) = v1
            [0, 0, 2, 6*t1, 12*t1**2, 20*t1**3]      # q_ddot(T) = a1
        ])

        b = np.array([q0, v0, a0, q1, v1, a1])

        # Solve for coefficients
        self.coefficients = np.linalg.solve(A, b)
        return self.coefficients

    def evaluate(self, t):
        # Evaluate trajectory at time t
        if self.coefficients is None:
            raise ValueError("Trajectory not generated")

        result = 0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (t ** i)
        return result

    def evaluate_derivative(self, t, order=1):
        # Evaluate derivative of trajectory at time t
        if self.coefficients is None:
            raise ValueError("Trajectory not generated")

        result = 0
        for i in range(order, len(self.coefficients)):
            coeff = self.coefficients[i]
            # Calculate derivative coefficient
            for j in range(order):
                coeff *= (i - j)
            result += coeff * (t ** (i - order))
        return result
```

### Optimization-Based Trajectory Generation
Use optimization to generate trajectories that minimize cost functions:

```python
from scipy.optimize import minimize
import numpy as np

class OptimizedTrajectory:
    def __init__(self, environment, robot):
        self.environment = environment
        self.robot = robot
        self.waypoints = None

    def generate_trajectory(self, start, goal, num_waypoints=10):
        # Optimize trajectory to minimize cost function
        # Cost function includes: path length, obstacle clearance, smoothness

        def cost_function(waypoints_flat):
            # Reshape flat array to waypoints
            waypoints = waypoints_flat.reshape(-1, len(start))

            # Calculate path length
            path_length = 0
            for i in range(1, len(waypoints)):
                path_length += np.linalg.norm(waypoints[i] - waypoints[i-1])

            # Calculate obstacle clearance penalty
            clearance_penalty = 0
            for waypoint in waypoints:
                dist_to_obstacle = self.environment.distance_to_obstacle(waypoint)
                if dist_to_obstacle < 0.5:  # Within 0.5m of obstacle
                    clearance_penalty += (0.5 - dist_to_obstacle) ** 2

            # Calculate smoothness penalty
            smoothness_penalty = 0
            for i in range(1, len(waypoints) - 1):
                prev_to_curr = waypoints[i] - waypoints[i-1]
                curr_to_next = waypoints[i+1] - waypoints[i]
                angle_deviation = np.arccos(
                    np.clip(np.dot(prev_to_curr, curr_to_next) /
                           (np.linalg.norm(prev_to_curr) * np.linalg.norm(curr_to_next)), -1, 1)
                )
                smoothness_penalty += angle_deviation ** 2

            return path_length + 10 * clearance_penalty + 5 * smoothness_penalty

        # Initial guess: straight line from start to goal
        initial_waypoints = np.linspace(start, goal, num_waypoints)
        initial_flat = initial_waypoints.flatten()

        # Optimize
        result = minimize(
            cost_function,
            initial_flat,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )

        self.waypoints = result.x.reshape(-1, len(start))
        return self.waypoints
```

## Isaac Motion Planning Components
### Isaac ROS Navigation
Isaac ROS provides GPU-accelerated navigation components:

#### GPU-Accelerated Path Planning
- **CUDA-based A***: Parallel A* implementation for grid-based planning
- **GPU Dijkstra**: Accelerated Dijkstra algorithm for weighted graphs
- **Voronoi-based planning**: GPU-accelerated Voronoi diagram computation

#### Trajectory Optimization
- **MPC (Model Predictive Control)**: GPU-accelerated receding horizon control
- **Nonlinear optimization**: Real-time trajectory optimization
- **Dynamic obstacle avoidance**: Reactive planning for moving obstacles

### Isaac Navigation 2 (Nav2) Integration
Isaac extends ROS 2 Navigation with GPU acceleration:

```python
# Isaac-accelerated navigation example
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from isaac_ros_nav_interfaces.srv import ComputePathToPose

class IsaacPathPlanner(Node):
    def __init__(self):
        super().__init__('isaac_path_planner')

        # Publishers and subscribers
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Service server for path computation
        self.compute_path_srv = self.create_service(
            ComputePathToPose,
            '/compute_path_to_pose',
            self.compute_path_callback
        )

        # Isaac-accelerated planner
        self.planner = self.initialize_isaac_planner()

    def initialize_isaac_planner(self):
        # Initialize GPU-accelerated planner
        # This would use Isaac's CUDA-based planning algorithms
        pass

    def compute_path_callback(self, request, response):
        try:
            # Use Isaac GPU-accelerated planning
            path = self.planner.plan(
                start=request.start,
                goal=request.goal,
                map_resolution=0.05  # 5cm resolution
            )

            response.path = path
            response.error_code = 0
            response.error_message = "Success"

            # Publish path for visualization
            self.path_pub.publish(path)

        except Exception as e:
            response.error_code = -1
            response.error_message = f"Planning failed: {str(e)}"

        return response

    def scan_callback(self, msg):
        # Update local costmap with laser scan data
        # This would use Isaac's GPU-accelerated costmap updates
        pass
```

## Dynamic Motion Planning
### Time-Varying Environments
Handle environments with moving obstacles:

```python
class DynamicMotionPlanner:
    def __init__(self, environment):
        self.environment = environment
        self.predicted_obstacles = {}  # Obstacle trajectories
        self.replanning_threshold = 0.5  # Replan if obstacle within 0.5m

    def plan_with_prediction(self, start, goal, current_time):
        # Predict obstacle positions at future times
        predicted_environment = self.predict_environment(current_time)

        # Plan in predicted environment
        path = self.plan_static(predicted_environment, start, goal)

        # Monitor for replanning opportunities
        self.schedule_replanning(path, current_time)

        return path

    def predict_environment(self, current_time):
        # Predict obstacle positions using motion models
        future_environment = self.environment.copy()

        for obstacle_id, trajectory in self.predicted_obstacles.items():
            predicted_pos = trajectory.predict(current_time)
            future_environment.update_obstacle_position(obstacle_id, predicted_pos)

        return future_environment

    def schedule_replanning(self, path, current_time):
        # Check if replanning is needed based on new sensor data
        def replan_if_needed():
            current_obstacles = self.get_current_obstacles()
            if self.path_is_invalid(path, current_obstacles):
                new_path = self.plan_with_prediction(
                    path.poses[0].pose,  # Current position
                    path.poses[-1].pose,  # Goal position
                    current_time
                )
                return new_path
        return replan_if_needed
```

### Reactive Planning
Combine global planning with local reactive behaviors:

```python
class ReactivePlanner:
    def __init__(self, global_planner, local_planner):
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.current_path = None
        self.lookahead_distance = 1.0

    def plan(self, robot_state, goal, sensor_data):
        # Check if global replanning is needed
        if self.should_replan(robot_state, goal):
            self.current_path = self.global_planner.plan(
                robot_state.pose, goal
            )

        # Generate local trajectory using local planner
        local_goal = self.get_local_goal(robot_state)
        local_trajectory = self.local_planner.plan(
            robot_state, local_goal, sensor_data
        )

        return local_trajectory

    def should_replan(self, robot_state, goal):
        # Replan if too far from global path or goal changed significantly
        if self.current_path is None:
            return True

        distance_to_path = self.compute_distance_to_path(
            robot_state.pose, self.current_path
        )

        return distance_to_path > self.replanning_threshold
```

## Trajectory Execution and Control
### Feedback Control for Trajectory Following
Implement control systems to follow planned trajectories:

```python
class TrajectoryFollower:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.current_trajectory = None
        self.current_index = 0
        self.lookahead_distance = 0.5
        self.kp = 2.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.05 # Derivative gain

    def follow_trajectory(self, current_state, trajectory):
        # Find closest point on trajectory
        closest_idx = self.find_closest_point(current_state, trajectory)

        # Determine lookahead point
        lookahead_idx = self.find_lookahead_point(
            current_state, trajectory, closest_idx
        )

        # Calculate control command
        control_cmd = self.compute_control(
            current_state, trajectory, lookahead_idx
        )

        return control_cmd

    def find_lookahead_point(self, current_state, trajectory, start_idx):
        # Find point on trajectory at lookahead distance
        current_pos = np.array([current_state.pose.position.x,
                               current_state.pose.position.y])

        for i in range(start_idx, len(trajectory.poses)):
            point = np.array([
                trajectory.poses[i].pose.position.x,
                trajectory.poses[i].pose.position.y
            ])

            distance = np.linalg.norm(point - current_pos)
            if distance >= self.lookahead_distance:
                return i

        # If no point is far enough, return the last point
        return len(trajectory.poses) - 1

    def compute_control(self, current_state, trajectory, target_idx):
        # Pure pursuit controller
        target_point = trajectory.poses[target_idx]

        # Transform target point to robot frame
        robot_pos = np.array([current_state.pose.position.x,
                             current_state.pose.position.y])
        target_pos = np.array([target_point.pose.position.x,
                              target_point.pose.position.y])

        # Calculate heading error
        target_vector = target_pos - robot_pos
        robot_heading = current_state.pose.orientation  # Convert to angle

        # Calculate control command based on heading error
        # This would involve more complex calculations in practice
        linear_vel = min(0.5, np.linalg.norm(target_vector))
        angular_vel = self.calculate_angular_command(
            target_vector, robot_heading
        )

        return linear_vel, angular_vel
```

## Performance Optimization
### GPU-Accelerated Planning
Leverage Isaac's GPU acceleration for Motion Planning:

#### Parallel Collision Detection
```python
# CUDA kernel for parallel collision detection (conceptual)
"""
__global__ void check_collisions_kernel(
    float* configurations,
    int num_configs,
    float* obstacle_data,
    int num_obstacles,
    bool* collision_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_configs) return;

    // Check collision for configuration at idx
    collision_results[idx] = is_collision_free(
        configurations + idx * CONFIG_DIM,
        obstacle_data,
        num_obstacles
    );
}
"""
```

#### Batch Processing
Process multiple planning requests simultaneously:

- **Temporal batching**: Combine multiple time steps
- **Spatial batching**: Process multiple robot configurations
- **Multi-query planning**: Share computation across multiple queries

### Memory Management
Efficient memory usage for real-time planning:

- **Memory pools**: Pre-allocate frequently used structures
- **GPU memory optimization**: Minimize transfers between CPU and GPU
- **Streaming**: Process data in chunks rather than all at once

## Integration with Perception Systems
### Perception-Planning Coupling
Motion Planning relies on perception data:

```python
class PerceptionAwarePlanner:
    def __init__(self, motion_planner, perception_system):
        self.motion_planner = motion_planner
        self.perception = perception_system
        self.uncertainty_model = self.initialize_uncertainty_model()

    def plan_with_uncertainty(self, start, goal, confidence_threshold=0.8):
        # Get perception data with uncertainty estimates
        obstacles = self.perception.get_detected_obstacles()
        uncertainty_map = self.perception.get_uncertainty_map()

        # Plan considering perception uncertainty
        safe_path = self.motion_planner.plan_with_uncertainty(
            start, goal, obstacles, uncertainty_map, confidence_threshold
        )

        return safe_path

    def update_environment_model(self, sensor_data):
        # Update environment model based on new sensor data
        new_obstacles = self.perception.process_sensor_data(sensor_data)

        # Update motion planner with new information
        self.motion_planner.update_environment(new_obstacles)

        # Potentially replan if environment changed significantly
        if self.environment_changed(new_obstacles):
            self.request_replanning()
```

## Quality Assurance and Validation
### Planning Quality Metrics
Evaluate Motion Planning performance:

- **Completeness**: Does the planner find a solution when one exists?
- **Optimality**: How close is the solution to optimal?
- **Computational efficiency**: Planning time and memory usage
- **Solution quality**: Path length, smoothness, safety margin

### Simulation-Based Validation
Use Isaac Sim for planning validation:

- **Synthetic environments**: Test in diverse simulated scenarios
- **Stress testing**: Validate performance under extreme conditions
- **Regression testing**: Ensure updates don't degrade planning quality
- **Edge case validation**: Test rare but critical scenarios

## Summary
Motion Planning and trajectory generation form the bridge between perception and action in Robotic Systems. The NVIDIA Isaac platform provides GPU-accelerated tools that enable real-time planning for complex robotic applications, from simple mobile robots to sophisticated manipulation systems.

The combination of sampling-based planners, optimization techniques, and GPU acceleration allows robots to navigate complex, dynamic environments while satisfying kinematic and dynamic constraints. In the next chapter, we'll explore how these planning systems integrate with edge deployment strategies to create efficient, real-time Robotic Systems.