---
sidebar_position: 4
---

# Navigation System Integration
## Overview
The navigation system is a critical component of the Autonomous Humanoid robot, enabling it to move safely and efficiently through complex environments. This system must handle various navigation challenges including obstacle avoidance, path planning, localization, and dynamic replanning in response to environmental changes. The navigation system integrates with other modules to provide seamless mobility as part of the complete voice-command-to-action pipeline.

The navigation system follows the ROS 2 Navigation2 framework while incorporating specialized capabilities for Humanoid Robotics applications, including integration with perception systems for enhanced environmental understanding and dynamic obstacle avoidance.

## Navigation Architecture
### Navigation System Components
The navigation system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMMAND INTERPRETATION                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Command         │  │ Intent          │  │ Goal            │ │
│  │ Reception       │  │ Classification  │  │ Transformation  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOCALIZATION SYSTEM                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Map Loading     │  │ Pose            │  │ AMCL            │ │
│  │                 │  │ Estimation      │  │ Localization    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MAPPING SYSTEM                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Occupancy       │  │ Costmap         │  │ Global Map      │ │
│  │ Mapping         │  │ Generation      │  │ Management      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PATH PLANNING                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Global Planner  │  │ Local Planner   │  │ Path            │ │
│  │ (A*)            │  │ (DWA/TEB)       │  │ Smoothing       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MOTION CONTROL                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Velocity        │  │ Obstacle        │  │ Safety          │ │
│  │ Commands        │  │ Avoidance       │  │ Monitoring      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ROBOT HARDWARE                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Wheel Encoders  │  │ IMU             │  │ Motor Control   │ │
│  │ Odometry        │  │ Sensors         │  │ Interface       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Localization System
### Robot Localization
The localization system determines the robot's position and orientation in the environment:

```python
import numpy as np
import math
from typing import Tuple, List, Optional
import threading
from dataclasses import dataclass

@dataclass
class Pose2D:
    """2D pose representation (x, y, theta)"""
    x: float
    y: float
    theta: float  # in radians

@dataclass
class Particle:
    """Particle for Monte Carlo Localization"""
    x: float
    y: float
    theta: float
    weight: float

class LocalizationSystem:
    def __init__(self, map_resolution=0.05, particle_count=1000):
        self.map_resolution = map_resolution
        self.particle_count = particle_count
        self.particles = []
        self.current_pose = Pose2D(0.0, 0.0, 0.0)
        self.map = None
        self.odom_pose = Pose2D(0.0, 0.0, 0.0)

        # Motion model parameters
        self.motion_model_params = {
            'alpha1': 0.2,  # Rotation noise from rotation
            'alpha2': 0.2,  # Rotation noise from translation
            'alpha3': 0.2,  # Translation noise from translation
            'alpha4': 0.2   # Translation noise from rotation
        }

        # Sensor model parameters
        self.sensor_model_params = {
            'z_hit': 0.8,       # Probability of hit
            'z_short': 0.1,     # Probability of short reading
            'z_max': 0.05,      # Probability of max reading
            'z_rand': 0.05,     # Probability of random reading
            'sigma_hit': 0.2,   # Standard deviation of hit
            'lambda_short': 0.1 # Decay rate of short reading
        }

        self.lock = threading.Lock()

    def initialize_particles(self, initial_pose: Pose2D, uncertainty: Tuple[float, float, float]):
        """Initialize particles with given uncertainty"""
        self.particles = []

        for _ in range(self.particle_count):
            # Add Gaussian noise to initial pose
            x = initial_pose.x + np.random.normal(0, uncertainty[0])
            y = initial_pose.y + np.random.normal(0, uncertainty[1])
            theta = initial_pose.theta + np.random.normal(0, uncertainty[2])

            # Normalize theta to [-π, π]
            theta = math.atan2(math.sin(theta), math.cos(theta))

            particle = Particle(x=x, y=y, theta=theta, weight=1.0/self.particle_count)
            self.particles.append(particle)

    def predict_motion(self, control: Tuple[float, float], dt: float):
        """Predict robot motion based on control input"""
        with self.lock:
            for particle in self.particles:
                # Extract control inputs
                delta_rot1, delta_trans, delta_rot2 = self._odometry_motion_model(
                    particle.x, particle.y, particle.theta,
                    control[0], control[1], dt
                )

                # Add noise to motion
                noise_rot1 = np.random.normal(0, self._get_rotation_noise(delta_rot1))
                noise_trans = np.random.normal(0, self._get_translation_noise(delta_trans))
                noise_rot2 = np.random.normal(0, self._get_rotation_noise(delta_rot2))

                # Update particle pose
                particle.x += (delta_trans + noise_trans) * math.cos(particle.theta + delta_rot1 + noise_rot1)
                particle.y += (delta_trans + noise_trans) * math.sin(particle.theta + delta_rot1 + noise_rot1)
                particle.theta += delta_rot1 + noise_rot1 + delta_rot2 + noise_rot2

                # Normalize theta
                particle.theta = math.atan2(math.sin(particle.theta), math.cos(particle.theta))

    def update_weights(self, sensor_data: List[float]):
        """Update particle weights based on sensor observations"""
        with self.lock:
            total_weight = 0.0

            for particle in self.particles:
                weight = self._sensor_model(particle, sensor_data)
                particle.weight = weight
                total_weight += weight

            # Normalize weights
            if total_weight > 0:
                for particle in self.particles:
                    particle.weight /= total_weight
            else:
                # If all weights are zero, reset to uniform distribution
                for particle in self.particles:
                    particle.weight = 1.0 / self.particle_count

    def resample(self):
        """Resample particles based on their weights"""
        with self.lock:
            # Calculate cumulative weights
            cumulative_weights = []
            cumsum = 0.0
            for particle in self.particles:
                cumsum += particle.weight
                cumulative_weights.append(cumsum)

            # Resample
            new_particles = []
            for _ in range(self.particle_count):
                # Sample a random value
                rand_val = np.random.random()

                # Find particle with cumulative weight >= rand_val
                for i, cum_weight in enumerate(cumulative_weights):
                    if rand_val <= cum_weight:
                        # Add particle to new set
                        new_particle = Particle(
                            x=self.particles[i].x,
                            y=self.particles[i].y,
                            theta=self.particles[i].theta,
                            weight=1.0 / self.particle_count
                        )
                        new_particles.append(new_particle)
                        break

            self.particles = new_particles

    def estimate_pose(self) -> Pose2D:
        """Estimate robot pose from particles"""
        with self.lock:
            if not self.particles:
                return self.current_pose

            # Calculate weighted average
            x_sum = 0.0
            y_sum = 0.0
            theta_sin_sum = 0.0
            theta_cos_sum = 0.0

            for particle in self.particles:
                x_sum += particle.weight * particle.x
                y_sum += particle.weight * particle.y
                theta_sin_sum += particle.weight * math.sin(particle.theta)
                theta_cos_sum += particle.weight * math.cos(particle.theta)

            estimated_x = x_sum
            estimated_y = y_sum
            estimated_theta = math.atan2(theta_sin_sum, theta_cos_sum)

            self.current_pose = Pose2D(estimated_x, estimated_y, estimated_theta)
            return self.current_pose

    def _odometry_motion_model(self, x: float, y: float, theta: float,
                              rot1: float, trans: float, rot2: float, dt: float = 1.0) -> Tuple[float, float, float]:
        """Odometry motion model for particle prediction"""
        # This is a simplified version - in practice, would use actual odometry
        return rot1, trans, rot2

    def _get_rotation_noise(self, rotation: float) -> float:
        """Get rotation noise based on motion model parameters"""
        return math.sqrt(
            self.motion_model_params['alpha1'] * rotation**2 +
            self.motion_model_params['alpha2'] * rotation**2
        )

    def _get_translation_noise(self, translation: float) -> float:
        """Get translation noise based on motion model parameters"""
        return math.sqrt(
            self.motion_model_params['alpha3'] * translation**2 +
            self.motion_model_params['alpha4'] * translation**2
        )

    def _sensor_model(self, particle: Particle, sensor_data: List[float]) -> float:
        """Calculate likelihood of sensor readings given particle pose"""
        # Simplified sensor model
        # In practice, this would use ray casting to the map
        total_likelihood = 1.0

        for i, sensor_reading in enumerate(sensor_data):
            if i < len(sensor_reading):  # Simplified for example
                # Calculate expected reading at particle's pose
                expected_reading = self._expected_sensor_reading(particle, i)

                # Calculate likelihood
                likelihood = self._gaussian_likelihood(sensor_reading, expected_reading, 0.1)
                total_likelihood *= likelihood

        return total_likelihood

    def _expected_sensor_reading(self, particle: Particle, beam_index: int) -> float:
        """Calculate expected sensor reading for a particle"""
        # Simplified - in practice, this would cast rays to the map
        return 1.0  # Default reading

    def _gaussian_likelihood(self, measurement: float, expected: float, sigma: float) -> float:
        """Calculate Gaussian likelihood"""
        diff = measurement - expected
        return math.exp(-0.5 * (diff / sigma)**2)

    def set_map(self, occupancy_map: np.ndarray, origin: Tuple[float, float]):
        """Set the occupancy map for localization"""
        self.map = occupancy_map
        self.map_origin = origin
```

## Path Planning System
### Global and Local Path Planning
```python
import heapq
from scipy.spatial import KDTree
import numpy as np
from typing import List, Tuple, Optional

class PathPlanner:
    def __init__(self, map_resolution=0.05, inflation_radius=0.5):
        self.map_resolution = map_resolution
        self.inflation_radius = inflation_radius
        self.occupancy_map = None
        self.map_origin = (0, 0)
        self.kd_tree = None

    def set_map(self, occupancy_map: np.ndarray, origin: Tuple[float, float]):
        """Set the occupancy map for path planning"""
        self.occupancy_map = occupancy_map
        self.map_origin = origin

    def plan_global_path(self, start: Pose2D, goal: Pose2D) -> Optional[List[Pose2D]]:
        """Plan global path using A* algorithm"""
        if self.occupancy_map is None:
            return None

        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid((start.x, start.y))
        goal_grid = self._world_to_grid((goal.x, goal.y))

        # Check if start and goal are valid
        if not self._is_valid_cell(start_grid) or not self._is_valid_cell(goal_grid):
            return None

        # Run A* path planning
        path_grid = self._a_star(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert grid path back to world coordinates
        path_world = [Pose2D(x, y, 0.0) for x, y in [self._grid_to_world(cell) for cell in path_grid]]

        return path_world

    def _world_to_grid(self, world_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y = world_coord
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return (grid_x, grid_y)

    def _grid_to_world(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        grid_x, grid_y = grid_coord
        world_x = self.map_origin[0] + grid_x * self.map_resolution
        world_y = self.map_origin[1] + grid_y * self.map_resolution
        return (world_x, world_y)

    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if grid cell is valid (within bounds and not occupied)"""
        x, y = cell

        # Check bounds
        if x < 0 or x >= self.occupancy_map.shape[1] or y < 0 or y >= self.occupancy_map.shape[0]:
            return False

        # Check occupancy (assuming 100 = occupied, 0 = free)
        return self.occupancy_map[y, x] < 50  # Threshold for "free"

    def _a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* path planning algorithm"""
        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Priority queue: (cost, position)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        # Closed set
        closed_set = set()

        # 8-connected neighborhood
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while open_set:
            current_cost, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start-to-goal

            closed_set.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if neighbor in closed_set:
                    continue

                if not self._is_valid_cell(neighbor):
                    continue

                # Calculate tentative g_score
                movement_cost = np.sqrt(dx**2 + dy**2)  # Diagonal movement cost
                tentative_g = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def smooth_path(self, path: List[Pose2D]) -> List[Pose2D]:
        """Smooth the path using path optimization techniques"""
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]  # Start with first point

        i = 0
        while i < len(path) - 1:
            # Try to find the furthest point that can be connected directly
            j = len(path) - 1

            while j > i + 1:
                # Check if direct path from path[i] to path[j] is collision-free
                if self._is_line_clear((path[i].x, path[i].y), (path[j].x, path[j].y)):
                    smoothed_path.append(path[j])
                    i = j
                    break
                else:
                    j -= 1

            if j == i + 1:  # No shortcut found, add next point
                smoothed_path.append(path[i + 1])
                i += 1

        return smoothed_path

    def _is_line_clear(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if line between two points is clear of obstacles"""
        # Simple Bresenham's line algorithm for collision checking
        x0, y0 = self._world_to_grid(start)
        x1, y1 = self._world_to_grid(end)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if not self._is_valid_cell((x, y)):
                return False

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True
```

## Local Path Planning and Obstacle Avoidance
### Dynamic Window Approach (DWA)
```python
class LocalPlanner:
    def __init__(self, robot_radius=0.3, max_vel_x=0.5, min_vel_x=0.0,
                 max_vel_theta=1.0, min_vel_theta=-1.0, max_acc_x=2.5, max_acc_theta=3.2,
                 dt=0.1, predict_time=1.5, heading_weight=0.8, velocity_weight=1.0,
                 obstacle_weight=1.5):
        self.robot_radius = robot_radius
        self.max_vel_x = max_vel_x
        self.min_vel_x = min_vel_x
        self.max_vel_theta = max_vel_theta
        self.min_vel_theta = min_vel_theta
        self.max_acc_x = max_acc_x
        self.max_acc_theta = max_acc_theta
        self.dt = dt
        self.predict_time = predict_time
        self.heading_weight = heading_weight
        self.velocity_weight = velocity_weight
        self.obstacle_weight = obstacle_weight

    def plan_local_path(self, current_pose: Pose2D, current_vel: Tuple[float, float],
                       goal: Pose2D, obstacles: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Plan local path using Dynamic Window Approach"""
        # Calculate velocity window
        vs = self._calc_dynamic_window(current_vel)

        # Initialize best trajectory and score
        best_traj = None
        best_score = float('-inf')

        # Evaluate all possible velocities in the window
        for vel_x in np.arange(vs[0], vs[1], 0.1):
            for vel_theta in np.arange(vs[2], vs[3], 0.05):
                # Simulate trajectory
                traj = self._predict_trajectory(current_pose, (vel_x, vel_theta))

                # Calculate scores
                heading_score = self._heading_score(traj, goal)
                velocity_score = self._velocity_score((vel_x, vel_theta))
                obstacle_score = self._obstacle_score(traj, obstacles)

                # Calculate total score
                total_score = (
                    self.heading_weight * heading_score +
                    self.velocity_weight * velocity_score -
                    self.obstacle_weight * obstacle_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_traj = (vel_x, vel_theta)

        if best_traj is None:
            # Emergency stop if no valid trajectory found
            return (0.0, 0.0)

        return best_traj

    def _calc_dynamic_window(self, current_vel: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Calculate dynamic window based on current velocity and constraints"""
        # Current velocity
        vel_x, vel_theta = current_vel

        # Calculate dynamic window
        min_vel_x = max(self.min_vel_x, vel_x - self.max_acc_x * self.dt)
        max_vel_x = min(self.max_vel_x, vel_x + self.max_acc_x * self.dt)
        min_vel_theta = max(self.min_vel_theta, vel_theta - self.max_acc_theta * self.dt)
        max_vel_theta = min(self.max_vel_theta, vel_theta + self.max_acc_theta * self.dt)

        return (min_vel_x, max_vel_x, min_vel_theta, max_vel_theta)

    def _predict_trajectory(self, start_pose: Pose2D, velocity: Tuple[float, float]) -> List[Pose2D]:
        """Predict trajectory based on constant velocity model"""
        traj = []
        pose = Pose2D(start_pose.x, start_pose.y, start_pose.theta)
        vel_x, vel_theta = velocity

        time_steps = int(self.predict_time / self.dt)

        for _ in range(time_steps):
            # Update pose using motion model
            pose.x += vel_x * math.cos(pose.theta) * self.dt
            pose.y += vel_x * math.sin(pose.theta) * self.dt
            pose.theta += vel_theta * self.dt

            # Normalize theta
            pose.theta = math.atan2(math.sin(pose.theta), math.cos(pose.theta))

            traj.append(Pose2D(pose.x, pose.y, pose.theta))

        return traj

    def _heading_score(self, trajectory: List[Pose2D], goal: Pose2D) -> float:
        """Calculate score based on heading towards goal"""
        if not trajectory:
            return 0.0

        # Calculate angle between robot orientation and goal
        final_pose = trajectory[-1]
        angle_to_goal = math.atan2(goal.y - final_pose.y, goal.x - final_pose.x)
        heading_error = abs(angle_to_goal - final_pose.theta)

        # Normalize to [0, π] and invert (higher score = better heading)
        heading_error = min(heading_error, 2 * math.pi - heading_error)
        return math.pi - heading_error

    def _velocity_score(self, velocity: Tuple[float, float]) -> float:
        """Calculate score based on forward velocity"""
        vel_x, _ = velocity
        return vel_x  # Higher forward velocity = higher score

    def _obstacle_score(self, trajectory: List[Pose2D], obstacles: List[Tuple[float, float]]) -> float:
        """Calculate score based on obstacle proximity"""
        if not trajectory or not obstacles:
            return 0.0

        min_dist = float('inf')

        for pose in trajectory:
            for obs_x, obs_y in obstacles:
                dist = math.sqrt((pose.x - obs_x)**2 + (pose.y - obs_y)**2)
                min_dist = min(min_dist, dist)

        # Return inverse of distance (closer to obstacles = worse score)
        if min_dist == float('inf'):
            return 0.0
        return 1.0 / min_dist if min_dist > 0 else float('inf')
```

## Navigation Integration
### Complete Navigation System
```python
import threading
import time
from collections import deque

class NavigationSystem:
    def __init__(self):
        # Initialize components
        self.localization = LocalizationSystem()
        self.path_planner = PathPlanner()
        self.local_planner = LocalPlanner()
        self.current_goal = None
        self.current_path = []
        self.path_index = 0
        self.is_navigating = False
        self.navigation_thread = None
        self.shutdown_flag = threading.Event()

        # State variables
        self.current_pose = Pose2D(0, 0, 0)
        self.current_velocity = (0.0, 0.0)
        self.obstacles = []

        # Navigation parameters
        self.arrival_threshold = 0.5  # meters
        self.replan_threshold = 1.0   # meters
        self.control_frequency = 10.0  # Hz

        # Threading
        self.nav_lock = threading.Lock()
        self.command_queue = deque()

    def set_map(self, occupancy_map: np.ndarray, map_origin: Tuple[float, float]):
        """Set the map for navigation"""
        self.localization.set_map(occupancy_map, map_origin)
        self.path_planner.set_map(occupancy_map, map_origin)

    def navigate_to_goal(self, goal_pose: Pose2D):
        """Start navigation to goal pose"""
        with self.nav_lock:
            if self.is_navigating:
                self.cancel_navigation()

            self.current_goal = goal_pose
            self.current_path = []
            self.path_index = 0
            self.is_navigating = True

            # Plan initial path
            start_pose = self.localization.estimate_pose()
            path = self.path_planner.plan_global_path(start_pose, goal_pose)

            if path:
                self.current_path = self.path_planner.smooth_path(path)
                self.path_index = 0
            else:
                print("Failed to plan path to goal")
                self.is_navigating = False
                return False

        # Start navigation thread
        self.navigation_thread = threading.Thread(target=self._navigation_worker)
        self.navigation_thread.daemon = True
        self.navigation_thread.start()

        return True

    def cancel_navigation(self):
        """Cancel current navigation"""
        with self.nav_lock:
            self.is_navigating = False
            self.current_goal = None
            self.current_path = []

    def _navigation_worker(self):
        """Navigation worker thread"""
        rate = 1.0 / self.control_frequency  # seconds per control cycle

        while self.is_navigating and not self.shutdown_flag.is_set():
            start_time = time.time()

            try:
                # Update current pose
                self.current_pose = self.localization.estimate_pose()

                # Check if goal reached
                if self._is_goal_reached():
                    print("Navigation goal reached!")
                    self._stop_navigation()
                    break

                # Check if replanning is needed
                if self._needs_replanning():
                    self._replan_path()

                # Execute local planning
                if self.current_path:
                    cmd_vel = self._execute_local_planning()

                    # Send velocity command to robot
                    self._send_velocity_command(cmd_vel)

            except Exception as e:
                print(f"Navigation error: {e}")
                self._stop_navigation()
                break

            # Control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, rate - elapsed)
            time.sleep(sleep_time)

    def _is_goal_reached(self) -> bool:
        """Check if robot has reached goal"""
        if not self.current_goal:
            return True

        dist_to_goal = math.sqrt(
            (self.current_pose.x - self.current_goal.x)**2 +
            (self.current_pose.y - self.current_goal.y)**2
        )

        return dist_to_goal <= self.arrival_threshold

    def _needs_replanning(self) -> bool:
        """Check if path replanning is needed"""
        if not self.current_path or self.path_index >= len(self.current_path):
            return True

        # Check if current path is too far from robot
        next_waypoint = self.current_path[min(self.path_index, len(self.current_path)-1)]
        dist_to_waypoint = math.sqrt(
            (self.current_pose.x - next_waypoint.x)**2 +
            (self.current_pose.y - next_waypoint.y)**2
        )

        return dist_to_waypoint > self.replan_threshold

    def _replan_path(self):
        """Replan path to goal"""
        if not self.current_goal:
            return

        start_pose = self.current_pose
        path = self.path_planner.plan_global_path(start_pose, self.current_goal)

        if path:
            self.current_path = self.path_planner.smooth_path(path)
            self.path_index = 0
        else:
            print("Failed to replan path to goal")

    def _execute_local_planning(self) -> Tuple[float, float]:
        """Execute local planning to follow path"""
        if not self.current_path or self.path_index >= len(self.current_path):
            return (0.0, 0.0)

        # Determine next waypoint to follow
        target_waypoint = self.current_path[min(self.path_index, len(self.current_path)-1)]

        # Update path index if close to current waypoint
        dist_to_waypoint = math.sqrt(
            (self.current_pose.x - target_waypoint.x)**2 +
            (self.current_pose.y - target_waypoint.y)**2
        )

        if dist_to_waypoint < 0.3:  # Waypoint reached threshold
            self.path_index += 1

        # Plan local trajectory to waypoint
        cmd_vel = self.local_planner.plan_local_path(
            self.current_pose,
            self.current_velocity,
            target_waypoint,
            self.obstacles
        )

        return cmd_vel

    def _send_velocity_command(self, cmd_vel: Tuple[float, float]):
        """Send velocity command to robot (ROS 2 interface would go here)"""
        # This is where ROS 2 publisher would send Twist message
        # For now, just print for simulation
        print(f"Sending velocity command: linear={cmd_vel[0]:.2f}, angular={cmd_vel[1]:.2f}")

    def _stop_navigation(self):
        """Stop navigation and send zero velocity"""
        with self.nav_lock:
            self.is_navigating = False
            self.current_goal = None
            self.current_path = []

        # Send stop command
        self._send_velocity_command((0.0, 0.0))

    def update_sensor_data(self, laser_scan: List[float], imu_data: dict,
                          odometry_data: dict, camera_data: dict = None):
        """Update sensor data for navigation"""
        # Process laser scan for obstacle detection
        if laser_scan:
            self.obstacles = self._process_laser_scan(laser_scan)

        # Update localization with sensor data
        self.localization.update_weights(laser_scan if laser_scan else [])

        # Update odometry
        if odometry_data:
            self.current_velocity = (
                odometry_data.get('linear_vel', 0.0),
                odometry_data.get('angular_vel', 0.0)
            )

    def _process_laser_scan(self, laser_scan: List[float]) -> List[Tuple[float, float]]:
        """Process laser scan to extract obstacles"""
        obstacles = []

        if not laser_scan:
            return obstacles

        # Convert laser scan to Cartesian coordinates
        angle_increment = 2 * math.pi / len(laser_scan)

        for i, range_val in enumerate(laser_scan):
            if not (np.isnan(range_val) or np.isinf(range_val)) and 0.1 < range_val < 10.0:
                angle = i * angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Transform to global coordinates using current pose
                global_x = self.current_pose.x + x * math.cos(self.current_pose.theta) - y * math.sin(self.current_pose.theta)
                global_y = self.current_pose.y + x * math.sin(self.current_pose.theta) + y * math.cos(self.current_pose.theta)

                obstacles.append((global_x, global_y))

        return obstacles

    def get_navigation_status(self) -> dict:
        """Get current navigation status"""
        return {
            'is_navigating': self.is_navigating,
            'current_pose': self.current_pose,
            'current_goal': self.current_goal,
            'path_remaining': len(self.current_path) - self.path_index if self.current_path else 0,
            'distance_to_goal': self._get_distance_to_goal()
        }

    def _get_distance_to_goal(self) -> float:
        """Get distance to goal"""
        if not self.current_goal:
            return float('inf')

        return math.sqrt(
            (self.current_pose.x - self.current_goal.x)**2 +
            (self.current_pose.y - self.current_goal.y)**2
        )

    def shutdown(self):
        """Shutdown navigation system"""
        self.shutdown_flag.set()
        self.cancel_navigation()
        if self.navigation_thread:
            self.navigation_thread.join(timeout=2.0)
```

## Integration with Voice Command System
### Navigation Command Handler
```python
class NavigationCommandHandler:
    """Handles navigation commands from voice command system"""

    def __init__(self, navigation_system: NavigationSystem):
        self.nav_system = navigation_system
        self.location_database = self._initialize_location_database()
        self.default_locations = {
            'kitchen': Pose2D(5.0, 3.0, 0.0),
            'bedroom': Pose2D(-2.0, 4.0, 0.0),
            'living room': Pose2D(0.0, 0.0, 0.0),
            'office': Pose2D(-1.0, -3.0, 0.0),
            'bathroom': Pose2D(3.0, -2.0, 0.0)
        }

    def handle_navigation_command(self, entities: dict) -> dict:
        """Handle navigation command with entities"""
        destination = entities.get('location', '').lower()

        # Look up destination in database
        target_pose = self._lookup_destination(destination)

        if target_pose is None:
            return {
                'success': False,
                'message': f"Unknown destination: {destination}",
                'action': 'request_clarification',
                'available_destinations': list(self.default_locations.keys())
            }

        # Attempt to navigate
        success = self.nav_system.navigate_to_goal(target_pose)

        if success:
            return {
                'success': True,
                'message': f"Navigating to {destination}",
                'action': 'navigation_started',
                'destination': destination,
                'target_pose': (target_pose.x, target_pose.y)
            }
        else:
            return {
                'success': False,
                'message': f"Could not navigate to {destination}",
                'action': 'navigation_failed',
                'destination': destination
            }

    def _lookup_destination(self, location_name: str) -> Optional[Pose2D]:
        """Look up destination in location database"""
        # Try exact match first
        if location_name in self.default_locations:
            return self.default_locations[location_name]

        # Try fuzzy matching
        for known_location in self.default_locations:
            if location_name in known_location or known_location in location_name:
                return self.default_locations[known_location]

        # If no match found, return None
        return None

    def _initialize_location_database(self) -> dict:
        """Initialize location database with learned locations"""
        # In a real system, this would load from persistent storage
        return self.default_locations

    def add_location(self, name: str, pose: Pose2D):
        """Add a new location to the database"""
        self.default_locations[name] = pose

    def get_known_locations(self) -> List[str]:
        """Get list of known locations"""
        return list(self.default_locations.keys())
```

## Safety and Recovery
### Navigation Safety Systems
```python
class NavigationSafetyManager:
    """Manages safety aspects of navigation"""

    def __init__(self, navigation_system: NavigationSystem):
        self.nav_system = navigation_system
        self.emergency_stop_active = False
        self.safety_zones = []
        self.collision_threshold = 0.5  # meters
        self.emergency_stop_distance = 0.3  # meters
        self.recovery_behaviors = {
            'stuck': self._recover_from_stuck,
            'collision': self._recover_from_collision,
            'lost': self._recover_from_lost
        }

    def check_safety_conditions(self, obstacles: List[Tuple[float, float]]) -> dict:
        """Check safety conditions and return any violations"""
        safety_report = {
            'is_safe': True,
            'violations': [],
            'actions': []
        }

        # Check for emergency stop conditions
        if self._is_emergency_stop_needed(obstacles):
            safety_report['is_safe'] = False
            safety_report['violations'].append('OBSTACLE_TOO_CLOSE')
            safety_report['actions'].append('EMERGENCY_STOP')

        # Check for safety zone violations
        for zone in self.safety_zones:
            if self._is_in_safety_zone(zone):
                safety_report['is_safe'] = False
                safety_report['violations'].append(f'SAFETY_ZONE_VIOLATION: {zone["name"]}')
                safety_report['actions'].append('AVOID_ZONE')

        # Check for human presence
        if self._is_human_too_close(obstacles):
            safety_report['is_safe'] = False
            safety_report['violations'].append('HUMAN_TOO_CLOSE')
            safety_report['actions'].append('SLOW_DOWN')

        return safety_report

    def _is_emergency_stop_needed(self, obstacles: List[Tuple[float, float]]) -> bool:
        """Check if emergency stop is needed due to close obstacles"""
        if not obstacles:
            return False

        # Find closest obstacle
        closest_dist = float('inf')
        for obs_x, obs_y in obstacles:
            dist = math.sqrt(obs_x**2 + obs_y**2)  # Relative to robot
            closest_dist = min(closest_dist, dist)

        return closest_dist < self.emergency_stop_distance

    def _is_human_too_close(self, obstacles: List[Tuple[float, float]]) -> bool:
        """Check if humans are too close (simplified human detection)"""
        # In a real system, this would use human detection algorithms
        # For now, assume obstacles within certain range might be humans
        if not obstacles:
            return False

        for obs_x, obs_y in obstacles:
            dist = math.sqrt(obs_x**2 + obs_y**2)
            if dist < 1.0:  # Within 1 meter
                return True

        return False

    def _is_in_safety_zone(self, zone: dict) -> bool:
        """Check if robot is in a safety zone"""
        # Simplified implementation
        current_pose = self.nav_system.current_pose
        zone_center = zone['center']
        zone_radius = zone['radius']

        dist = math.sqrt(
            (current_pose.x - zone_center[0])**2 +
            (current_pose.y - zone_center[1])**2
        )

        return dist <= zone_radius

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.nav_system._send_velocity_command((0.0, 0.0))

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False

    def add_safety_zone(self, name: str, center: Tuple[float, float], radius: float,
                       zone_type: str = 'restricted'):
        """Add a safety zone"""
        zone = {
            'name': name,
            'center': center,
            'radius': radius,
            'type': zone_type
        }
        self.safety_zones.append(zone)

    def _recover_from_stuck(self):
        """Recovery behavior for stuck robot"""
        print("Attempting recovery from stuck condition...")
        # Implement backing up and turning
        self.nav_system._send_velocity_command((-0.2, 0.5))  # Back up and turn
        time.sleep(2.0)
        self.nav_system._send_velocity_command((0.0, 0.0))   # Stop

    def _recover_from_collision(self):
        """Recovery behavior for collision"""
        print("Collision detected, attempting recovery...")
        # Implement collision recovery
        self.nav_system._send_velocity_command((-0.3, 0.0))  # Back up
        time.sleep(1.5)
        self.nav_system._send_velocity_command((0.0, 0.0))   # Stop

    def _recover_from_lost(self):
        """Recovery behavior for lost localization"""
        print("Localization lost, attempting recovery...")
        # Implement relocalization strategies
        # For now, just stop and request assistance
        self.nav_system._send_velocity_command((0.0, 0.0))
```

The navigation system integration provides comprehensive capabilities for Autonomous Humanoid robot navigation, including localization, path planning, obstacle avoidance, and safety management. This system integrates seamlessly with the voice command processing and planning systems to enable natural language navigation commands.