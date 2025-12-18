#!/usr/bin/env python3
"""
RRT Motion Planning Example for Isaac-based Robotics

This script demonstrates the implementation of a Rapidly-exploring Random Tree (RRT)
algorithm, which is commonly used in robotics for path planning. This is essential
for the AI-robot brain's navigation capabilities.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time


class RRTPlanner:
    """
    RRT (Rapidly-exploring Random Tree) Planner Implementation
    This is a sampling-based motion planning algorithm suitable for high-dimensional spaces
    """

    def __init__(self, start, goal, bounds, obstacles=None, step_size=1.0, max_iter=10000):
        """
        Initialize the RRT planner

        Args:
            start: Starting configuration [x, y]
            goal: Goal configuration [x, y]
            bounds: Environment bounds [(min_x, max_x), (min_y, max_y)]
            obstacles: List of obstacles (circles: [x, y, radius] or rectangles: [x, y, width, height])
            step_size: Maximum step size for tree extension
            max_iter: Maximum number of iterations
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.bounds = bounds
        self.obstacles = obstacles or []
        self.step_size = step_size
        self.max_iter = max_iter

        # RRT tree: {node_id: (configuration, parent_id)}
        self.tree = {0: (self.start, None)}
        self.nodes = [self.start]  # List of configurations
        self.node_count = 1

        # Parameters
        self.goal_bias = 0.1  # Probability of sampling goal
        self.goal_tolerance = 0.5  # Distance tolerance to goal

    def plan(self, visualize=False):
        """
        Plan a path from start to goal using RRT algorithm

        Args:
            visualize: Whether to visualize the planning process

        Returns:
            Path as a list of configurations, or None if no path found
        """
        if visualize:
            self._setup_visualization()

        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self._sample_free_space()

            # Find nearest node in tree
            nearest_idx = self._nearest_node(q_rand)

            # Extend tree toward random point
            q_new = self._extend_toward(nearest_idx, q_rand)

            if q_new is not None:
                # Add new node to tree
                self.tree[self.node_count] = (q_new, nearest_idx)
                self.nodes.append(q_new)

                if visualize and i % 50 == 0:  # Visualize every 50 iterations
                    self._update_visualization()

                # Check if goal is reached
                if self._distance(q_new, self.goal) < self.goal_tolerance:
                    path = self._reconstruct_path(self.node_count)
                    if visualize:
                        self._finalize_visualization(path)
                    return path

                self.node_count += 1

        # If we get here, no path was found
        if visualize:
            plt.title("RRT Planning - No Path Found")
            plt.show()
        return None

    def _sample_free_space(self):
        """Sample a random configuration in free space"""
        while True:
            # Sample random point in bounds
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            q_rand = np.array([x, y])

            # Check if point is collision-free
            if self._is_collision_free(q_rand):
                return q_rand

    def _nearest_node(self, q_rand):
        """Find nearest node in tree to q_rand using KDTree for efficiency"""
        if len(self.nodes) == 1:
            return 0

        # Use KDTree for efficient nearest neighbor search
        tree = KDTree(self.nodes)
        distance, idx = tree.query(q_rand)
        return idx

    def _extend_toward(self, nearest_idx, q_rand):
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
        if self._is_path_collision_free(q_nearest, q_new):
            return q_new

        return None

    def _is_collision_free(self, q):
        """Check if configuration q is collision-free"""
        for obs in self.obstacles:
            if self._point_in_obstacle(q, obs):
                return False
        return True

    def _is_path_collision_free(self, q_from, q_to):
        """Check if path from q_from to q_to is collision-free using multiple samples"""
        # Simple linear interpolation check with multiple samples
        steps = max(1, int(np.linalg.norm(q_to - q_from) / 0.2))  # Check every 0.2m
        for i in range(1, steps + 1):
            t = i / steps
            q_check = q_from + t * (q_to - q_from)
            if not self._is_collision_free(q_check):
                return False
        return True

    def _point_in_obstacle(self, point, obstacle):
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

    def _reconstruct_path(self, goal_node_idx):
        """Reconstruct path from goal to start"""
        path = []
        current_idx = goal_node_idx

        while current_idx is not None:
            config = self.tree[current_idx][0]
            path.append(config.copy())
            current_idx = self.tree[current_idx][1]

        return path[::-1]  # Reverse to get start->goal path

    def _distance(self, q1, q2):
        """Calculate Euclidean distance between configurations"""
        return np.linalg.norm(q1 - q2)

    def _setup_visualization(self):
        """Setup visualization for planning process"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        self.ax.set_title("RRT Planning Process")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Draw obstacles
        for obs in self.obstacles:
            if len(obs) == 3:  # Circle
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
                self.ax.add_patch(circle)
            elif len(obs) == 4:  # Rectangle
                rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], color='red', alpha=0.3)
                self.ax.add_patch(rect)

        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        self.ax.legend()

    def _update_visualization(self):
        """Update visualization during planning"""
        if hasattr(self, 'fig'):
            # Clear previous tree visualization
            for line in self.ax.lines[2:]:  # Keep start/goal points
                line.remove()

            # Draw tree edges
            for node_id, (config, parent_id) in self.tree.items():
                if parent_id is not None:
                    parent_config = self.tree[parent_id][0]
                    self.ax.plot([parent_config[0], config[0]],
                                [parent_config[1], config[1]], 'b-', alpha=0.3, linewidth=0.5)

            # Draw current tree nodes
            nodes_array = np.array(self.nodes)
            if len(nodes_array) > 0:
                self.ax.scatter(nodes_array[:, 0], nodes_array[:, 1],
                              c='blue', s=1, alpha=0.6)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def _finalize_visualization(self, path):
        """Finalize visualization with the found path"""
        if hasattr(self, 'fig'):
            # Draw the final path
            if path:
                path_array = np.array(path)
                self.ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=3, label='Path')
                self.ax.scatter(path_array[:, 0], path_array[:, 1], c='green', s=20, zorder=5)

            self.ax.set_title("RRT Planning - Path Found!")
            self.ax.legend()
            plt.ioff()
            plt.show()


class RRTStarPlanner(RRTPlanner):
    """
    RRT* (Optimal RRT) Planner - Extension of RRT that provides asymptotic optimality
    This version includes rewiring to find better paths over time
    """

    def __init__(self, start, goal, bounds, obstacles=None, step_size=1.0, max_iter=10000, rewire_radius=2.0):
        super().__init__(start, goal, bounds, obstacles, step_size, max_iter)
        self.rewire_radius = rewire_radius

    def plan(self, visualize=False):
        """
        Plan a path using RRT* algorithm with rewiring for optimality
        """
        if visualize:
            self._setup_visualization()

        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_bias:
                q_rand = self.goal
            else:
                q_rand = self._sample_free_space()

            # Find nearest node in tree
            nearest_idx = self._nearest_node(q_rand)

            # Extend tree toward random point
            q_new = self._extend_toward(nearest_idx, q_rand)

            if q_new is not None:
                # Find parent with minimum cost
                min_cost_parent_idx = self._find_min_cost_parent(q_new)

                if min_cost_parent_idx is not None:
                    # Add new node to tree with optimal parent
                    self.tree[self.node_count] = (q_new, min_cost_parent_idx)
                    self.nodes.append(q_new)

                    # Rewire nearby nodes if new path is better
                    self._rewire_nearby_nodes(self.node_count, q_new)

                    if visualize and i % 50 == 0:
                        self._update_visualization()

                    # Check if goal is reached
                    if self._distance(q_new, self.goal) < self.goal_tolerance:
                        path = self._reconstruct_path(self.node_count)
                        if visualize:
                            self._finalize_visualization(path)
                        return path

                    self.node_count += 1

        # No path found
        if visualize:
            plt.title("RRT* Planning - No Path Found")
            plt.show()
        return None

    def _find_min_cost_parent(self, q_new):
        """Find the parent node that provides minimum cost to reach q_new"""
        # Find all nodes within rewire radius
        nearby_nodes = self._get_nearby_nodes(q_new, self.rewire_radius)

        min_cost = float('inf')
        min_cost_parent = None

        for node_idx in nearby_nodes:
            config = self.tree[node_idx][0]
            # Check if path from this node to new config is collision-free
            if self._is_path_collision_free(config, q_new):
                # Calculate cost (distance from start to this node + distance to new node)
                path_cost = self._get_path_cost(node_idx) + self._distance(config, q_new)
                if path_cost < min_cost:
                    min_cost = path_cost
                    min_cost_parent = node_idx

        return min_cost_parent

    def _get_path_cost(self, node_idx):
        """Get the cost (path length) from start to given node"""
        cost = 0.0
        current_idx = node_idx
        parent_idx = self.tree[current_idx][1]

        while parent_idx is not None:
            current_config = self.tree[current_idx][0]
            parent_config = self.tree[parent_idx][0]
            cost += self._distance(current_config, parent_config)
            current_idx = parent_idx
            parent_idx = self.tree[current_idx][1]

        return cost

    def _get_nearby_nodes(self, q_new, radius):
        """Get all nodes within specified radius of q_new"""
        nearby = []
        for node_idx, (config, _) in self.tree.items():
            if self._distance(config, q_new) <= radius:
                nearby.append(node_idx)
        return nearby

    def _rewire_nearby_nodes(self, new_node_idx, new_config):
        """Rewire nearby nodes if going through new node is better"""
        nearby_nodes = self._get_nearby_nodes(new_config, self.rewire_radius)

        for node_idx in nearby_nodes:
            if node_idx == new_node_idx:  # Skip the new node itself
                continue

            node_config = self.tree[node_idx][0]
            # Check if path from new node to this node is collision-free
            if self._is_path_collision_free(new_config, node_config):
                # Calculate new potential cost
                new_cost = (self._get_path_cost(new_node_idx) +
                           self._distance(new_config, node_config))

                # If new cost is better than current cost, rewire
                current_cost = self._get_path_cost(node_idx)
                if new_cost < current_cost:
                    # Update parent of this node to new node
                    old_parent = self.tree[node_idx][1]
                    self.tree[node_idx] = (node_config, new_node_idx)


def create_sample_environment():
    """
    Create a sample environment with obstacles for testing
    """
    bounds = [(-10, 10), (-10, 10)]  # x: -10 to 10, y: -10 to 10

    # Define obstacles: circles [x, y, radius] and rectangles [x, y, width, height]
    obstacles = [
        # Circular obstacles
        [2, 2, 1.5],      # Circle at (2,2) with radius 1.5
        [-3, -3, 1.2],    # Circle at (-3,-3) with radius 1.2
        [5, -2, 1.0],     # Circle at (5,-2) with radius 1.0

        # Rectangular obstacles
        [-1, 3, 2, 1],    # Rectangle from (-1,3) with width 2, height 1
        [3, 5, 1.5, 2],   # Rectangle from (3,5) with width 1.5, height 2
        [-5, 1, 1, 3],    # Rectangle from (-5,1) with width 1, height 3
    ]

    return bounds, obstacles


def benchmark_planners():
    """
    Benchmark different planning approaches
    """
    print("Benchmarking RRT vs RRT* planners...")

    bounds, obstacles = create_sample_environment()
    start = [-8, -8]
    goal = [8, 8]

    # Test RRT
    print("\nTesting RRT planner...")
    rrt_planner = RRTPlanner(start, goal, bounds, obstacles, step_size=0.5, max_iter=2000)
    start_time = time.time()
    rrt_path = rrt_planner.plan()
    rrt_time = time.time() - start_time

    if rrt_path:
        rrt_length = sum(np.linalg.norm(np.array(rrt_path[i+1]) - np.array(rrt_path[i]))
                        for i in range(len(rrt_path)-1))
        print(f"RRT: Path found in {rrt_time:.2f}s, length: {rrt_length:.2f}m, {len(rrt_path)} waypoints")
    else:
        print("RRT: No path found")

    # Test RRT*
    print("\nTesting RRT* planner...")
    rrtstar_planner = RRTStarPlanner(start, goal, bounds, obstacles, step_size=0.5, max_iter=2000)
    start_time = time.time()
    rrtstar_path = rrtstar_planner.plan()
    rrtstar_time = time.time() - start_time

    if rrtstar_path:
        rrtstar_length = sum(np.linalg.norm(np.array(rrtstar_path[i+1]) - np.array(rrtstar_path[i]))
                            for i in range(len(rrtstar_path)-1))
        print(f"RRT*: Path found in {rrtstar_time:.2f}s, length: {rrtstar_length:.2f}m, {len(rrtstar_path)} waypoints")
    else:
        print("RRT*: No path found")


def main():
    """
    Main function demonstrating RRT planning
    """
    print("RRT Motion Planning Example for Isaac-based Robotics")
    print("="*60)

    # Create a sample environment
    bounds, obstacles = create_sample_environment()

    # Define start and goal
    start = [-8, -8]
    goal = [8, 8]

    print(f"Environment bounds: {bounds}")
    print(f"Number of obstacles: {len(obstacles)}")
    print(f"Start: {start}, Goal: {goal}")

    # Create RRT planner
    planner = RRTPlanner(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=0.5,
        max_iter=2000
    )

    print("\nPlanning path using RRT algorithm...")
    path = planner.plan(visualize=True)  # Set to False to run without visualization

    if path:
        print(f"\nPath found with {len(path)} waypoints!")
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                         for i in range(len(path)-1))
        print(f"Total path length: {path_length:.2f} meters")

        # Show first few and last few waypoints
        print(f"First 3 waypoints: {path[:3]}")
        print(f"Last 3 waypoints: {path[-3:]}")
    else:
        print("\nNo path found! Try adjusting the environment or parameters.")

    # Run benchmark
    print("\n" + "="*60)
    benchmark_planners()

    print("\nRRT planning example completed!")
    print("This demonstrates path planning concepts essential for AI-robot brain navigation systems.")


if __name__ == "__main__":
    main()