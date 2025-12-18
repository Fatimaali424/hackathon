---
sidebar_position: 3
---

# Planning System Integration

## Overview

The planning system integration is a critical component of the autonomous humanoid robot, responsible for transforming high-level user commands into executable action sequences. This system bridges the gap between command interpretation and physical execution, handling complex multi-step tasks that require navigation, manipulation, and coordination of various subsystems.

The planning system must handle both high-level task planning (what to do) and motion planning (how to do it), ensuring that the robot can execute complex commands while respecting environmental constraints, safety requirements, and resource limitations.

## Planning Architecture

### Hierarchical Planning Structure

The planning system follows a hierarchical architecture with multiple levels of abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL TASK PLANNER                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Task Decomposer │  │ Constraint      │  │ Plan Validator  │ │
│  │                 │  │ Manager         │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MOTION PLANNING                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Path Planner    │  │ Trajectory      │  │ Collision       │ │
│  │                 │  │ Generator       │  │ Checker         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION PLANNER                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Navigation      │  │ Manipulation    │  │ Control         │ │
│  │ Planner         │  │ Planner         │  │ Sequencer       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOW-LEVEL CONTROL                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Motor Control   │  │ Sensor Fusion   │  │ Safety Monitor  │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## High-Level Task Planning

### Task Decomposition

The high-level task planner decomposes complex user commands into manageable subtasks:

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import heapq

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    WAIT = "wait"
    CONDITIONAL = "conditional"

@dataclass
class Task:
    """Represents a single task in the plan"""
    id: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of tasks that must complete first
    priority: int = 0
    estimated_duration: float = 0.0  # seconds
    resources_required: List[str] = None  # e.g., "arm", "navigation", "camera"

    def __post_init__(self):
        if self.resources_required is None:
            self.resources_required = []

@dataclass
class Plan:
    """Represents a complete execution plan"""
    tasks: List[Task]
    constraints: Dict[str, Any]
    plan_graph: nx.DiGraph = None

    def __post_init__(self):
        self._build_plan_graph()

    def _build_plan_graph(self):
        """Build dependency graph for task execution"""
        self.plan_graph = nx.DiGraph()

        for task in self.tasks:
            self.plan_graph.add_node(task.id, task=task)

        for task in self.tasks:
            for dep_id in task.dependencies:
                self.plan_graph.add_edge(dep_id, task.id)

class HighLevelTaskPlanner:
    def __init__(self):
        self.task_database = self._initialize_task_database()
        self.constraint_checker = ConstraintChecker()

    def decompose_command(self, command_interpretation: Dict) -> Plan:
        """Decompose high-level command into executable tasks"""
        intent = command_interpretation.get('intent')
        entities = command_interpretation.get('entities', {})

        # Map command to task decomposition strategy
        if intent == 'navigation':
            return self._decompose_navigation(entities)
        elif intent == 'object_interaction':
            return self._decompose_object_interaction(entities)
        elif intent == 'manipulation':
            return self._decompose_manipulation(entities)
        elif intent == 'action':
            return self._decompose_action(entities)
        else:
            # For unknown intents, try to decompose based on entities
            return self._decompose_generic(entities)

    def _decompose_navigation(self, entities: Dict) -> Plan:
        """Decompose navigation commands into tasks"""
        tasks = []

        # 1. Perceive current location
        tasks.append(Task(
            id="perceive_current_location",
            task_type=TaskType.PERCEPTION,
            description="Perceive current location",
            parameters={"sensor": "camera"},
            dependencies=[],
            estimated_duration=2.0,
            resources_required=["camera"]
        ))

        # 2. Plan path to destination
        destination = entities.get('location', 'unknown')
        tasks.append(Task(
            id=f"plan_path_to_{destination}",
            task_type=TaskType.NAVIGATION,
            description=f"Plan path to {destination}",
            parameters={"destination": destination},
            dependencies=["perceive_current_location"],
            estimated_duration=5.0,
            resources_required=["navigation"]
        ))

        # 3. Navigate to destination
        tasks.append(Task(
            id=f"navigate_to_{destination}",
            task_type=TaskType.NAVIGATION,
            description=f"Navigate to {destination}",
            parameters={"destination": destination},
            dependencies=[f"plan_path_to_{destination}"],
            estimated_duration=30.0,  # Variable based on distance
            resources_required=["navigation", "sensors"]
        ))

        return Plan(tasks=tasks, constraints={
            "max_navigation_time": 120.0,
            "safety_distance": 0.5
        })

    def _decompose_object_interaction(self, entities: Dict) -> Plan:
        """Decompose object interaction commands into tasks"""
        tasks = []

        obj = entities.get('object', 'unknown')
        location = entities.get('location', 'current')

        # 1. Navigate to object location
        if location != 'current':
            tasks.append(Task(
                id=f"navigate_to_{location}",
                task_type=TaskType.NAVIGATION,
                description=f"Navigate to {location}",
                parameters={"destination": location},
                dependencies=[],
                estimated_duration=20.0,
                resources_required=["navigation", "sensors"]
            ))

        # 2. Perceive object
        nav_dep = [f"navigate_to_{location}"] if location != 'current' else []
        tasks.append(Task(
            id=f"perceive_{obj}",
            task_type=TaskType.PERCEPTION,
            description=f"Perceive {obj}",
            parameters={"object": obj, "sensor": "camera"},
            dependencies=nav_dep,
            estimated_duration=3.0,
            resources_required=["camera", "perception"]
        ))

        # 3. Plan manipulation
        perceive_dep = [f"perceive_{obj}"] + nav_dep
        tasks.append(Task(
            id=f"plan_grasp_{obj}",
            task_type=TaskType.MANIPULATION,
            description=f"Plan grasp for {obj}",
            parameters={"object": obj, "action": "grasp"},
            dependencies=perceive_dep,
            estimated_duration=5.0,
            resources_required=["manipulation_planner"]
        ))

        # 4. Execute manipulation
        plan_dep = [f"plan_grasp_{obj}"] + perceive_dep
        tasks.append(Task(
            id=f"grasp_{obj}",
            task_type=TaskType.MANIPULATION,
            description=f"Grasp {obj}",
            parameters={"object": obj, "action": "grasp"},
            dependencies=plan_dep,
            estimated_duration=10.0,
            resources_required=["manipulation", "arm"]
        ))

        return Plan(tasks=tasks, constraints={
            "max_manipulation_time": 60.0,
            "object_recognition_threshold": 0.8
        })

    def _decompose_manipulation(self, entities: Dict) -> Plan:
        """Decompose manipulation commands into tasks"""
        tasks = []

        obj = entities.get('object', 'unknown')
        manipulation_type = entities.get('manipulation_type', 'manipulate')

        # 1. Perceive object
        tasks.append(Task(
            id=f"perceive_{obj}",
            task_type=TaskType.PERCEPTION,
            description=f"Perceive {obj}",
            parameters={"object": obj, "sensor": "camera"},
            dependencies=[],
            estimated_duration=3.0,
            resources_required=["camera", "perception"]
        ))

        # 2. Plan manipulation
        tasks.append(Task(
            id=f"plan_{manipulation_type}_{obj}",
            task_type=TaskType.MANIPULATION,
            description=f"Plan {manipulation_type} for {obj}",
            parameters={"object": obj, "action": manipulation_type},
            dependencies=[f"perceive_{obj}"],
            estimated_duration=5.0,
            resources_required=["manipulation_planner"]
        ))

        # 3. Execute manipulation
        tasks.append(Task(
            id=f"{manipulation_type}_{obj}",
            task_type=TaskType.MANIPULATION,
            description=f"{manipulation_type.title()} {obj}",
            parameters={"object": obj, "action": manipulation_type},
            dependencies=[f"plan_{manipulation_type}_{obj}"],
            estimated_duration=10.0,
            resources_required=["manipulation", "arm"]
        ))

        return Plan(tasks=tasks, constraints={
            "max_manipulation_time": 60.0,
            "safety_force_limit": 10.0
        })

    def _decompose_generic(self, entities: Dict) -> Plan:
        """Generic task decomposition for unknown intents"""
        tasks = []

        # Default to basic perception and navigation
        tasks.append(Task(
            id="perceive_environment",
            task_type=TaskType.PERCEPTION,
            description="Perceive environment",
            parameters={"sensor": "camera"},
            dependencies=[],
            estimated_duration=2.0,
            resources_required=["camera"]
        ))

        return Plan(tasks=tasks, constraints={
            "max_execution_time": 300.0
        })

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate plan against constraints"""
        violations = []

        # Check for cycles in dependency graph
        if not nx.is_directed_acyclic_graph(plan.plan_graph):
            violations.append("Plan contains cyclic dependencies")

        # Check resource conflicts
        resource_conflicts = self._check_resource_conflicts(plan)
        violations.extend(resource_conflicts)

        # Check temporal constraints
        temporal_violations = self._check_temporal_constraints(plan)
        violations.extend(temporal_violations)

        return len(violations) == 0, violations

    def _check_resource_conflicts(self, plan: Plan) -> List[str]:
        """Check for resource conflicts in plan"""
        violations = []

        # Simple resource conflict checking
        for task in plan.tasks:
            for other_task in plan.tasks:
                if task.id != other_task.id:
                    # Check if tasks require same resource and could conflict
                    common_resources = set(task.resources_required) & set(other_task.resources_required)
                    if common_resources:
                        # Check if they have overlapping execution windows
                        # This is simplified - in practice would check actual timing
                        violations.append(f"Potential resource conflict between {task.id} and {other_task.id} for resources {common_resources}")

        return violations

    def _check_temporal_constraints(self, plan: Plan) -> List[str]:
        """Check temporal constraints"""
        violations = []

        # Check if plan duration exceeds limits
        total_estimated_time = sum(task.estimated_duration for task in plan.tasks)
        max_time = plan.constraints.get("max_execution_time", float('inf'))

        if total_estimated_time > max_time:
            violations.append(f"Plan estimated duration ({total_estimated_time}s) exceeds maximum allowed time ({max_time}s)")

        return violations

    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize plan for efficiency"""
        # This would implement plan optimization algorithms
        # For now, return the original plan
        return plan
```

## Motion Planning Integration

### Path Planning System

```python
import numpy as np
from scipy.spatial import KDTree
import heapq
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class MotionPlanner:
    def __init__(self, resolution=0.1, inflation_radius=0.5):
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.grid_map = None
        self.occupancy_grid = None

    def set_map(self, occupancy_grid: np.ndarray, origin: Tuple[float, float]):
        """Set the occupancy grid map for planning"""
        self.occupancy_grid = occupancy_grid
        self.origin = origin
        self.grid_resolution = 1.0  # Grid cell size in meters

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Plan path using A* algorithm"""
        if self.occupancy_grid is None:
            raise ValueError("Map not set. Call set_map() first.")

        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # Check if start and goal are valid
        if not self._is_valid_cell(start_grid) or not self._is_valid_cell(goal_grid):
            return None

        # Run A* path planning
        path_grid = self._a_star(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert grid path back to world coordinates
        path_world = [self._grid_to_world(cell) for cell in path_grid]

        return path_world

    def _world_to_grid(self, world_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y = world_coord
        grid_x = int((x - self.origin[0]) / self.grid_resolution)
        grid_y = int((y - self.origin[1]) / self.grid_resolution)
        return (grid_x, grid_y)

    def _grid_to_world(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        grid_x, grid_y = grid_coord
        world_x = self.origin[0] + grid_x * self.grid_resolution
        world_y = self.origin[1] + grid_y * self.grid_resolution
        return (world_x, world_y)

    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if grid cell is valid (within bounds and not occupied)"""
        x, y = cell

        # Check bounds
        if x < 0 or x >= self.occupancy_grid.shape[1] or y < 0 or y >= self.occupancy_grid.shape[0]:
            return False

        # Check occupancy (assuming 100 = occupied, 0 = free)
        return self.occupancy_grid[y, x] < 50  # Threshold for "free"

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

        # Visited set
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

class TrajectoryGenerator:
    """Generate smooth trajectories from path points"""

    def __init__(self, max_velocity=1.0, max_acceleration=0.5):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_trajectory(self, path: List[Tuple[float, float]],
                           start_velocity: float = 0.0,
                           end_velocity: float = 0.0) -> List[Dict[str, float]]:
        """Generate trajectory with velocity profiles"""
        if len(path) < 2:
            return []

        trajectory = []

        # Calculate path length and segment lengths
        total_length = 0.0
        segment_lengths = []

        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            segment_length = np.sqrt(dx*dx + dy*dy)
            segment_lengths.append(segment_length)
            total_length += segment_length

        # Generate trajectory points
        cumulative_distance = 0.0
        current_velocity = start_velocity

        for i, (segment_length) in enumerate(segment_lengths):
            # Calculate time for this segment (simplified)
            # In practice, would use more sophisticated velocity profiling
            avg_velocity = (self.max_velocity + current_velocity) / 2.0
            segment_time = segment_length / avg_velocity if avg_velocity > 0 else 0.1

            # Add point to trajectory
            trajectory.append({
                'position': path[i+1],
                'velocity': current_velocity,
                'acceleration': 0.0,  # Simplified
                'time': cumulative_distance / self.max_velocity if self.max_velocity > 0 else 0,
                'segment_length': segment_length
            })

            cumulative_distance += segment_length

            # Update velocity for next segment (simplified)
            current_velocity = min(self.max_velocity, current_velocity + self.max_acceleration * segment_time)

        return trajectory
```

## Task Execution Planning

### Execution Planner

```python
from datetime import datetime, timedelta
import threading
import time

class ExecutionPlanner:
    """Plans and manages execution of tasks"""

    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.resource_manager = ResourceManager()
        self.execution_lock = threading.Lock()

    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a complete plan"""
        start_time = time.time()

        # Validate plan before execution
        is_valid, violations = HighLevelTaskPlanner().validate_plan(plan)
        if not is_valid:
            return {
                'success': False,
                'error': f'Plan validation failed: {violations}',
                'execution_time': 0.0,
                'completed_tasks': [],
                'failed_tasks': []
            }

        # Sort tasks by dependencies and priority
        execution_order = self._determine_execution_order(plan)

        # Execute tasks in order
        for task_id in execution_order:
            task = self._get_task_by_id(plan, task_id)

            # Acquire required resources
            resources_acquired = self.resource_manager.acquire_resources(task.resources_required)
            if not resources_acquired:
                self.failed_tasks.append(task_id)
                continue

            try:
                # Execute task
                task_result = self._execute_single_task(task)

                if task_result['success']:
                    self.completed_tasks.append(task_id)
                else:
                    self.failed_tasks.append(task_id)

            except Exception as e:
                self.failed_tasks.append(task_id)
                print(f"Task {task_id} failed with error: {e}")
            finally:
                # Release resources
                self.resource_manager.release_resources(task.resources_required)

        total_time = time.time() - start_time

        return {
            'success': len(self.failed_tasks) == 0,
            'execution_time': total_time,
            'completed_tasks': self.completed_tasks.copy(),
            'failed_tasks': self.failed_tasks.copy(),
            'total_tasks': len(plan.tasks)
        }

    def _determine_execution_order(self, plan: Plan) -> List[str]:
        """Determine execution order based on dependencies and priorities"""
        # Use topological sort for dependency ordering
        topo_order = list(nx.topological_sort(plan.plan_graph))

        # Group tasks by priority
        priority_groups = {}
        for task_id in topo_order:
            task = plan.plan_graph.nodes[task_id]['task']
            priority = task.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(task_id)

        # Sort priorities (higher numbers = higher priority)
        sorted_priorities = sorted(priority_groups.keys(), reverse=True)

        # Build final order
        execution_order = []
        for priority in sorted_priorities:
            execution_order.extend(priority_groups[priority])

        return execution_order

    def _get_task_by_id(self, plan: Plan, task_id: str):
        """Get task by ID from plan"""
        for task in plan.tasks:
            if task.id == task_id:
                return task
        return None

    def _execute_single_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task"""
        try:
            # Simulate task execution
            execution_time = task.estimated_duration
            time.sleep(min(0.1, execution_time))  # Simulate execution time

            # Simulate success/failure based on task type
            success_probability = 0.9  # 90% success rate
            success = np.random.random() < success_probability

            return {
                'success': success,
                'task_id': task.id,
                'execution_time': execution_time,
                'result': 'Task completed successfully' if success else 'Task failed'
            }
        except Exception as e:
            return {
                'success': False,
                'task_id': task.id,
                'execution_time': 0.0,
                'result': f'Task failed with error: {str(e)}'
            }

class ResourceManager:
    """Manage shared resources for task execution"""

    def __init__(self):
        self.available_resources = {
            'navigation': True,
            'manipulation': True,
            'camera': True,
            'arm': True,
            'sensors': True,
            'perception': True,
            'manipulation_planner': True
        }
        self.resource_locks = {resource: threading.Lock() for resource in self.available_resources}
        self.resource_usage = {resource: None for resource in self.available_resources}  # Task ID using resource

    def acquire_resources(self, resource_list: List[str]) -> bool:
        """Acquire required resources"""
        acquired_resources = []

        try:
            for resource in resource_list:
                if resource in self.resource_locks:
                    if self.resource_locks[resource].acquire(blocking=False):
                        if self.available_resources[resource]:
                            self.available_resources[resource] = False
                            acquired_resources.append(resource)
                        else:
                            # Resource not available, release previously acquired resources
                            for acquired in acquired_resources:
                                self.available_resources[acquired] = True
                                self.resource_locks[acquired].release()
                            return False
                    else:
                        # Could not acquire lock, release previously acquired resources
                        for acquired in acquired_resources:
                            self.available_resources[acquired] = True
                            self.resource_locks[acquired].release()
                        return False
            return True
        except Exception:
            # On error, release all acquired resources
            for acquired in acquired_resources:
                self.available_resources[acquired] = True
                self.resource_locks[acquired].release()
            return False

    def release_resources(self, resource_list: List[str]):
        """Release resources"""
        for resource in resource_list:
            if resource in self.resource_locks:
                with self.resource_locks[resource]:
                    self.available_resources[resource] = True