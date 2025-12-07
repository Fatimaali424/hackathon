# API Contracts: Isaac Module for Physical AI & Humanoid Robotics

## Overview

This document defines the API contracts for Module 3: The AI-Robot Brain (NVIDIA Isaac). The module covers Isaac Sim integration, perception pipelines, and navigation systems using Isaac ROS packages.

## Isaac Sim Environment Management API

### Create Isaac Sim Scene
- **Endpoint**: `POST /isaac-sim/scenes`
- **Description**: Creates a new Isaac Sim scene with specified configuration
- **Request**:
  - Body: JSON object with scene configuration
    ```json
    {
      "name": "string (required)",
      "description": "string (optional)",
      "usd_path": "string (required)",
      "robot_models": ["string (required)"],
      "lighting": {
        "intensity": "number (optional, default: 1.0)",
        "color": "string (optional, default: '#ffffff')"
      },
      "physics": {
        "gravity": "number (optional, default: -9.81)",
        "friction": "number (optional, default: 0.5)"
      }
    }
    ```
- **Response**:
  - 201 Created: Scene created successfully
    ```json
    {
      "id": "string",
      "name": "string",
      "status": "CREATED | CONFIGURED | RUNNING",
      "created_at": "timestamp"
    }
    ```
  - 400 Bad Request: Invalid scene configuration
  - 409 Conflict: Scene with same name already exists

### Start Simulation
- **Endpoint**: `POST /isaac-sim/scenes/{scene_id}/start`
- **Description**: Starts the simulation for a specified scene
- **Path Parameters**:
  - `scene_id`: string (required) - ID of the scene to start
- **Response**:
  - 200 OK: Simulation started successfully
    ```json
    {
      "scene_id": "string",
      "status": "RUNNING",
      "started_at": "timestamp"
    }
    ```
  - 404 Not Found: Scene not found
  - 409 Conflict: Scene already running

### Stop Simulation
- **Endpoint**: `POST /isaac-sim/scenes/{scene_id}/stop`
- **Description**: Stops the simulation for a specified scene
- **Path Parameters**:
  - `scene_id`: string (required) - ID of the scene to stop
- **Response**:
  - 200 OK: Simulation stopped successfully
    ```json
    {
      "scene_id": "string",
      "status": "STOPPED",
      "stopped_at": "timestamp"
    }
    ```
  - 404 Not Found: Scene not found

## Isaac ROS Perception Pipeline API

### Create Perception Pipeline
- **Endpoint**: `POST /isaac-ros/pipelines/perception`
- **Description**: Creates a new perception pipeline with specified sensors and processing nodes
- **Request**:
  - Body: JSON object with pipeline configuration
    ```json
    {
      "name": "string (required)",
      "input_sensors": [
        {
          "sensor_name": "string (required)",
          "ros_topic": "string (required)",
          "type": "RGB_CAMERA | DEPTH_CAMERA | LIDAR | IMU"
        }
      ],
      "processing_nodes": [
        {
          "name": "string (required)",
          "type": "VSLAM | DEPTH_ESTIMATION | FEATURE_EXTRACTION",
          "parameters": "object (optional)"
        }
      ],
      "output_topics": ["string (required)"]
    }
    ```
- **Response**:
  - 201 Created: Pipeline created successfully
    ```json
    {
      "id": "string",
      "name": "string",
      "status": "DEFINED | VALIDATED | RUNNING",
      "created_at": "timestamp"
    }
    ```
  - 400 Bad Request: Invalid pipeline configuration

### Start Perception Pipeline
- **Endpoint**: `POST /isaac-ros/pipelines/{pipeline_id}/start`
- **Description**: Starts the specified perception pipeline
- **Path Parameters**:
  - `pipeline_id`: string (required) - ID of the pipeline to start
- **Response**:
  - 200 OK: Pipeline started successfully
    ```json
    {
      "pipeline_id": "string",
      "status": "RUNNING",
      "started_at": "timestamp"
    }
    ```
  - 404 Not Found: Pipeline not found

### Get Pipeline Status
- **Endpoint**: `GET /isaac-ros/pipelines/{pipeline_id}`
- **Description**: Gets the current status of a perception pipeline
- **Path Parameters**:
  - `pipeline_id`: string (required) - ID of the pipeline
- **Response**:
  - 200 OK: Pipeline status retrieved successfully
    ```json
    {
      "id": "string",
      "name": "string",
      "status": "DEFINED | VALIDATED | RUNNING | PAUSED | STOPPED",
      "performance": {
        "fps": "number",
        "accuracy": "number"
      },
      "last_updated": "timestamp"
    }
    ```
  - 404 Not Found: Pipeline not found

## Navigation and RL Control API

### Create Navigation Task
- **Endpoint**: `POST /isaac-ros/navigation/tasks`
- **Description**: Creates a new navigation task with specified start/goal positions
- **Request**:
  - Body: JSON object with navigation task configuration
    ```json
    {
      "name": "string (required)",
      "scene_id": "string (required)",
      "robot_model": "string (required)",
      "start_pose": {
        "position": {"x": "number", "y": "number", "z": "number"},
        "orientation": {"x": "number", "y": "number", "z": "number", "w": "number"}
      },
      "goal_region": {
        "center": {"x": "number", "y": "number", "z": "number"},
        "radius": "number"
      },
      "obstacles": [
        {
          "type": "SPHERE | BOX | CYLINDER",
          "position": {"x": "number", "y": "number", "z": "number"},
          "dimensions": {"x": "number", "y": "number", "z": "number"}
        }
      ]
    }
    ```
- **Response**:
  - 201 Created: Navigation task created successfully
    ```json
    {
      "id": "string",
      "name": "string",
      "status": "PLANNED | EXECUTING | SUCCEEDED | FAILED | ABORTED",
      "created_at": "timestamp"
    }
    ```
  - 400 Bad Request: Invalid navigation task configuration

### Execute Navigation Task
- **Endpoint**: `POST /isaac-ros/navigation/tasks/{task_id}/execute`
- **Description**: Executes the specified navigation task
- **Path Parameters**:
  - `task_id`: string (required) - ID of the navigation task
- **Response**:
  - 200 OK: Navigation task execution started successfully
    ```json
    {
      "task_id": "string",
      "status": "EXECUTING",
      "started_at": "timestamp"
    }
    ```
  - 404 Not Found: Task not found

### Get Navigation Task Results
- **Endpoint**: `GET /isaac-ros/navigation/tasks/{task_id}/results`
- **Description**: Gets the results of a completed navigation task
- **Path Parameters**:
  - `task_id`: string (required) - ID of the navigation task
- **Response**:
  - 200 OK: Results retrieved successfully
    ```json
    {
      "task_id": "string",
      "status": "SUCCEEDED | FAILED | ABORTED",
      "metrics": {
        "execution_time": "number (seconds)",
        "path_length": "number (meters)",
        "success_rate": "number (0-1)"
      },
      "completed_at": "timestamp"
    }
    ```
  - 404 Not Found: Task not found
  - 409 Conflict: Task not yet completed

## RL Policy Management API

### Create RL Control Policy
- **Endpoint**: `POST /isaac-ros/rl/policies`
- **Description**: Creates a new reinforcement learning control policy
- **Request**:
  - Body: JSON object with RL policy configuration
    ```json
    {
      "name": "string (required)",
      "robot_model": "string (required)",
      "environment_id": "string (required)",
      "action_space": {
        "type": "CONTINUOUS | DISCRETE",
        "dimensions": "number",
        "bounds": {"min": "number", "max": "number"}
      },
      "observation_space": {
        "type": "CONTINUOUS | DISCRETE",
        "dimensions": "number",
        "bounds": {"min": "number", "max": "number"}
      },
      "reward_function": {
        "type": "string (required)",
        "parameters": "object"
      },
      "training_episodes": "number (default: 1000)"
    }
    ```
- **Response**:
  - 201 Created: RL policy created successfully
    ```json
    {
      "id": "string",
      "name": "string",
      "status": "INITIALIZED | TRAINING | EVALUATING | TRAINED | CONVERGED",
      "created_at": "timestamp"
    }
    ```
  - 400 Bad Request: Invalid RL policy configuration

### Train RL Policy
- **Endpoint**: `POST /isaac-ros/rl/policies/{policy_id}/train`
- **Description**: Starts training for the specified RL policy
- **Path Parameters**:
  - `policy_id`: string (required) - ID of the RL policy
- **Response**:
  - 200 OK: Training started successfully
    ```json
    {
      "policy_id": "string",
      "status": "TRAINING",
      "started_at": "timestamp",
      "current_episode": 0,
      "total_episodes": "number"
    }
    ```
  - 404 Not Found: Policy not found

## Error Response Format

All error responses follow this format:
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object (optional)"
  }
}
```

## Common Error Codes

- `INVALID_INPUT`: Request body contains invalid data
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `RESOURCE_CONFLICT`: Request conflicts with existing resource state
- `PERMISSION_DENIED`: Insufficient permissions for the operation
- `INTERNAL_ERROR`: Unexpected internal error occurred