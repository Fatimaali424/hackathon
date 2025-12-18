# API Contracts: Physical AI & Humanoid Robotics Book

## Overview
This document defines the API contracts relevant to the Physical AI & Humanoid Robotics book project. These contracts represent the interfaces that students and developers will interact with when implementing robotic systems as described in the book. The contracts follow REST principles where applicable and ROS 2 service/action patterns where appropriate.

## 1. Robot Control API

### 1.1 Navigation Service
**Service Name**: `/move_to_pose`
**Service Type**: `geometry_msgs/srv/Pose`

#### Request
```json
{
  "target_pose": {
    "position": {
      "x": 1.5,
      "y": 2.0,
      "z": 0.0
    },
    "orientation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.707,
      "w": 0.707
    }
  },
  "frame_id": "map",
  "timeout": 30.0
}
```

#### Response
```json
{
  "success": true,
  "message": "Navigation completed successfully",
  "execution_time": 12.5,
  "final_pose": {
    "position": {
      "x": 1.49,
      "y": 1.98,
      "z": 0.0
    },
    "orientation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.705,
      "w": 0.709
    }
  }
}
```

#### Error Response
```json
{
  "success": false,
  "error_code": "NAVIGATION_FAILED",
  "message": "Failed to reach target pose within timeout",
  "recovery_options": ["retry", "use_alternative_path", "manual_control"]
}
```

### 1.2 Manipulation Action
**Action Name**: `/execute_manipulation`
**Action Type**: `control_msgs/action/FollowJointTrajectory`

#### Goal
```json
{
  "trajectory": {
    "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    "points": [
      {
        "positions": [0.0, -1.0, 0.0, -2.0, 0.0, 0.0],
        "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "time_from_start": 1.0
      },
      {
        "positions": [0.5, -1.2, 0.3, -1.8, 0.2, 0.1],
        "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "time_from_start": 2.0
      }
    ]
  }
}
```

#### Feedback
```json
{
  "current_state": "executing",
  "current_point": 1,
  "remaining_points": 1,
  "estimated_completion_time": 1.5,
  "current_error": 0.02
}
```

#### Result
```json
{
  "success": true,
  "error_code": 0,
  "message": "Manipulation completed successfully",
  "execution_time": 2.3,
  "final_positions": [0.5, -1.2, 0.3, -1.8, 0.2, 0.1]
}
```

## 2. Perception API

### 2.1 Object Detection Service
**Service Name**: `/detect_objects`
**Service Type**: Custom service `ObjectDetection`

#### Request
```json
{
  "sensor_source": "camera_front",
  "confidence_threshold": 0.7,
  "object_classes": ["person", "cup", "bottle", "chair"],
  "roi": {
    "x": 0.0,
    "y": 0.0,
    "width": 1.0,
    "height": 1.0
  }
}
```

#### Response
```json
{
  "objects": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 200,
        "height": 300
      },
      "center_3d": {
        "x": 1.5,
        "y": 0.8,
        "z": 0.0
      },
      "distance": 2.1
    }
  ],
  "detection_time": 0.15,
  "sensor_status": "active"
}
```

### 2.2 SLAM Service
**Service Name**: `/get_map`
**Service Type**: `nav_msgs/srv/GetMap`

#### Request
```json
{
  "get_current_map": true,
  "resolution": 0.05,
  "map_frame": "map"
}
```

#### Response
```json
{
  "map": {
    "header": {
      "frame_id": "map",
      "stamp": "2025-12-17T10:30:00Z"
    },
    "info": {
      "resolution": 0.05,
      "width": 2000,
      "height": 2000,
      "origin": {
        "position": {"x": -50.0, "y": -50.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
      }
    },
    "data": [0, 0, 0, 100, 100, 0, ...]  // Array of occupancy values
  },
  "map_time": 1.2
}
```

## 3. Human-Robot Interaction API

### 3.1 Speech Recognition Action
**Action Name**: `/recognize_speech`
**Action Type**: Custom action `SpeechRecognition`

#### Goal
```json
{
  "recording_duration": 5.0,
  "language_code": "en-US",
  "grammar_file": "robot_commands.gram",
  "confidence_threshold": 0.8
}
```

#### Feedback
```json
{
  "status": "recording",
  "remaining_time": 2.3,
  "audio_level": 0.65,
  "partial_transcript": "Please go to the"
}
```

#### Result
```json
{
  "success": true,
  "transcript": "Please go to the kitchen and bring me a cup",
  "confidence": 0.89,
  "command_parsed": {
    "action": "navigate_and_manipulate",
    "target_location": "kitchen",
    "object": "cup",
    "task_id": "task_001"
  },
  "processing_time": 0.8
}
```

### 3.2 Command Processing Service
**Service Name**: `/process_command`
**Service Type**: Custom service `CommandProcessing`

#### Request
```json
{
  "command_text": "Please go to the kitchen and bring me a cup",
  "command_type": "natural_language",
  "user_context": {
    "user_id": "user_001",
    "location": "living_room",
    "preferences": {
      "speed_preference": "moderate",
      "safety_level": "high"
    }
  }
}
```

#### Response
```json
{
  "parsed_commands": [
    {
      "id": "nav_001",
      "type": "navigation",
      "target": "kitchen",
      "priority": 1
    },
    {
      "id": "detect_001",
      "type": "object_detection",
      "object_type": "cup",
      "priority": 2
    },
    {
      "id": "manip_001",
      "type": "manipulation",
      "action": "grasp",
      "object_id": "cup_001",
      "priority": 3
    }
  ],
  "execution_plan": {
    "sequence": ["nav_001", "detect_001", "manip_001"],
    "estimated_time": 120.0,
    "resource_requirements": ["navigation", "manipulation", "perception"]
  },
  "confidence": 0.92
}
```

## 4. System Monitoring API

### 4.1 Robot Status Service
**Service Name**: `/get_robot_status`
**Service Type**: Custom service `RobotStatus`

#### Request
```json
{
  "include_detailed_status": true,
  "components": ["battery", "motors", "sensors", "computation"]
}
```

#### Response
```json
{
  "timestamp": "2025-12-17T10:30:00Z",
  "overall_status": "operational",
  "battery": {
    "level": 0.85,
    "time_remaining": 120.0,
    "status": "normal"
  },
  "motors": {
    "status": "normal",
    "temperature_avg": 35.2,
    "current_avg": 1.2
  },
  "sensors": {
    "camera_front": "active",
    "lidar_3d": "active",
    "imu": "active",
    "microphone_array": "active"
  },
  "computation": {
    "cpu_usage": 0.65,
    "gpu_usage": 0.45,
    "memory_usage": 0.72,
    "thermal_status": "normal"
  }
}
```

### 4.2 Simulation Control Service
**Service Name**: `/simulation_control`
**Service Type**: Custom service `SimulationControl`

#### Request
```json
{
  "command": "reset",
  "target_scene": "home_environment_01",
  "reset_position": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "yaw": 0.0
  }
}
```

#### Response
```json
{
  "success": true,
  "message": "Simulation reset completed",
  "new_timestamp": "2025-12-17T10:31:00Z",
  "scene_loaded": "home_environment_01",
  "robot_pose": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "yaw": 0.0
  },
  "reset_time": 0.8
}
```

## 5. Data Management API

### 5.1 Logging Service
**Service Name**: `/log_data`
**Service Type**: Custom service `DataLogging`

#### Request
```json
{
  "log_type": "task_execution",
  "data": {
    "task_id": "task_001",
    "start_time": "2025-12-17T10:30:00Z",
    "end_time": "2025-12-17T10:32:30Z",
    "status": "completed",
    "execution_path": [
      {"x": 0.0, "y": 0.0, "timestamp": "2025-12-17T10:30:00Z"},
      {"x": 1.5, "y": 0.8, "timestamp": "2025-12-17T10:31:15Z"}
    ],
    "performance_metrics": {
      "accuracy": 0.95,
      "efficiency": 0.87,
      "safety_score": 0.98
    }
  },
  "persistence_level": "persistent"
}
```

#### Response
```json
{
  "success": true,
  "log_id": "log_20251217_103000_task_001",
  "stored_location": "/logs/2025/12/17/",
  "retention_policy": "30_days",
  "index_created": true,
  "storage_size": 2048
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Descriptive error message",
    "details": {
      "component": "component_name",
      "timestamp": "2025-12-17T10:30:00Z",
      "severity": "warning|error|critical"
    },
    "suggested_action": "Recommended action to resolve the issue"
  }
}
```

### Common Error Codes
- `TIMEOUT_ERROR`: Operation exceeded time limit
- `PERMISSION_DENIED`: Insufficient permissions for operation
- `RESOURCE_UNAVAILABLE`: Required resource not available
- `VALIDATION_FAILED`: Input validation failed
- `COMMUNICATION_ERROR`: Network or communication failure
- `CALIBRATION_REQUIRED`: Sensor calibration needed
- `SAFETY_VIOLATION`: Requested action violates safety constraints

## Security Considerations

### Authentication
All services requiring privileged access should implement:
- Token-based authentication
- Role-based access control
- Session management

### Authorization
- Read-only access for basic monitoring
- Control access limited to authorized operators
- Administrative functions restricted to system administrators

### Data Protection
- Encryption of sensitive data in transit
- Secure logging of personal information
- Compliance with data protection regulations