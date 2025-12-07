# Data Model: Isaac Module for Physical AI & Humanoid Robotics

## Overview

This document defines the key data entities and relationships for Module 3: The AI-Robot Brain (NVIDIA Isaac). The module focuses on photorealistic simulation with Isaac Sim, perception pipelines, and navigation with reinforcement learning.

## Core Entities

### Isaac Sim Environment
- **Name**: String (required) - Unique identifier for the simulation environment
- **Description**: Text - Human-readable description of the environment
- **USD Scene Path**: String (required) - Path to the USD file defining the scene
- **Robot Models**: Array of Robot Model references
- **Lighting Conditions**: Object - Parameters for photorealistic lighting
- **Physics Properties**: Object - Gravity, friction, collision parameters
- **Sensor Configurations**: Array of Sensor Configuration objects
- **Synthetic Data Settings**: Object - Parameters for generating synthetic sensor data

### Robot Model
- **Name**: String (required) - Human-readable name of the robot
- **USD Asset Path**: String (required) - Path to the USD robot definition
- **URDF Path**: String (optional) - Path to URDF for ROS compatibility
- **Joint Definitions**: Array of Joint Definition objects
- **Link Definitions**: Array of Link Definition objects
- **Sensor Mounts**: Array of Sensor Mount objects
- **Actuator Specifications**: Array of Actuator Specification objects

### Joint Definition
- **Name**: String (required) - Unique name of the joint
- **Type**: Enum (revolute, prismatic, fixed, etc.) - Type of joint
- **Parent Link**: String (required) - Name of parent link
- **Child Link**: String (required) - Name of child link
- **Axis**: Vector3 - Joint axis of rotation/translation
- **Limits**: Object - Min/max values for joint movement
- **Dynamics**: Object - Friction, damping, stiffness parameters

### Sensor Configuration
- **Name**: String (required) - Unique name of the sensor
- **Type**: Enum (RGB camera, depth camera, LiDAR, IMU, etc.) - Sensor type
- **Mount Point**: String (required) - Where the sensor is mounted on the robot
- **Parameters**: Object - Sensor-specific parameters (resolution, range, etc.)
- **ROS Topic**: String - ROS topic name for sensor data
- **Isaac Sim Component**: String - Isaac Sim component type

### Perception Pipeline
- **Name**: String (required) - Unique name of the perception pipeline
- **Input Sensors**: Array of Sensor Configuration references
- **Processing Nodes**: Array of Processing Node objects
- **Output Types**: Array of Output Type enums (pose, depth, features, etc.)
- **Performance Metrics**: Object - FPS, accuracy, latency measurements
- **Calibration Data**: Object - Camera intrinsic/extrinsic parameters

### Processing Node
- **Name**: String (required) - Name of the processing node
- **Type**: Enum (VSLAM, depth estimation, feature extraction, etc.)
- **Input Topics**: Array of ROS topic strings
- **Output Topics**: Array of ROS topic strings
- **Parameters**: Object - Algorithm-specific parameters
- **Isaac ROS Package**: String - Name of Isaac ROS package used

### Navigation Task
- **Name**: String (required) - Name of the navigation task
- **Environment**: Isaac Sim Environment reference
- **Start Pose**: Pose object - Starting position and orientation
- **Goal Region**: Region object - Target area for navigation
- **Obstacles**: Array of Obstacle objects - Static/dynamic obstacles
- **Success Criteria**: Object - Conditions for task completion
- **Metrics**: Object - Performance measurements (time, path length, etc.)

### RL Control Policy
- **Name**: String (required) - Name of the reinforcement learning policy
- **Robot Model**: Robot Model reference
- **Environment**: Isaac Sim Environment reference
- **Action Space**: Object - Definition of possible actions
- **Observation Space**: Object - Definition of sensor observations
- **Reward Function**: Object - Reward structure for training
- **Training Episodes**: Integer - Number of episodes for training
- **Policy Network**: Object - Neural network architecture

## Relationships

### Environment Contains
- Isaac Sim Environment "contains" multiple Robot Models
- Isaac Sim Environment "contains" multiple Sensor Configurations

### Robot Composed Of
- Robot Model "contains" multiple Joint Definitions
- Robot Model "contains" multiple Link Definitions
- Robot Model "has" multiple Sensor Mounts

### Perception Pipeline Uses
- Perception Pipeline "uses" multiple Sensor Configurations
- Perception Pipeline "contains" multiple Processing Nodes

### Navigation Depends On
- Navigation Task "uses" Isaac Sim Environment
- Navigation Task "uses" Robot Model

### Control Policy Governs
- RL Control Policy "controls" Robot Model
- RL Control Policy "operates in" Isaac Sim Environment

## Data Validation Rules

### Isaac Sim Environment
- USD Scene Path must be a valid file path
- At least one Robot Model must be associated
- Physics Properties must have valid numerical values

### Robot Model
- USD Asset Path must reference a valid USD file
- Joint Definitions must form a valid kinematic tree
- No duplicate joint names within a robot

### Sensor Configuration
- ROS Topic names must follow ROS naming conventions
- Parameters must be within valid ranges for the sensor type
- Mount Point must correspond to a valid link on the robot

### Perception Pipeline
- Input and output topics must be properly connected
- Processing Nodes must form a valid processing graph (no cycles)
- All referenced sensors must exist

### Navigation Task
- Start Pose and Goal Region must be within environment bounds
- Success Criteria must be achievable
- Obstacles must not completely block path to goal

### RL Control Policy
- Action and Observation spaces must be properly defined
- Reward function must encourage desired behavior
- Training episodes must be sufficient for convergence

## State Transitions

### Isaac Sim Environment States
- `CREATED` → `CONFIGURED` → `RUNNING` → `PAUSED` → `STOPPED`

### Perception Pipeline States
- `DEFINED` → `VALIDATED` → `RUNNING` → `PAUSED` → `STOPPED`

### Navigation Task States
- `PLANNED` → `EXECUTING` → `SUCCEEDED` | `FAILED` | `ABORTED`

### RL Training States
- `INITIALIZED` → `TRAINING` → `EVALUATING` → `TRAINED` | `CONVERGED`

## Constraints

### Performance Constraints
- Perception pipelines must process data in real-time (30+ FPS)
- Navigation planning must complete within 1 second
- RL control policies must execute at 50+ Hz

### Resource Constraints
- Isaac Sim scenes must run on 8GB+ GPU systems
- Sensor data must be processed within available bandwidth
- Training data must fit within available storage

### Compatibility Constraints
- All components must work with ROS 2 Humble
- USD assets must be compatible with Isaac Sim 2023.1.0
- Code examples must run on Ubuntu 22.04

## Glossary

- **USD (Universal Scene Description)**: NVIDIA's format for 3D scenes and assets
- **VSLAM (Visual Simultaneous Localization and Mapping)**: Technique for localizing a robot using visual input
- **TF (Transform)**: ROS system for tracking coordinate frame relationships
- **Isaac ROS**: NVIDIA's collection of ROS 2 packages for robotics perception and control
- **Synthetic Data**: Artificially generated training data from simulation
- **Sim-to-Real Transfer**: Technique of training in simulation and applying to real robots