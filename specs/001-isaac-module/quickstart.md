# Quickstart: Isaac Module for Physical AI & Humanoid Robotics

## Overview

This quickstart guide will help you set up the environment for Module 3: The AI-Robot Brain (NVIDIA Isaac). Follow these steps to prepare your system for working with Isaac Sim, Isaac ROS, and the perception pipelines covered in this module.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better recommended)
- 16GB+ system RAM
- 50GB+ available storage
- 8+ CPU cores recommended

### Software Requirements
- Ubuntu 22.04 LTS
- NVIDIA GPU drivers (535 or newer)
- CUDA 11.8 or newer
- ROS 2 Humble Hawksbill

## Installation Steps

### 1. Install NVIDIA Isaac Sim
```bash
# Download Isaac Sim from NVIDIA Developer website
# Follow the installation guide at: https://docs.omniverse.nvidia.com/isaacsim/latest/installation-guide/index.html

# Verify installation
cd ~/isaac-sim
./isaac-sim.sh
```

### 2. Install Isaac ROS
```bash
# Add NVIDIA package repository
sudo apt update
sudo apt install curl gnupg2 lsb-release
curl -sSL https://repos.mapd.com/apt/GPG-KEY-apt-get-mapd-latest | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] https://repos.mapd.com/apt/$(lsb_release -cs) $(lsb_release -cs)-mapd-latest main" > /etc/apt/sources.list.d/mapd.list'

# Install Isaac ROS packages
sudo apt update
sudo apt install nvidia-isaac-ros
```

### 3. Set up ROS 2 Environment
```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Create workspace for Isaac examples
mkdir -p ~/isaac_ws/src
cd ~/isaac_ws
colcon build
source install/setup.bash
```

### 4. Verify Installation
```bash
# Test Isaac Sim can launch
~/isaac-sim/isaac-sim.sh --version

# Test ROS 2 with Isaac packages
ros2 pkg list | grep isaac

# Check GPU availability
nvidia-smi
```

## Basic Isaac Sim Workflow

### 1. Launch Isaac Sim
```bash
cd ~/isaac-sim
./isaac-sim.sh
```

### 2. Create a Basic Scene
1. Open Isaac Sim Omniverse Create
2. Create a new stage
3. Add a simple robot (e.g., Carter robot from the Isaac Sim assets)
4. Add basic lighting and environment
5. Save as USD file

### 3. Run a Simple Simulation
1. In Isaac Sim, press PLAY to start simulation
2. Verify physics are working correctly
3. Check that the robot responds to basic commands

## Basic Isaac ROS Workflow

### 1. Launch Isaac ROS Bridge
```bash
# Source ROS 2 and Isaac ROS
source /opt/ros/humble/setup.bash
source /usr/local/cuda-11.8/setup.sh  # Adjust CUDA version as needed
source /tmp/isaac_ros_ws/install/setup.bash

# Launch Isaac ROS bringup
ros2 launch isaac_ros_bringup isaac_ros_launch.py
```

### 2. Verify Sensor Data
```bash
# Check available topics
ros2 topic list | grep camera

# View camera data
ros2 run image_view image_view --ros-args --remap /image:=/rgb_camera/image_raw
```

## First Perception Pipeline

### 1. Set up VSLAM Node
```bash
# Launch ORB-SLAM2 through Isaac ROS
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

### 2. Test with Isaac Sim
1. In Isaac Sim, ensure RGB camera is publishing data
2. In RViz, visualize the pose and map topics
3. Move the camera around to see SLAM in action

## Troubleshooting

### Common Issues

**Isaac Sim won't launch:**
- Check NVIDIA drivers are properly installed: `nvidia-smi`
- Verify GPU has 8GB+ VRAM
- Ensure CUDA version matches Isaac Sim requirements

**ROS 2 nodes not communicating:**
- Verify all terminals source the same ROS 2 installation
- Check ROS_DOMAIN_ID is consistent across terminals
- Ensure Isaac ROS packages are properly installed

**Poor simulation performance:**
- Reduce scene complexity during development
- Lower rendering quality settings in Isaac Sim
- Close other GPU-intensive applications

### Verification Commands

```bash
# Check Isaac Sim installation
ls -la ~/isaac-sim/

# Check Isaac ROS packages
dpkg -l | grep isaac

# Check Isaac Sim can communicate with ROS
ros2 topic list | grep isaac
```

## Next Steps

1. Complete Chapter 9 hands-on task: Create Isaac Sim scene with humanoid robot
2. Complete Chapter 10 hands-on task: Build perception pipeline with RGB-D camera
3. Explore the sample projects in Isaac Sim
4. Practice ROS 2 command line tools with Isaac ROS packages

## Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/user-guide.html)
- [Isaac ROS Documentation](https://docs.nvidia.com/isaac/packages/isaac_ros_bringup/index.html)
- [ROS 2 Humble Tutorials](https://docs.ros.org/en/humble/Tutorials.html)