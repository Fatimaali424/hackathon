---
sidebar_position: 8
---

# Module 2 Assignment: Comprehensive Simulation Project
## Assignment Overview
The Module 2 assignment is a comprehensive simulation project that integrates all concepts learned throughout this module. You will design, implement, and validate a complete Digital Twin system for a robotic platform that includes physics simulation, advanced visualization, and sim-to-real transfer considerations.

## Assignment Objectives
After completing this assignment, you will be able to:
- Design a complete Digital Twin system for a robotic platform
- Implement physics-based simulation with realistic sensor models
- Integrate Unity for advanced visualization and user interaction
- Validate sim-to-real transfer methodologies
- Document simulation design decisions and performance analysis

## Assignment Requirements
### Part 1: Robot Simulation Environment (30 points)
Design and implement a complete simulation environment for a mobile manipulator robot with the following specifications:

1. **Robot Model**:
   - Create a URDF model for a mobile robot platform with a 6-DOF manipulator arm
   - Include realistic physical properties (mass, inertia, friction coefficients)
   - Add appropriate sensors (camera, LIDAR, IMU, joint position sensors)

2. **Environment Design**:
   - Create a Gazebo world with multiple rooms and obstacles
   - Include interactive objects that can be manipulated by the robot
   - Add dynamic elements (moving obstacles, changing lighting conditions)

3. **Physics Configuration**:
   - Configure realistic physics parameters for accurate simulation
   - Implement appropriate collision detection and response
   - Validate contact dynamics and friction models

### Part 2: Unity Integration (30 points)
Implement Unity integration for advanced visualization and Human-Robot Interaction:

1. **Visualization System**:
   - Create synchronized visualization between Gazebo and Unity
   - Implement real-time rendering of robot state and sensor data
   - Add visual effects for robot status and sensor feedback

2. **User Interface**:
   - Design intuitive teleoperation interface in Unity
   - Implement joystick controls for robot navigation
   - Add visualization of sensor data (LIDAR rays, camera feed)

3. **Communication Bridge**:
   - Implement ROS-TCP connection between Unity and ROS 2
   - Ensure reliable data transmission with appropriate error handling
   - Optimize data serialization for performance

### Part 3: sim-to-real Validation (25 points)
Perform sim-to-real transfer validation and analysis:

1. **Domain Randomization**:
   - Implement domain randomization techniques for improved transfer
   - Randomize visual and physical parameters in simulation
   - Validate the impact on algorithm robustness

2. **Performance Analysis**:
   - Compare simulation vs. real-world performance metrics
   - Analyze discrepancies and propose explanations
   - Document lessons learned for future sim-to-real transfer

3. **Validation Methodology**:
   - Design systematic validation procedures
   - Create metrics for measuring simulation fidelity
   - Validate robot behaviors in both simulation and reality

### Part 4: Documentation and Analysis (15 points)
Provide comprehensive documentation and analysis:

1. **Technical Documentation**:
   - Document the complete simulation architecture
   - Explain design decisions and trade-offs
   - Provide setup and configuration instructions

2. **Performance Analysis**:
   - Analyze simulation performance metrics
   - Document computational requirements and optimization strategies
   - Compare different simulation configurations

3. **Reflection Report**:
   - Reflect on challenges encountered during development
   - Discuss lessons learned about Digital Twin systems
   - Propose improvements for future work

## Implementation Guidelines
### Setup Requirements
```bash
# Clone the assignment repository
git clone https://github.com/your-organization/module2-assignment.git
cd module2-assignment

# Create the necessary ROS 2 packages
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_python assignment_simulation --dependencies rclpy std_msgs geometry_msgs sensor_msgs visualization_msgs
ros2 pkg create --build-type ament_python assignment_unity_bridge --dependencies rclpy std_msgs geometry_msgs sensor_msgs
```

### Robot Model Implementation
Create a URDF model in `assignment_simulation/assignment_simulation/robot_model.urdf`:

```xml
<?xml version="1.0"?>
<robot name="mobile_manipulator">
  <!-- Base chassis -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.8 0.6 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.8 0.6 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="20.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Differential drive wheels -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Connect base to wheels -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="-0.4 0.35 0" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="-0.4 -0.35 0" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- 6-DOF manipulator arm (simplified) -->
  <link name="arm_base">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="arm_base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_base"/>
    <origin xyz="0.3 0 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>

  <!-- Additional arm links and joints would continue here -->
</robot>
```

### Simulation Environment Implementation
Create a Gazebo world file in `assignment_simulation/assignment_simulation/assignment_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="assignment_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom environment with multiple rooms -->
    <model name="room_partition">
      <pose>0 0 1.5 0 0 0</pose>
      <link name="wall_1">
        <pose>0 0 0 0 0 0</pose>
        <collision name="wall_1_collision">
          <geometry>
            <box>
              <size>5 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_1_visual">
          <geometry>
            <box>
              <size>5 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Interactive objects -->
    <model name="interactive_cube">
      <pose>2 1 0.5 0 0 0</pose>
      <link name="cube_link">
        <collision name="cube_collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="cube_visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.0017</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0017</iyy>
            <iyz>0</iyz>
            <izz>0.0017</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Add more objects as needed -->
  </world>
</sdf>
```

### Unity Visualization Node
Create the Unity visualization node in `assignment_unity_bridge/assignment_unity_bridge/unity_visualizer.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import JointState, LaserScan, Image
from nav_msgs.msg import Odometry
import socket
import json
import threading
import time
from collections import deque


class AssignmentUnityVisualizerNode(Node):
    def __init__(self):
        super().__init__('assignment_unity_visualizer_node')

        # ROS 2 subscribers for robot state
        self.odom_subscriber = self.create_subscription(
            Odometry, '/robot/odom', self.odom_callback, 10)

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/robot/joint_states', self.joint_state_callback, 10)

        self.laser_subscriber = self.create_subscription(
            LaserScan, '/robot/laser_scan', self.laser_callback, 10)

        self.image_subscriber = self.create_subscription(
            Image, '/robot/camera/image_raw', self.image_callback, 10)

        # Publisher for Unity commands
        self.unity_command_publisher = self.create_publisher(
            String, '/unity_commands', 10)

        # TCP connection to Unity
        self.unity_socket = None
        self.socket_lock = threading.Lock()
        self.connect_to_unity()

        # Timer for sending state updates to Unity
        self.update_timer = self.create_timer(0.033, self.send_state_to_unity)  # ~30 Hz

        # Robot state storage
        self.robot_pose = Pose()
        self.joint_positions = {}
        self.laser_data = None
        self.image_data = None
        self.state_history = deque(maxlen=100)  # For performance analysis

        self.get_logger().info('Assignment Unity Visualizer Node initialized')

    def connect_to_unity(self):
        """Establish connection to Unity application"""
        try:
            with self.socket_lock:
                self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.unity_socket.connect(('localhost', 10000))
                self.get_logger().info('Connected to Unity')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Unity: {e}')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        """Update joint positions from joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def laser_callback(self, msg):
        """Update laser scan data"""
        self.laser_data = {
            'ranges': list(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        }

    def image_callback(self, msg):
        """Handle image data from robot camera"""
        # Store image metadata (actual image data would be handled differently in practice)
        self.image_data = {
            'width': msg.width,
            'height': msg.height,
            'encoding': msg.encoding,
            'is_bigendian': msg.is_bigendian,
            'step': msg.step
        }

    def send_state_to_unity(self):
        """Send robot state to Unity for visualization"""
        if self.unity_socket:
            try:
                with self.socket_lock:
                    state_data = {
                        'timestamp': self.get_clock().now().nanoseconds,
                        'robot_pose': {
                            'x': self.robot_pose.position.x,
                            'y': self.robot_pose.position.y,
                            'z': self.robot_pose.position.z,
                            'qx': self.robot_pose.orientation.x,
                            'qy': self.robot_pose.orientation.y,
                            'qz': self.robot_pose.orientation.z,
                            'qw': self.robot_pose.orientation.w
                        },
                        'joint_positions': self.joint_positions,
                        'laser_data': self.laser_data,
                        'image_data': self.image_data
                    }

                    serialized_data = json.dumps(state_data)
                    self.unity_socket.send(serialized_data.encode('utf-8'))

                    # Store for performance analysis
                    self.state_history.append({
                        'timestamp': time.time(),
                        'data_size': len(serialized_data)
                    })
            except Exception as e:
                self.get_logger().error(f'Failed to send data to Unity: {e}')

    def send_command_to_unity(self, command):
        """Send command to Unity interface"""
        msg = String()
        msg.data = json.dumps(command)
        self.unity_command_publisher.publish(msg)

    def destroy_node(self):
        """Clean up socket connection"""
        if self.unity_socket:
            with self.socket_lock:
                self.unity_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AssignmentUnityVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Performance Analysis Tools
Create a performance analysis script in `assignment_simulation/assignment_simulation/performance_analyzer.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from builtin_interfaces.msg import Time
import time
from collections import deque
import statistics


class PerformanceAnalyzer(Node):
    def __init__(self):
        super().__init__('performance_analyzer')

        # Publishers for performance metrics
        self.simulation_rate_pub = self.create_publisher(Float32, '/performance/simulation_rate', 10)
        self.cpu_usage_pub = self.create_publisher(Float32, '/performance/cpu_usage', 10)
        self.memory_usage_pub = self.create_publisher(Float32, '/performance/memory_usage', 10)

        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.time()

        # Timer for performance analysis
        self.analysis_timer = self.create_timer(1.0, self.analyze_performance)

        self.get_logger().info('Performance Analyzer initialized')

    def record_frame_time(self):
        """Record time for current simulation frame"""
        current_time = time.time()
        if self.last_frame_time:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time

    def analyze_performance(self):
        """Analyze and publish performance metrics"""
        if len(self.frame_times) > 0:
            avg_frame_time = statistics.mean(self.frame_times)
            if avg_frame_time > 0:
                sim_rate = 1.0 / avg_frame_time
                rate_msg = Float32()
                rate_msg.data = sim_rate
                self.simulation_rate_pub.publish(rate_msg)
                self.get_logger().info(f'Simulation rate: {sim_rate:.2f} Hz')

    def get_cpu_usage(self):
        """Get current CPU usage (placeholder implementation)"""
        # This would use psutil in a real implementation
        return 50.0  # Placeholder value

    def get_memory_usage(self):
        """Get current memory usage (placeholder implementation)"""
        # This would use psutil in a real implementation
        return 60.0  # Placeholder value


def main(args=None):
    rclpy.init(args=args)
    analyzer = PerformanceAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Assignment Deliverables
Submit the following deliverables in a single archive:

1. **Source Code** (40 points):
   - Complete ROS 2 packages for simulation and Unity bridge
   - URDF model files and Gazebo world files
   - Unity project files (or export package)
   - All configuration and launch files

2. **Documentation** (30 points):
   - Technical documentation of the simulation architecture
   - Setup and configuration guide
   - Performance analysis report
   - Reflection on sim-to-real transfer challenges

3. **Demonstration Video** (20 points):
   - Video showing the complete system in operation
   - Demonstration of Unity visualization and teleoperation
   - Performance metrics and validation results

4. **Code Quality** (10 points):
   - Clean, well-documented code
   - Proper error handling and logging
   - Adherence to ROS 2 and Unity best practices

## Grading Rubric
| Component | Points | Criteria |
|-----------|--------|----------|
| Robot Simulation | 30 | Realistic physics, proper URDF, sensor integration |
| Unity Integration | 30 | Synchronization, visualization quality, UI design |
| sim-to-real Validation | 25 | Domain randomization, analysis quality, methodology |
| Documentation | 15 | Completeness, clarity, technical accuracy |
| **Total** | **100** | |

## Submission Guidelines
1. **Archive Format**: Create a ZIP archive named `module2_assignment_lastname_firstname.zip`
2. **Directory Structure**:
   ```
   module2_assignment_lastname_firstname/
   ├── src/
   │   ├── assignment_simulation/
   │   └── assignment_unity_bridge/
   ├── unity_project/
   ├── docs/
   │   ├── technical_documentation.pdf
   │   └── performance_analysis.pdf
   ├── videos/
   │   └── demonstration.mp4
   └── README.md
   ```
3. **Deadline**: Submit via the course management system by the specified deadline
4. **Late Policy**: 10% deduction per day late, maximum 3 days

## Resources and References
- [Gazebo Simulation Documentation](http://gazebosim.org/)
- [Unity Robotics Hub Documentation](https://unity.com/products/unity-robotics)
- [ROS 2 Simulation Tutorials](https://docs.ros.org/en/humble/Tutorials/Simulators.html)
- [NVIDIA Isaac Sim Documentation](https://docs.nvidia.com/isaac/isaac_sim/)
- [Sim-to-Real Transfer Research Papers](https://arxiv.org/search?query=sim-to-real+transfer&searchtype=all&abstracts=show&order=-announced_date_first&size=50)

## Tips for Success
1. **Start Early**: This is a comprehensive project requiring significant development time
2. **Iterative Development**: Build and test components incrementally
3. **Performance Monitoring**: Continuously monitor simulation performance
4. **Documentation as You Go**: Don't leave documentation until the end
5. **Validation Early**: Test sim-to-real transfer concepts throughout development

## Support and Questions
For technical questions about this assignment, please:
- Use the course discussion forum for general questions
- Attend office hours for specific implementation issues
- Consult the provided documentation and tutorials
- Collaborate with classmates on concepts (not code)