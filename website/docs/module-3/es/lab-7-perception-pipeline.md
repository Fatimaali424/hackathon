---
sidebar_position: 5
---

# Lab 7: Basic Perception Pipeline with Isaac
## Overview
This lab provides hands-on experience with NVIDIA Isaac's perception capabilities by implementing a basic Perception Pipeline. You will learn to configure Isaac ROS nodes for object detection, semantic segmentation, and sensor data processing, then integrate these components into a cohesive perception system.

## Learning Objectives
After completing this lab, you will be able to:
- Configure Isaac ROS perception nodes for object detection
- Integrate multiple perception algorithms into a single pipeline
- Process and visualize perception results
- Evaluate Perception Pipeline performance
- Troubleshoot common Perception Pipeline issues

## Prerequisites
- Completion of Module 1 (ROS 2 fundamentals)
- Completion of Module 2 (Simulation concepts)
- Basic understanding of computer vision concepts
- Access to a Jetson platform or Isaac-compatible development environment

## Hardware and Software Requirements
### Required Hardware- Jetson Orin AGX/NX or equivalent development platform
- RGB-D camera (Intel RealSense D435 or equivalent)
- Monitor and input devices for development
- Adequate power supply and cooling for Jetson platform

### Required Software- JetPack 5.1+ with Isaac ROS packages
- ROS 2 Humble Hawksbill
- Isaac Sim for testing (optional but recommended)
- Docker for containerized deployment
- Basic development tools (Git, Python 3.10+)

## Lab Setup
### Environment Configuration
1. **Verify Isaac ROS Installation**
   ```bash
   # Check for Isaac ROS packages
   apt list --installed | grep isaac-ros
   ```

2. **Set up workspace**
   ```bash
   mkdir -p ~/isaac_ws/src
   cd ~/isaac_ws
   colcon build
   source install/setup.bash
   ```

3. **Verify GPU acceleration**
   ```bash
   nvidia-smi
   # Should show GPU utilization and available memory
   ```

### Camera Configuration
For this lab, we'll use an Intel RealSense camera. Configure the camera launch file:

```xml
<!-- perception_pipeline.launch.xml -->
<launch>
  <!-- Launch RealSense camera -->
  <include file="$(find-pkg-share realsense2_camera)/launch/rs_launch.py">
    <arg name="enable_rgbd" value="true"/>
    <arg name="align_depth.enable" value="true"/>
    <arg name="pointcloud.enable" value="true"/>
  </include>

  <!-- Launch Isaac object detection -->
  <node pkg="isaac_ros_detectnet" exec="isaac_ros_detectnet" name="detectnet">
    <param name="input_topic" value="/camera/color/image_raw"/>
    <param name="output_topic" value="/detectnet/detections"/>
    <param name="model_name" value="ssd_mobilenet_v2_coco"/>
    <param name="confidence_threshold" value="0.7"/>
  </node>

  <!-- Launch Isaac segmentation -->
  <node pkg="isaac_ros_segmentation" exec="isaac_ros_segmentation" name="segmentation">
    <param name="input_image_topic" value="/camera/color/image_raw"/>
    <param name="output_topic" value="/segmentation/masks"/>
    <param name="model_name" value="deeplabv3_pascal_voc"/>
  </node>

  <!-- Launch Isaac depth processing -->
  <node pkg="isaac_ros_depth_preprocessor" exec="isaac_ros_depth_preprocessor" name="depth_preprocessor">
    <param name="image_topic" value="/camera/aligned_depth_to_color/image_raw"/>
    <param name="output_topic" value="/depth_preprocessor/depth_filtered"/>
    <param name="fill_holes" value="true"/>
    <param name="max_depth" value="3.0"/>
  </node>
</launch>
```

## Implementation Steps
### Step 1: Basic Object Detection Pipeline
Create the core object detection node:

```python
#!/usr/bin/env python3
# perception_pipeline.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribers for camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detectnet/detections',
            self.detection_callback,
            10
        )

        # Publisher for annotated images
        self.annotated_pub = self.create_publisher(
            Image,
            '/perception_pipeline/annotated_image',
            10
        )

        # Storage for detections
        self.latest_detections = None
        self.latest_image = None

        self.get_logger().info('Isaac Perception Pipeline initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_image = cv_image.copy()

            # If we have detections, annotate the image
            if self.latest_detections is not None:
                annotated_image = self.annotate_image(
                    cv_image, self.latest_detections
                )
                annotated_msg = self.cv_bridge.cv2_to_imgmsg(
                    annotated_image, encoding='bgr8'
                )
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detection_callback(self, msg):
        """Process incoming detections"""
        self.latest_detections = msg.detections

    def annotate_image(self, image, detections):
        """Annotate image with detection results"""
        annotated = image.copy()

        for detection in detections:
            # Get bounding box
            bbox = detection.bbox
            x = int(bbox.center.position.x - bbox.size_x / 2)
            y = int(bbox.center.position.y - bbox.size_y / 2)
            w = int(bbox.size_x)
            h = int(bbox.size_y)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label and confidence
            label = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score
            text = f'{label}: {confidence:.2f}'
            cv2.putText(
                annotated, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return annotated

def main(args=None):
    rclpy.init(args=args)
    perception_pipeline = IsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        perception_pipeline.get_logger().info('Shutting down perception pipeline')
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Enhanced Perception with 3D Information
Now let's create a node that combines 2D detections with depth information:

```python
#!/usr/bin/env python3
# perception_3d.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class Isaac3DPerception(Node):
    def __init__(self):
        super().__init__('isaac_3d_perception')

        self.cv_bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detectnet/detections', self.detection_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10
        )

        # Publishers
        self.object_3d_pub = self.create_publisher(
            PointStamped, '/perception_3d/object_position', 10
        )

        # Camera parameters
        self.camera_matrix = None
        self.latest_detections = None
        self.latest_depth = None

        self.get_logger().info('Isaac 3D Perception node initialized')

    def info_callback(self, msg):
        """Get camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = np.array(depth_image, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    def detection_callback(self, msg):
        """Process detections and estimate 3D positions"""
        self.latest_detections = msg.detections

        if self.latest_depth is not None and self.camera_matrix is not None:
            for detection in msg.detections:
                # Calculate 3D position from 2D detection and depth
                x_center = int(detection.bbox.center.position.x)
                y_center = int(detection.bbox.center.position.y)

                # Get depth at detection center (with some averaging)
                depth_roi = self.latest_depth[
                    max(0, y_center-10):min(self.latest_depth.shape[0], y_center+10),
                    max(0, x_center-10):min(self.latest_depth.shape[1], x_center+10)
                ]

                if depth_roi.size > 0:
                    avg_depth = np.nanmedian(depth_roi[depth_roi > 0])
                    if not np.isnan(avg_depth) and avg_depth > 0:
                        # Convert pixel coordinates to 3D world coordinates
                        point_3d = self.pixel_to_world(
                            x_center, y_center, avg_depth, self.camera_matrix
                        )

                        # Publish 3D position
                        point_msg = PointStamped()
                        point_msg.header = msg.header
                        point_msg.point.x = point_3d[0]
                        point_msg.point.y = point_3d[1]
                        point_msg.point.z = point_3d[2]
                        self.object_3d_pub.publish(point_msg)

    def pixel_to_world(self, u, v, depth, camera_matrix):
        """Convert pixel coordinates to 3D world coordinates"""
        # Camera intrinsic parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return [x, y, z]

    def image_callback(self, msg):
        """Process image (for synchronization)"""
        pass

def main(args=None):
    rclpy.init(args=args)
    perception_3d = Isaac3DPerception()

    try:
        rclpy.spin(perception_3d)
    except KeyboardInterrupt:
        perception_3d.get_logger().info('Shutting down 3D perception node')
    finally:
        perception_3d.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Perception Pipeline Launch File
Create a comprehensive launch file that brings everything together:

```python
# perception_pipeline_launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Declare launch arguments
    declare_namespace = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace'
    )

    # RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense_camera',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'pointcloud.enable': True,
            'depth_module.profile': '640x480x30',
            'color_module.profile': '640x480x30'
        }]
    )

    # Isaac object detection node
    detectnet_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet',
        name='isaac_detectnet',
        parameters=[{
            'input_topic': '/color/image_raw',
            'output_topic': '/detectnet/detections',
            'model_name': 'ssd_mobilenet_v2_coco',
            'confidence_threshold': 0.7,
            'debug_mode': False
        }],
        remappings=[
            ('/color/image_raw', '/camera/color/image_raw'),
            ('/detectnet/detections', '/detectnet/detections')
        ]
    )

    # Custom perception pipeline node
    perception_pipeline = Node(
        package='your_perception_package',
        executable='perception_pipeline',
        name='perception_pipeline',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    # 3D perception node
    perception_3d = Node(
        package='your_perception_package',
        executable='perception_3d',
        name='perception_3d',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        declare_namespace,
        realsense_node,
        detectnet_node,
        perception_pipeline,
        perception_3d
    ])
```

## Testing and Validation
### Basic Testing
1. **Launch the Perception Pipeline:**
   ```bash
   ros2 launch your_perception_package perception_pipeline_launch.py
   ```

2. **Visualize results:**
   ```bash
   # In another terminal
   rqt_image_view /perception_pipeline/annotated_image
   ```

3. **Monitor detection topics:**
   ```bash
   ros2 topic echo /detectnet/detections
   ```

### Performance Evaluation
Create a simple evaluation script to measure pipeline performance:

```python
#!/usr/bin/env python3
# evaluate_perception.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
import time

class PerceptionEvaluator(Node):
    def __init__(self):
        super().__init__('perception_evaluator')

        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.sub_detections = self.create_subscription(
            Detection2DArray, '/detectnet/detections', self.detection_callback, 10
        )

        self.last_image_time = None
        self.last_detection_time = None
        self.latency_samples = []

    def image_callback(self, msg):
        self.last_image_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def detection_callback(self, msg):
        if self.last_image_time is not None:
            detection_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            latency = detection_time - self.last_image_time
            self.latency_samples.append(latency)

            # Print average latency every 10 samples
            if len(self.latency_samples) % 10 == 0:
                avg_latency = sum(self.latency_samples[-10:]) / 10
                self.get_logger().info(f'Average latency (last 10): {avg_latency:.3f}s')

def main(args=None):
    rclpy.init(args=args)
    evaluator = PerceptionEvaluator()

    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        # Print final statistics
        if evaluator.latency_samples:
            avg_latency = sum(evaluator.latency_samples) / len(evaluator.latency_samples)
            min_latency = min(evaluator.latency_samples)
            max_latency = max(evaluator.latency_samples)

            print(f'\nPerception Pipeline Performance:')
            print(f'  Average latency: {avg_latency:.3f}s')
            print(f'  Min latency: {min_latency:.3f}s')
            print(f'  Max latency: {max_latency:.3f}s')
            print(f'  Total samples: {len(evaluator.latency_samples)}')

    evaluator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting
### Common Issues and Solutions
1. **No detections appearing:**
   - Verify camera is publishing images: `ros2 topic echo /camera/color/image_raw`
   - Check Isaac DetectNet node is running: `ros2 run isaac_ros_detectnet isaac_ros_detectnet`
   - Verify model files are properly installed

2. **High latency:**
   - Check GPU utilization: `nvidia-smi`
   - Reduce input resolution
   - Use lighter model (e.g., MobileNet instead of ResNet)

3. **Memory errors:**
   - Monitor GPU memory: `nvidia-smi`
   - Reduce batch size in model parameters
   - Clear unused GPU memory

4. **Synchronization issues:**
   - Use message filters for proper timestamp synchronization
   - Add delays to allow nodes to initialize

## Lab Deliverables
Complete the following tasks to finish the lab:

1. **Implement the basic Perception Pipeline** with object detection
2. **Enhance with 3D position estimation** using depth information
3. **Evaluate performance** using the provided evaluation script
4. **Document your results** including:
   - Average detection latency
   - Detection accuracy in your test environment
   - Any challenges encountered and solutions
   - Suggestions for improvement

## Assessment Criteria
Your lab implementation will be assessed based on:
- **Functionality**: Does the Perception Pipeline work correctly?
- **Performance**: Are latency and accuracy within acceptable ranges?
- **Code Quality**: Is the code well-structured and documented?
- **Problem Solving**: How effectively did you troubleshoot issues?
- **Analysis**: Quality of performance evaluation and insights provided

## Extensions (Optional)
For advanced students, consider implementing:
- **Semantic segmentation** integration with object detection
- **Multi-object tracking** using Isaac's tracking nodes
- **3D object detection** using point cloud data
- **Perception quality metrics** and confidence estimation

## Summary
This lab provided hands-on experience with Isaac's perception capabilities, from basic object detection to 3D position estimation. You learned to configure, integrate, and evaluate perception components for robotic applications. These skills form the foundation for more complex perception systems in real-world robotic applications.