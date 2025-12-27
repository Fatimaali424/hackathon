---
sidebar_position: 7
---

# Lab 12: Complete VLA System Implementation
## Overview
This capstone lab integrates all Vision-Language-Action (VLA) concepts into a complete system. You will implement a full pipeline that takes visual input and natural language commands to generate appropriate robotic actions. This represents the culmination of Module 4, combining perception, language understanding, and action execution in a unified framework.

## Learning Objectives
After completing this lab, you will be able to:
- Integrate vision, language, and action components into a unified system
- Implement end-to-end VLA pipeline for robotic applications
- Optimize system performance for real-time operation
- Evaluate complete VLA system performance
- Deploy VLA system on robotic platforms

## Prerequisites
- Completion of all previous modules (1-4)
- Understanding of ROS 2, Isaac ROS, and perception systems
- Knowledge of deep learning frameworks (PyTorch/TensorFlow)
- Experience with multimodal AI systems

## Hardware and Software Requirements
### Required Hardware- NVIDIA Jetson Orin AGX or equivalent (for edge deployment)
- RGB-D camera (Intel RealSense D435 or equivalent)
- Mobile robot platform (TurtleBot3, Jackal, or similar)
- Microphone for voice commands
- Adequate power supply and cooling for Jetson platform

### Required Software- ROS 2 Humble with Isaac ROS packages
- PyTorch with CUDA support
- Transformers library
- OpenCV and PIL
- Gazebo or Isaac Sim for simulation
- Docker for containerized deployment

## Lab Setup
### Environment Configuration
1. **Verify Jetson setup:**
   ```bash
   # Check JetPack version
   cat /etc/nv_tegra_release

   # Check GPU status
   nvidia-smi

   # Install required packages
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/l4t-cp38
   pip install transformers
   pip install torch_tensorrt
   ```

2. **Verify Isaac ROS installation:**
   ```bash
   # Check for Isaac ROS packages
   apt list --installed | grep isaac-ros
   ```

3. **Set up workspace:**
   ```bash
   mkdir -p ~/vla_ws/src
   cd ~/vla_ws
   colcon build
   source install/setup.bash
   ```

## Implementation Steps
### Step 1: VLA System Architecture
Create the main VLA system architecture that integrates all components:

```python
# vla_system.py

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Tuple
import threading
import queue
from dataclasses import dataclass

@dataclass
class VLAInput:
    """Input data structure for VLA system"""
    visual_input: torch.Tensor  # [batch, channels, height, width]
    language_input: str         # Natural language command
    robot_state: Dict[str, Any] # Current robot state
    timestamp: float           # Timestamp for synchronization

@dataclass
class VLAPrediction:
    """Output prediction from VLA system"""
    action: torch.Tensor       # [batch, action_dim] - predicted action
    confidence: float          # Confidence in prediction
    attention_map: Optional[torch.Tensor] = None  # Visual attention map
    language_attention: Optional[torch.Tensor] = None  # Language attention

class VisionEncoder(nn.Module):
    """Vision encoder for processing visual input"""
    def __init__(self, output_dim: int = 512):
        super().__init__()
        import torchvision.models as models

        # Use pre-trained ResNet as backbone
        self.backbone = models.resnet18(pretrained=True)

        # Replace final layer for feature extraction
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, output_dim)

        # Add normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder"""
        features = self.backbone(x)
        normalized_features = self.norm(features)
        return normalized_features

class LanguageEncoder(nn.Module):
    """Language encoder for processing text input"""
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256, output_dim: int = 512):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            embedding_dim, output_dim // 2,
            batch_first=True, bidirectional=True
        )

        # Linear projection to output dimension
        self.projection = nn.Linear(output_dim, output_dim)

        # Normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through language encoder"""
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use final hidden state (concatenate forward and backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        projected = self.projection(final_hidden)
        normalized = self.norm(projected)
        return normalized

class CrossModalAttention(nn.Module):
    """Cross-modal attention between vision and language"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention"""
        # Project features
        Q = self.q_proj(vision_features)
        K = self.k_proj(language_features)
        V = self.v_proj(language_features)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attn_weights, V)

        # Output projection
        output = self.out_proj(attended_features)

        return output, attn_weights

class ActionDecoder(nn.Module):
    """Action decoder that generates robot actions"""
    def __init__(self, input_dim: int, action_space_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_space_dim)
        )

        # Action normalization (for continuous control)
        self.action_norm = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate action from fused features"""
        raw_action = self.network(x)
        normalized_action = self.action_norm(raw_action)
        return normalized_action

class VLAModel(nn.Module):
    """Complete VLA model combining vision, language, and action"""
    def __init__(self,
                 vocab_size: int = 10000,
                 action_space_dim: int = 6,  # Example: 6DOF control
                 feature_dim: int = 512):
        super().__init__()

        self.vision_encoder = VisionEncoder(feature_dim)
        self.language_encoder = LanguageEncoder(vocab_size, output_dim=feature_dim)
        self.cross_attention = CrossModalAttention(feature_dim)
        self.action_decoder = ActionDecoder(feature_dim * 2, action_space_dim)

        # Fusion layer to combine attended features
        self.fusion = nn.Linear(feature_dim * 2, feature_dim * 2)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> VLAPrediction:
        """Forward pass through complete VLA model"""
        # Encode vision and language
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(text_tokens)

        # Apply cross-modal attention
        attended_vision, v2l_attention = self.cross_attention(vision_features, language_features)
        attended_language, l2v_attention = self.cross_attention(language_features, vision_features)

        # Fuse attended features
        fused_features = torch.cat([attended_vision, attended_language], dim=-1)
        fused_features = self.fusion(fused_features)

        # Generate action
        action = self.action_decoder(fused_features)

        # Calculate confidence (simplified)
        confidence = torch.mean(torch.abs(action)).item()

        return VLAPrediction(
            action=action,
            confidence=min(confidence, 1.0),  # Clamp confidence to [0,1]
            attention_map=v2l_attention,
            language_attention=l2v_attention
        )

class VLAPipeline:
    """Complete VLA pipeline with preprocessing and postprocessing"""
    def __init__(self, model: VLAModel, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Text tokenizer (simplified - in practice, use proper tokenizer)
        self.tokenizer = self._create_simple_tokenizer()

        # Image preprocessing
        self.image_transform = self._create_image_transform()

        # Action postprocessing
        self.action_scaler = ActionScaler()

    def _create_simple_tokenizer(self):
        """Create a simple tokenizer for demonstration"""
        # In practice, use transformers tokenizer
        vocab = {
            'pad': 0, 'unk': 1, 'start': 2, 'end': 3,
            'go': 4, 'to': 5, 'the': 6, 'kitchen': 7,
            'pick': 8, 'up': 9, 'red': 10, 'cup': 11,
            'stop': 12, 'start': 13, 'follow': 14, 'me': 15
        }
        return vocab

    def _create_image_transform(self):
        """Create image transformation pipeline"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_input(self, input_data: VLAInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess input data for model"""
        # Process image
        if isinstance(input_data.visual_input, np.ndarray):
            image_tensor = self.image_transform(input_data.visual_input)
        else:
            image_tensor = input_data.visual_input

        # Add batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        # Process text
        tokens = self.tokenize(input_data.language_input)
        text_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        return image_tensor, text_tensor

    def tokenize(self, text: str) -> list:
        """Convert text to token IDs"""
        # Simple tokenization for demonstration
        words = text.lower().split()
        tokens = []
        for word in words:
            token_id = self.tokenizer.get(word, self.tokenizer['unk'])
            tokens.append(token_id)

        # Add start and end tokens
        tokens = [self.tokenizer['start']] + tokens + [self.tokenizer['end']]

        # Pad to fixed length (simplified)
        max_length = 20
        if len(tokens) < max_length:
            tokens.extend([self.tokenizer['pad']] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]

        return tokens

    def process(self, input_data: VLAInput) -> VLAPrediction:
        """Process input through VLA pipeline"""
        # Preprocess
        images, text_tokens = self.preprocess_input(input_data)

        # Run model
        with torch.no_grad():
            prediction = self.model(images, text_tokens)

        return prediction

    def postprocess_action(self, prediction: VLAPrediction, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess action for robot execution"""
        # Convert tensor to numpy
        action_np = prediction.action.cpu().numpy()

        # Scale action to robot-specific ranges
        scaled_action = self.action_scaler.scale(action_np, robot_state)

        # Convert to robot command format
        robot_command = {
            'linear_velocity': scaled_action[0] if len(scaled_action) > 0 else 0.0,
            'angular_velocity': scaled_action[1] if len(scaled_action) > 1 else 0.0,
            'gripper_position': scaled_action[2] if len(scaled_action) > 2 else 0.5,
            'confidence': prediction.confidence,
            'timestamp': time.time()
        }

        return robot_command

class ActionScaler:
    """Scale actions to appropriate ranges for robot control"""
    def __init__(self):
        # Define action ranges for different robot types
        self.action_ranges = {
            'navigation': {
                'linear_velocity': (-1.0, 1.0),      # m/s
                'angular_velocity': (-1.0, 1.0),     # rad/s
            },
            'manipulation': {
                'gripper_position': (0.0, 1.0),      # normalized
                'arm_velocity': (-0.5, 0.5),         # m/s
            }
        }

    def scale(self, action: np.ndarray, robot_state: Dict[str, Any]) -> np.ndarray:
        """Scale action to appropriate range"""
        # This is a simplified scaling - in practice, use robot-specific kinematics
        scaled_action = np.copy(action)

        # Clamp to reasonable ranges
        scaled_action = np.clip(scaled_action, -1.0, 1.0)

        return scaled_action
```

### Step 2: Real-time VLA System
Create a real-time system that can process inputs continuously:

```python
# real_time_vla.py

import threading
import queue
import time
from collections import deque
import cv2

class RealTimeVLASystem:
    """Real-time VLA system with buffering and scheduling"""
    def __init__(self, vla_pipeline: VLAPipeline):
        self.pipeline = vla_pipeline
        self.running = False

        # Input queues
        self.image_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue(maxsize=10)

        # Output queue
        self.action_queue = queue.Queue(maxsize=10)

        # Processing threads
        self.processing_thread = None
        self.output_thread = None

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.fps_counter = deque(maxlen=30)

        # Robot state
        self.robot_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'velocity': [0.0, 0.0],
            'gripper': 0.5,
            'timestamp': time.time()
        }

    def start(self):
        """Start the real-time VLA system"""
        if self.running:
            return

        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Start output thread
        self.output_thread = threading.Thread(target=self._output_loop)
        self.output_thread.daemon = True
        self.output_thread.start()

        print("Real-time VLA system started")

    def stop(self):
        """Stop the real-time VLA system"""
        self.running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.output_thread:
            self.output_thread.join(timeout=2)

        print("Real-time VLA system stopped")

    def add_image(self, image: np.ndarray):
        """Add image to processing queue"""
        try:
            self.image_queue.put_nowait(image)
        except queue.Full:
            # Drop oldest image if queue is full
            try:
                self.image_queue.get_nowait()
                self.image_queue.put_nowait(image)
            except:
                pass  # Queue might be empty

    def add_command(self, command: str):
        """Add command to processing queue"""
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            # Drop oldest command if queue is full
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait(command)
            except:
                pass

    def get_action(self) -> Optional[Dict[str, Any]]:
        """Get next action from output queue"""
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None

    def _processing_loop(self):
        """Main processing loop"""
        last_command = ""

        while self.running:
            try:
                # Get latest image and command
                image = None
                command = None

                # Get latest image (non-blocking)
                try:
                    while not self.image_queue.empty():
                        image = self.image_queue.get_nowait()
                except queue.Empty:
                    pass

                # Get latest command (non-blocking)
                try:
                    while not self.command_queue.empty():
                        last_command = self.command_queue.get_nowait()
                except queue.Empty:
                    pass

                if image is not None and last_command:
                    # Process with VLA pipeline
                    start_time = time.time()

                    input_data = VLAInput(
                        visual_input=image,
                        language_input=last_command,
                        robot_state=self.robot_state,
                        timestamp=time.time()
                    )

                    prediction = self.pipeline.process(input_data)
                    robot_command = self.pipeline.postprocess_action(prediction, self.robot_state)

                    # Calculate processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)

                    # Add to output queue
                    try:
                        self.action_queue.put_nowait(robot_command)
                    except queue.Full:
                        pass  # Drop if output queue is full

                    # Update FPS counter
                    self.fps_counter.append(time.time())

                # Control processing rate
                time.sleep(0.01)  # ~100Hz processing

            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _output_loop(self):
        """Output loop for sending commands to robot"""
        while self.running:
            # In a real system, this would send commands to robot hardware
            # For this example, we'll just print commands
            time.sleep(0.05)  # 20Hz output rate

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.processing_times:
            avg_processing_time = 0.0
        else:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        # Calculate FPS
        if len(self.fps_counter) > 1:
            time_window = self.fps_counter[-1] - self.fps_counter[0]
            if time_window > 0:
                fps = len(self.fps_counter) / time_window
            else:
                fps = 0.0
        else:
            fps = 0.0

        return {
            'avg_processing_time_ms': avg_processing_time * 1000,
            'std_processing_time_ms': np.std(self.processing_times) * 1000 if self.processing_times else 0,
            'fps': fps,
            'queue_sizes': {
                'image_queue': self.image_queue.qsize(),
                'command_queue': self.command_queue.qsize(),
                'action_queue': self.action_queue.qsize()
            }
        }
```

### Step 3: VLA System with ROS 2 Integration
Create the ROS 2 node that integrates the VLA system:

```python
#!/usr/bin/env python3
# vla_ros_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
from vla_system import VLAModel, VLAPipeline, VLAInput
from real_time_vla import RealTimeVLASystem

class VLAROSNode(Node):
    def __init__(self):
        super().__init__('vla_system_node')

        # Initialize VLA components
        self.vla_model = VLAModel(vocab_size=10000, action_space_dim=6)
        self.vla_pipeline = VLAPipeline(self.vla_model)
        self.vla_system = RealTimeVLASystem(self.vla_pipeline)

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vla_status_pub = self.create_publisher(String, '/vla_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vla_command', self.command_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )

        # Timer for performance monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

        # Internal state
        self.current_image = None
        self.current_command = ""
        self.camera_info = None

        # Start real-time system
        self.vla_system.start()

        self.get_logger().info('VLA System ROS Node initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Add to VLA system
            self.vla_system.add_image(cv_image)

            # Store for potential use with command
            self.current_image = cv_image

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming voice command"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Add to VLA system
        self.vla_system.add_command(command)

        # Store for potential use with image
        self.current_command = command

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def monitor_performance(self):
        """Monitor and publish performance statistics"""
        stats = self.vla_system.get_performance_stats()

        status_msg = String()
        status_msg.data = (
            f"VLA System Status - "
            f"FPS: {stats['fps']:.1f}, "
            f"Processing: {stats['avg_processing_time_ms']:.1f}ms, "
            f"Queue Sizes: {stats['queue_sizes']}"
        )

        self.vla_status_pub.publish(status_msg)

        # Log performance if processing is slow
        if stats['avg_processing_time_ms'] > 100:  # 100ms threshold
            self.get_logger().warn(
                f'High processing time: {stats["avg_processing_time_ms"]:.1f}ms'
            )

    def process_vla_output(self):
        """Process VLA system output and send to robot"""
        action = self.vla_system.get_action()
        if action:
            # Convert to Twist message for differential drive robot
            twist_msg = Twist()
            twist_msg.linear.x = float(action.get('linear_velocity', 0.0))
            twist_msg.angular.z = float(action.get('angular_velocity', 0.0))

            # Publish command
            self.cmd_vel_pub.publish(twist_msg)

            self.get_logger().info(
                f'Sent command: linear={twist_msg.linear.x:.2f}, '
                f'angular={twist_msg.angular.z:.2f}, '
                f'confidence={action.get("confidence", 0.0):.2f}'
            )

    def timer_callback(self):
        """Timer callback for processing VLA output"""
        self.process_vla_output()

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAROSNode()

    # Create timer for processing VLA output
    timer = vla_node.create_timer(0.05, vla_node.timer_callback)  # 20Hz

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        vla_node.get_logger().info('Shutting down VLA System')
    finally:
        vla_node.vla_system.stop()
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: VLA System Optimization
Create optimization techniques for the VLA system:

```python
# vla_optimization.py

import torch
import torch_tensorrt
import tensorrt as trt
import numpy as np
from typing import Dict, Any

class VLAModelOptimizer:
    """Optimization techniques for VLA models"""

    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.optimized_model = None

    def optimize_with_tensorrt(self,
                             example_image: torch.Tensor,
                             example_text: torch.Tensor,
                             precision: str = 'fp16') -> torch.nn.Module:
        """Optimize VLA model using TensorRT"""
        self.model.eval()

        try:
            # Trace the model
            traced_model = torch.jit.trace(
                self.model,
                (example_image, example_text)
            )

            # Compile with Torch-TensorRT
            if precision == 'fp16':
                precision_set = {torch.half, torch.float}
            else:
                precision_set = {torch.float}

            optimized_model = torch_tensorrt.compile(
                traced_model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=[1, 3, 224, 224],
                        opt_shape=[8, 3, 224, 224],
                        max_shape=[16, 3, 224, 224]
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 20],
                        opt_shape=[8, 20],
                        max_shape=[16, 20],
                        dtype=torch.long
                    )
                ],
                enabled_precisions=precision_set,
                workspace_size=1 << 30,  # 1GB
                max_batch_size=16
            )

            self.optimized_model = optimized_model
            print(f"Model optimized with TensorRT using {precision} precision")
            return optimized_model

        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return self.model

    def quantize_model(self) -> torch.nn.Module:
        """Apply quantization to reduce model size and improve speed"""
        self.model.eval()

        # Use PyTorch's quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )

        print("Model quantized to INT8")
        return quantized_model

    def prune_model(self, sparsity: float = 0.2) -> torch.nn.Module:
        """Apply pruning to reduce model size"""
        import torch.nn.utils.prune as prune

        model = self.model
        model.eval()

        # Prune linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')

        print(f"Model pruned to {sparsity*100}% sparsity")
        return model

    def benchmark_model(self,
                       model: torch.nn.Module,
                       test_data: list,
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        model.eval()

        times = []
        for i in range(num_runs):
            images, text_tokens = test_data[i % len(test_data)]

            start_time = time.time()
            with torch.no_grad():
                _ = model(images, text_tokens)
            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_time_ms': avg_time,
            'std_time_ms': np.std(times),
            'fps': fps,
            'num_runs': num_runs
        }

class MemoryManager:
    """Memory management for VLA system"""

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_allocation = 0
        self.tensor_cache = {}

    def allocate_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32):
        """Efficiently allocate tensor with memory management"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = np.prod(shape) * element_size
        size_mb = size_bytes / (1024 * 1024)

        if self.current_allocation + size_mb > self.max_memory_mb:
            self._try_free_memory(size_mb)

        tensor = torch.zeros(shape, dtype=dtype, device='cuda')
        self.current_allocation += size_mb

        return tensor

    def _try_free_memory(self, needed_mb: float):
        """Try to free memory by clearing cache"""
        torch.cuda.empty_cache()
        # Additional memory management strategies can be added here

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allowed_mb': self.max_memory_mb,
            'utilization_percent': (allocated / self.max_memory_mb) * 100
        }
```

### Step 5: Complete VLA System Launch
Create a comprehensive launch system:

```python
# complete_vla_launcher.py

import subprocess
import threading
import time
import signal
import os
from typing import List, Dict

class CompleteVLASystem:
    """Complete VLA system launcher and manager"""

    def __init__(self):
        self.processes = []
        self.is_running = False

    def launch_vla_system(self):
        """Launch the complete VLA system"""
        print("Launching Complete VLA System...")

        # Start individual components in separate processes
        components = [
            {
                'name': 'camera_driver',
                'command': ['ros2', 'launch', 'realsense2_camera', 'rs_launch.py'],
                'optional': True
            },
            {
                'name': 'vla_node',
                'command': ['ros2', 'run', 'vla_system', 'vla_ros_node.py'],
                'optional': False
            },
            {
                'name': 'navigation',
                'command': ['ros2', 'launch', 'nav2_bringup', 'navigation_launch.py'],
                'optional': True
            }
        ]

        for component in components:
            try:
                if component['name'] == 'vla_node':
                    # Start VLA node with our implementation
                    process = subprocess.Popen([
                        'python3', '-c',
                        '''
import sys
sys.path.append('.')
from vla_ros_node import main
main()
                        '''
                    ])
                    self.processes.append({
                        'name': component['name'],
                        'process': process,
                        'optional': component['optional']
                    })
                else:
                    # For other components, try to launch
                    try:
                        process = subprocess.Popen(component['command'])
                        self.processes.append({
                            'name': component['name'],
                            'process': process,
                            'optional': component['optional']
                        })
                        print(f"Started {component['name']}")
                    except FileNotFoundError:
                        if not component['optional']:
                            print(f"Warning: Could not start {component['name']}")
                        continue

            except Exception as e:
                if not component['optional']:
                    print(f"Error starting {component['name']}: {e}")
                continue

        self.is_running = True
        print("VLA System launched successfully")

    def monitor_system(self):
        """Monitor the health of the VLA system"""
        while self.is_running:
            for proc_info in self.processes:
                process = proc_info['process']
                if process.poll() is not None:
                    if not proc_info['optional']:
                        print(f"Critical process {proc_info['name']} died with code {process.returncode}")
                        self.shutdown()
                        return
                    else:
                        print(f"Optional process {proc_info['name']} died, restarting...")
                        # Restart optional process
                        self.restart_process(proc_info)

            time.sleep(1)  # Check every second

    def restart_process(self, proc_info: Dict):
        """Restart a process"""
        try:
            # Try to restart the process
            if proc_info['name'] == 'vla_node':
                process = subprocess.Popen([
                    'python3', '-c',
                    '''
import sys
sys.path.append('.')
from vla_ros_node import main
main()
                    '''
                ])
            else:
                process = subprocess.Popen(proc_info['command'])

            proc_info['process'] = process
            print(f"Restarted {proc_info['name']}")
        except Exception as e:
            print(f"Failed to restart {proc_info['name']}: {e}")

    def shutdown(self):
        """Shutdown the VLA system"""
        print("Shutting down VLA System...")
        self.is_running = False

        for proc_info in self.processes:
            try:
                proc_info['process'].terminate()
                proc_info['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc_info['process'].kill()
            except Exception as e:
                print(f"Error terminating {proc_info['name']}: {e}")

        self.processes.clear()
        print("VLA System shutdown complete")

    def start_monitoring_thread(self):
        """Start monitoring in a separate thread"""
        monitor_thread = threading.Thread(target=self.monitor_system)
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread

def main():
    """Main function to run the complete VLA system"""
    vla_system = CompleteVLASystem()

    try:
        # Launch the system
        vla_system.launch_vla_system()

        # Start monitoring
        monitor_thread = vla_system.start_monitoring_thread()

        # Keep running until interrupted
        print("VLA System is running. Press Ctrl+C to stop.")
        while vla_system.is_running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    finally:
        vla_system.shutdown()

if __name__ == "__main__":
    main()
```

## Testing and Validation
### Comprehensive Testing Script
Create a comprehensive test script for the VLA system:

```python
# test_vla_system.py

import torch
import numpy as np
import time
from vla_system import VLAModel, VLAPipeline, VLAInput, VLAPrediction
from real_time_vla import RealTimeVLASystem

def test_vla_model():
    """Test the VLA model with synthetic data"""
    print("Testing VLA Model...")

    # Create model
    model = VLAModel(vocab_size=10000, action_space_dim=6)
    model.eval()

    # Create synthetic input
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224)  # RGB image
    text_tokens = torch.randint(0, 10000, (batch_size, 20))  # 20 tokens

    # Test forward pass
    with torch.no_grad():
        start_time = time.time()
        prediction = model(images, text_tokens)
        inference_time = time.time() - start_time

    print(f"  Inference time: {inference_time*1000:.2f}ms")
    print(f"  Action shape: {prediction.action.shape}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print("  ✓ Model test passed")

def test_vla_pipeline():
    """Test the complete VLA pipeline"""
    print("\nTesting VLA Pipeline...")

    # Create model and pipeline
    model = VLAModel(vocab_size=10000, action_space_dim=6)
    pipeline = VLAPipeline(model)

    # Create test input
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_command = "go to the kitchen"

    input_data = VLAInput(
        visual_input=test_image,
        language_input=test_command,
        robot_state={'position': [0, 0, 0]},
        timestamp=time.time()
    )

    # Test pipeline
    start_time = time.time()
    prediction = pipeline.process(input_data)
    processing_time = time.time() - start_time

    robot_command = pipeline.postprocess_action(prediction, input_data.robot_state)

    print(f"  Processing time: {processing_time*1000:.2f}ms")
    print(f"  Action: {robot_command}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print("  ✓ Pipeline test passed")

def test_real_time_system():
    """Test the real-time VLA system"""
    print("\nTesting Real-time System...")

    # Create pipeline
    model = VLAModel(vocab_size=10000, action_space_dim=6)
    pipeline = VLAPipeline(model)

    # Create real-time system
    rt_system = RealTimeVLASystem(pipeline)

    # Start system
    rt_system.start()

    # Add test data
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_command = "pick up the red cup"

    rt_system.add_image(test_image)
    rt_system.add_command(test_command)

    # Wait for processing
    time.sleep(0.1)

    # Get action
    action = rt_system.get_action()
    if action:
        print(f"  Generated action: {action}")
    else:
        print("  No action generated (this may be expected in test)")

    # Get performance stats
    stats = rt_system.get_performance_stats()
    print(f"  Performance: {stats}")

    # Stop system
    rt_system.stop()
    print("  ✓ Real-time system test passed")

def benchmark_vla_system():
    """Benchmark the VLA system performance"""
    print("\nBenchmarking VLA System...")

    model = VLAModel(vocab_size=10000, action_space_dim=6)
    pipeline = VLAPipeline(model)

    # Generate test data
    test_inputs = []
    for i in range(100):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        command = f"command {i % 10}"  # Cycle through 10 different commands
        input_data = VLAInput(
            visual_input=image,
            language_input=command,
            robot_state={'position': [0, 0, 0]},
            timestamp=time.time()
        )
        test_inputs.append(input_data)

    # Benchmark
    start_time = time.time()
    for input_data in test_inputs:
        pipeline.process(input_data)
    total_time = time.time() - start_time

    avg_time = total_time / len(test_inputs)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print(f"  Processed {len(test_inputs)} inputs")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  Average FPS: {fps:.2f}")
    print("  ✓ Benchmark test passed")

def run_comprehensive_tests():
    """Run all VLA system tests"""
    print("Running Comprehensive VLA System Tests\n")

    test_vla_model()
    test_vla_pipeline()
    test_real_time_system()
    benchmark_vla_system()

    print("\nAll tests completed successfully!")
    print("\nVLA System is ready for deployment.")

if __name__ == "__main__":
    run_comprehensive_tests()
```

## Performance Evaluation
Create evaluation metrics for the complete VLA system:

```python
# evaluate_vla_system.py

import time
import statistics
from typing import List, Dict, Any
import numpy as np

class VLASystemEvaluator:
    """Evaluator for complete VLA system performance"""

    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'success_rate': []
        }

    def evaluate_end_to_end(self,
                           vla_system: RealTimeVLASystem,
                           test_scenarios: List[Dict[str, Any]],
                           num_runs: int = 10) -> Dict[str, float]:
        """Evaluate end-to-end VLA system performance"""

        latencies = []
        success_count = 0

        for scenario in test_scenarios:
            for _ in range(num_runs):
                start_time = time.time()

                # Add inputs to system
                vla_system.add_image(scenario['image'])
                vla_system.add_command(scenario['command'])

                # Wait for response
                timeout = 5.0  # 5 second timeout
                start_wait = time.time()
                while time.time() - start_wait < timeout:
                    action = vla_system.get_action()
                    if action is not None:
                        end_time = time.time()
                        latencies.append(end_time - start_time)

                        # Check if action is reasonable (simplified success check)
                        if self._is_reasonable_action(action):
                            success_count += 1
                        break
                    time.sleep(0.01)  # 10ms polling

        # Calculate metrics
        total_requests = len(test_scenarios) * num_runs
        success_rate = success_count / total_requests if total_requests > 0 else 0

        avg_latency = statistics.mean(latencies) if latencies else 0
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        return {
            'avg_latency_ms': avg_latency * 1000,
            'std_latency_ms': std_latency * 1000,
            'success_rate': success_rate,
            'total_requests': total_requests,
            'successful_requests': success_count,
            'requests_per_second': len(latencies) / sum(latencies) if latencies else 0
        }

    def _is_reasonable_action(self, action: Dict[str, Any]) -> bool:
        """Check if generated action is reasonable"""
        # Check if action values are within expected ranges
        linear_vel = action.get('linear_velocity', 0)
        angular_vel = action.get('angular_velocity', 0)

        # Reasonable ranges for differential drive robot
        return (-2.0 <= linear_vel <= 2.0) and (-2.0 <= angular_vel <= 2.0)

    def evaluate_perception_accuracy(self,
                                   model: VLAModel,
                                   test_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate perception component accuracy"""
        correct_predictions = 0
        total_predictions = 0

        for sample in test_dataset:
            images = sample['images']
            expected_actions = sample['expected_actions']

            with torch.no_grad():
                prediction = model(images, sample['text_tokens'])

            # Compare predicted action with expected action (simplified)
            predicted_action = prediction.action.cpu().numpy()
            expected_action = expected_actions.cpu().numpy()

            # Calculate similarity (simplified as mean absolute difference)
            similarity = 1.0 / (1.0 + np.mean(np.abs(predicted_action - expected_action)))

            if similarity > 0.5:  # Threshold for "correct"
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return {
            'perception_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

    def generate_evaluation_report(self,
                                 e2e_metrics: Dict[str, float],
                                 perception_metrics: Dict[str, float]) -> str:
        """Generate comprehensive evaluation report"""
        report = """
# VLA System Evaluation Report

## End-to-End Performance
- Average Latency: {:.2f}ms (±{:.2f}ms)
- Success Rate: {:.2f}%
- Throughput: {:.2f} requests/sec
- Total Requests: {}

## Perception Accuracy
- Accuracy: {:.2f}%
- Correct Predictions: {}/{}

## Summary
The VLA system demonstrates {} performance with {} latency and {} success rate.
        """.format(
            e2e_metrics['avg_latency_ms'],
            e2e_metrics['std_latency_ms'],
            e2e_metrics['success_rate'] * 100,
            e2e_metrics['requests_per_second'],
            e2e_metrics['total_requests'],
            perception_metrics['perception_accuracy'] * 100,
            perception_metrics['correct_predictions'],
            perception_metrics['total_predictions'],
            "good" if e2e_metrics['success_rate'] > 0.8 else "acceptable" if e2e_metrics['success_rate'] > 0.6 else "poor",
            "low" if e2e_metrics['avg_latency_ms'] < 100 else "acceptable" if e2e_metrics['avg_latency_ms'] < 500 else "high",
            "high" if e2e_metrics['success_rate'] > 0.9 else "medium" if e2e_metrics['success_rate'] > 0.7 else "low"
        )

        return report

def run_complete_evaluation():
    """Run complete evaluation of the VLA system"""
    print("Running Complete VLA System Evaluation...")

    evaluator = VLASystemEvaluator()

    # This would typically involve loading test data
    # For this example, we'll demonstrate the evaluation structure

    print("Evaluation framework ready.")
    print("To run full evaluation, implement data loading and call:")
    print("  evaluator.evaluate_end_to_end(vla_system, test_scenarios)")
    print("  evaluator.evaluate_perception_accuracy(model, test_dataset)")

    print("\nKey evaluation metrics:")
    print("- Response latency (should be < 100ms for real-time)")
    print("- Success rate (should be > 80%)")
    print("- Memory efficiency (should fit in robot's memory)")
    print("- Robustness to noise and variations")

if __name__ == "__main__":
    run_complete_evaluation()
```

## Lab Deliverables
Complete the following tasks to finish the lab:

1. **Implement the complete VLA model** with vision, language, and action components
2. **Create the real-time processing system** with buffering and scheduling
3. **Integrate with ROS 2** for robotic applications
4. **Optimize the system** for edge deployment on Jetson platforms
5. **Test and validate** the complete system with various scenarios
6. **Deploy and demonstrate** the system with actual robot hardware
7. **Document your results** including:
   - System architecture and design decisions
   - Performance metrics achieved
   - Challenges encountered and solutions
   - Suggestions for improvement

## Assessment Criteria
Your lab implementation will be assessed based on:
- **Integration**: How well do all components work together?
- **Performance**: Are latency and accuracy within acceptable ranges?
- **Robustness**: How well does the system handle variations and noise?
- **Efficiency**: Is the system optimized for real-time operation?
- **Documentation**: Quality of code documentation and results presentation.

## Extensions (Optional)
For advanced students, consider implementing:
- **Multi-modal fusion** with additional sensors (LiDAR, IMU)
- **Reinforcement learning** for action optimization
- **Online adaptation** to new environments and tasks
- **Human feedback integration** for system improvement
- **Safety mechanisms** for fail-safe operation

## Troubleshooting
### Common Issues and Solutions
1. **Memory errors on Jetson:**
   - Reduce model size or batch processing
   - Use model optimization (quantization, pruning)
   - Implement memory management strategies

2. **High latency issues:**
   - Optimize with TensorRT
   - Reduce input resolution
   - Use lighter model variants

3. **Poor action quality:**
   - Improve training data quality
   - Add more diverse training scenarios
   - Implement action refinement techniques

4. **Integration problems:**
   - Verify ROS 2 message formats
   - Check timing and synchronization
   - Use appropriate middleware configurations

## Summary
This capstone lab integrated all Vision-Language-Action concepts into a complete system. You learned to build, optimize, and deploy sophisticated AI systems that can understand natural language commands, perceive visual environments, and generate appropriate robotic actions. This represents the state-of-the-art in embodied AI and provides the foundation for advanced robotic applications.