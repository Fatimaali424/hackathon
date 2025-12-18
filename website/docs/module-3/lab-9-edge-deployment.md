---
sidebar_position: 7
---

# Lab 9: Edge Deployment and Optimization

## Overview

This lab focuses on deploying and optimizing Isaac-based robotic applications on edge computing platforms, specifically NVIDIA Jetson devices. You will learn to containerize applications, optimize for resource constraints, implement efficient memory management, and validate performance in resource-constrained environments.

## Learning Objectives

After completing this lab, you will be able to:
- Containerize Isaac applications using Docker for Jetson platforms
- Optimize neural network models for edge deployment using TensorRT
- Implement efficient resource management for real-time applications
- Profile and optimize application performance on edge devices
- Deploy applications with appropriate power and thermal management

## Prerequisites

- Completion of Module 1 (ROS 2 fundamentals)
- Completion of Module 2 (Simulation concepts)
- Completion of Module 3 (Isaac perception and planning)
- Basic understanding of Docker and containerization
- Access to a Jetson platform (Orin AGX/NX/Nano) or cross-compilation environment

## Hardware and Software Requirements

### Required Hardware
- NVIDIA Jetson Orin AGX/NX/Nano development kit
- Power supply capable of delivering required power (varies by Jetson model)
- MicroSD card (64GB+ recommended) or eMMC storage
- USB-C cable for power and data (if needed)
- Cooling solution (active cooling recommended for Orin AGX)

### Required Software
- JetPack 5.1+ installed on Jetson device
- Docker and nvidia-docker2
- Isaac ROS packages
- TensorRT development libraries
- Development tools (Git, Python 3.10+, build tools)

## Lab Setup

### Environment Preparation

1. **Verify Jetson Setup:**
   ```bash
   # Check Jetson model and JetPack version
   cat /etc/nv_tegra_release

   # Check GPU status
   nvidia-smi

   # Check available memory
   free -h
   ```

2. **Install Docker:**
   ```bash
   sudo apt update
   sudo apt install docker.io nvidia-docker2
   sudo systemctl restart docker
   sudo usermod -aG docker $USER
   ```

3. **Verify Docker with GPU support:**
   ```bash
   sudo docker run --rm --gpus all nvcr.io/nvidia/cuda:11.8-devel-ubuntu20.04 nvidia-smi
   ```

### Development Environment Setup

Create the project structure:

```bash
mkdir -p ~/edge_deployment_ws/{src,build,install,log}
cd ~/edge_deployment_ws

# Create Docker directory for container builds
mkdir -p dockerfiles
mkdir -p config
mkdir -p scripts
```

## Implementation Steps

### Step 1: Containerized Application Structure

Create a Dockerfile optimized for Jetson deployment:

```dockerfile
# dockerfiles/edge_robot_app.Dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-ros:galactic-ros-base-l4t-r35.2.1
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    vim \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Create application directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install application-specific dependencies
RUN pip3 install \
    torch==1.13.0 \
    torchvision==0.14.0 \
    torchaudio==0.13.0 \
    --index-url https://download.pytorch.org/whl/l4t-cp38

# Install TensorRT Python bindings
RUN pip3 install nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com

# Copy application code
COPY src /app/src
COPY launch /app/launch
COPY config /app/config

# Create non-root user for security
RUN useradd -m -s /bin/bash robotuser && \
    usermod -aG dialout robotuser

# Set ownership
RUN chown -R robotuser:robotuser /app
USER robotuser

# Set up ROS workspace
COPY ros_package /opt/ros_ws/src/edge_robot_app
WORKDIR /opt/ros_ws

# Build ROS package
RUN source /opt/ros/galactic/setup.bash && \
    colcon build --packages-select edge_robot_app

# Source ROS environment
RUN echo "source /opt/ros/galactic/setup.bash" >> ~/.bashrc
RUN echo "source /opt/ros_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /app
CMD ["bash", "-c", "source /opt/ros/galactic/setup.bash && source /opt/ros_ws/install/setup.bash && python3 src/main.py"]
```

Create requirements.txt for Python dependencies:

```
# requirements.txt
numpy==1.21.6
scipy==1.7.3
opencv-python==4.6.0.66
Pillow==9.3.0
requests==2.28.1
psutil==5.9.4
GPUtil==1.4.0
torch-tensorrt==1.2.0
numba==0.56.4
```

### Step 2: TensorRT Model Optimization

Create a model optimization script:

```python
# src/model_optimizer.py

import torch
import torch_tensorrt
import tensorrt as trt
import numpy as np
import logging

class ModelOptimizer:
    def __init__(self, precision='fp16', max_batch_size=1):
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)

    def optimize_torch_model(self, model, example_input, output_path):
        """
        Optimize PyTorch model using Torch-TensorRT
        """
        try:
            # Set model to evaluation mode
            model.eval()

            # Trace the model
            traced_model = torch.jit.trace(model, example_input)

            # Compile with Torch-TensorRT
            optimized_model = torch_tensorrt.compile(
                traced_model,
                inputs=[example_input],
                enabled_precisions={torch.float, torch.half} if self.precision == 'fp16' else {torch.float},
                workspace_size=1 << 30,  # 1GB workspace
                max_batch_size=self.max_batch_size
            )

            # Save optimized model
            torch.jit.save(optimized_model, output_path)
            self.logger.info(f"Model optimized and saved to {output_path}")

            return optimized_model

        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            return None

    def optimize_with_tensorrt(self, onnx_model_path, output_path, input_shape):
        """
        Optimize ONNX model using native TensorRT
        """
        try:
            # Create logger
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()

            # Parse ONNX model
            parser = trt.OnnxParser(network, logger)
            with open(onnx_model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(f"TensorRT Parser Error: {parser.get_error(error).desc()}")
                    return False

            # Configure optimization
            if self.precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                # Add calibration here if needed

            # Set workspace size
            config.max_workspace_size = 1 << 30  # 1GB

            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)

            # Save engine
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)

            self.logger.info(f"TensorRT engine saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating TensorRT engine: {e}")
            return False

    def benchmark_model(self, model, input_tensor, num_runs=100):
        """
        Benchmark model performance
        """
        import time

        # Warm up
        for _ in range(10):
            _ = model(input_tensor)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        avg_time = sum(times) / len(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        self.logger.info(f"Model benchmark - Avg: {avg_time:.2f}ms, FPS: {fps:.2f}")
        return avg_time, fps

# Example usage
if __name__ == "__main__":
    # Example: Optimize a simple model
    import torchvision.models as models

    # Load a model
    model = models.resnet18(pretrained=True)
    example_input = torch.randn(1, 3, 224, 224).cuda()

    optimizer = ModelOptimizer(precision='fp16')

    # Optimize the model
    optimized_model = optimizer.optimize_torch_model(
        model, example_input, "optimized_model.ts"
    )

    if optimized_model:
        # Benchmark both models
        print("Original model:")
        optimizer.benchmark_model(model.cuda(), example_input)

        print("Optimized model:")
        optimizer.benchmark_model(optimized_model, example_input)
```

### Step 3: Resource Management and Optimization

Create a resource manager for efficient edge deployment:

```python
# src/resource_manager.py

import psutil
import GPUtil
import threading
import time
import logging
from collections import deque
import numpy as np

class ResourceManager:
    def __init__(self, max_cpu_percent=80, max_gpu_percent=85, max_memory_percent=80):
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_percent = max_gpu_percent
        self.max_memory_percent = max_memory_percent

        self.monitoring = False
        self.adaptation_enabled = True

        self.cpu_history = deque(maxlen=30)  # 30 second history
        self.gpu_history = deque(maxlen=30)
        self.memory_history = deque(maxlen=30)

        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent

                # Monitor GPU usage (if available)
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100  # Primary GPU
                    gpu_memory_percent = gpus[0].memoryUtil * 100
                else:
                    gpu_percent = 0
                    gpu_memory_percent = 0

                # Store in history
                with self.lock:
                    self.cpu_history.append(cpu_percent)
                    self.gpu_history.append(max(gpu_percent, gpu_memory_percent))
                    self.memory_history.append(memory_percent)

                # Check if adaptation is needed
                if self.adaptation_enabled:
                    self._check_adaptation_needed()

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")

    def _check_adaptation_needed(self):
        """Check if resource adaptation is needed"""
        with self.lock:
            # Calculate recent averages
            if len(self.cpu_history) > 0:
                avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
            else:
                avg_cpu = 0

            if len(self.gpu_history) > 0:
                avg_gpu = sum(self.gpu_history) / len(self.gpu_history)
            else:
                avg_gpu = 0

            if len(self.memory_history) > 0:
                avg_memory = sum(self.memory_history) / len(self.memory_history)
            else:
                avg_memory = 0

        # Trigger adaptation if resources are overutilized
        if (avg_cpu > self.max_cpu_percent or
            avg_gpu > self.max_gpu_percent or
            avg_memory > self.max_memory_percent):
            self.logger.warning(f"Resource adaptation triggered: CPU={avg_cpu:.1f}%, GPU={avg_gpu:.1f}%, Memory={avg_memory:.1f}%")
            self._perform_adaptation()

    def _perform_adaptation(self):
        """Perform resource adaptation"""
        # This is where you'd implement adaptation strategies
        # Examples: reduce processing frequency, lower model precision, etc.
        self.logger.info("Performing resource adaptation...")

    def get_resource_stats(self):
        """Get current resource utilization statistics"""
        with self.lock:
            if len(self.cpu_history) > 0:
                cpu_stats = {
                    'current': self.cpu_history[-1] if self.cpu_history else 0,
                    'average': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
                    'peak': max(self.cpu_history) if self.cpu_history else 0
                }
            else:
                cpu_stats = {'current': 0, 'average': 0, 'peak': 0}

            if len(self.gpu_history) > 0:
                gpu_stats = {
                    'current': self.gpu_history[-1] if self.gpu_history else 0,
                    'average': sum(self.gpu_history) / len(self.gpu_history) if self.gpu_history else 0,
                    'peak': max(self.gpu_history) if self.gpu_history else 0
                }
            else:
                gpu_stats = {'current': 0, 'average': 0, 'peak': 0}

            if len(self.memory_history) > 0:
                memory_stats = {
                    'current': self.memory_history[-1] if self.memory_history else 0,
                    'average': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
                    'peak': max(self.memory_history) if self.memory_history else 0
                }
            else:
                memory_stats = {'current': 0, 'average': 0, 'peak': 0}

        return {
            'cpu': cpu_stats,
            'gpu': gpu_stats,
            'memory': memory_stats
        }

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)

class MemoryManager:
    def __init__(self, max_memory_mb=2048):
        self.max_memory_mb = max_memory_mb
        self.memory_pool = {}
        self.current_allocation = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def allocate_tensor(self, shape, dtype=np.float32):
        """Efficiently allocate tensor with memory pooling"""
        import torch

        # Calculate memory requirement
        element_size = np.dtype(dtype).itemsize
        size_bytes = np.prod(shape) * element_size
        size_mb = size_bytes / (1024 * 1024)

        with self.lock:
            if self.current_allocation + size_mb > self.max_memory_mb:
                self.logger.warning(f"Memory allocation would exceed limit: {self.current_allocation + size_mb:.1f}MB / {self.max_memory_mb}MB")
                # Try to free some memory or raise exception
                self._try_free_memory(size_mb)

            # Create tensor
            tensor = torch.zeros(shape, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype, device='cuda')
            self.current_allocation += size_mb

        return tensor

    def _try_free_memory(self, needed_mb):
        """Try to free memory by clearing cache"""
        import torch
        torch.cuda.empty_cache()
        # Additional memory management strategies can be added here

    def release_tensor(self, tensor):
        """Release tensor back to pool"""
        import torch
        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

        with self.lock:
            self.current_allocation -= size_mb
            tensor = None  # This should free the tensor

    def get_memory_stats(self):
        """Get memory usage statistics"""
        import torch
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allowed_mb': self.max_memory_mb,
            'utilization_percent': (allocated / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0
        }
```

### Step 4: Real-Time Performance Optimization

Create a real-time performance optimizer:

```python
# src/performance_optimizer.py

import time
import threading
import logging
from collections import deque
import subprocess
import os

class RealTimeOptimizer:
    def __init__(self, target_frequency=30.0, max_latency_ms=33):
        self.target_frequency = target_frequency
        self.max_latency_ms = max_latency_ms
        self.current_frequency = target_frequency

        self.execution_times = deque(maxlen=100)
        self.period_times = deque(maxlen=100)

        self.logger = logging.getLogger(__name__)
        self.adaptation_enabled = True

        # CPU affinity for real-time performance
        self.cpu_affinity = [0]  # Run on CPU 0

    def configure_real_time(self):
        """Configure real-time settings"""
        try:
            # Set CPU affinity
            os.sched_setaffinity(0, self.cpu_affinity)
            self.logger.info(f"Set CPU affinity to {self.cpu_affinity}")

            # Try to set real-time priority (requires appropriate permissions)
            try:
                import ctypes
                from ctypes import util
                libc = ctypes.CDLL(util.find_library("c"))

                # Try to set SCHED_FIFO with high priority
                param = ctypes.c_int(80)  # High priority
                result = libc.sched_setscheduler(
                    os.getpid(),
                    ctypes.c_int(1),  # SCHED_FIFO
                    ctypes.byref(param)
                )

                if result == 0:
                    self.logger.info("Set real-time scheduling (SCHED_FIFO)")
                else:
                    self.logger.warning("Could not set real-time scheduling")

            except Exception as e:
                self.logger.warning(f"Real-time scheduling setup failed: {e}")

        except Exception as e:
            self.logger.warning(f"Real-time configuration failed: {e}")

    def start_performance_monitoring(self):
        """Start performance monitoring in background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_performance(self):
        """Monitor performance in background"""
        while self.monitoring:
            if len(self.execution_times) > 10:
                avg_execution = sum(self.execution_times) / len(self.execution_times)
                max_execution = max(self.execution_times)

                if max_execution > self.max_latency_ms:
                    self.logger.warning(f"High latency detected: {max_execution:.2f}ms > {self.max_latency_ms}ms")

                    if self.adaptation_enabled:
                        self._adapt_performance()

            time.sleep(1)  # Check every second

    def _adapt_performance(self):
        """Adapt performance parameters"""
        if len(self.execution_times) > 10:
            avg_execution = sum(self.execution_times) / len(self.execution_times)

            # Adjust frequency based on performance
            if avg_execution > self.max_latency_ms * 0.8:  # 80% of max latency
                # Reduce frequency to ensure timing
                self.current_frequency = max(10, self.current_frequency * 0.8)
                self.logger.info(f"Reduced frequency to {self.current_frequency:.1f}Hz for timing compliance")
            elif avg_execution < self.max_latency_ms * 0.5:  # 50% of max latency
                # Can potentially increase frequency
                self.current_frequency = min(self.target_frequency, self.current_frequency * 1.1)
                self.logger.info(f"Increased frequency to {self.current_frequency:.1f}Hz")

    def time_critical_loop(self, work_function, *args, **kwargs):
        """Execute time-critical work with performance monitoring"""
        start_time = time.perf_counter()

        # Execute the work function
        result = work_function(*args, **kwargs)

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Record execution time
        self.execution_times.append(execution_time_ms)

        # Calculate required period
        required_period_ms = 1000.0 / self.current_frequency

        # Calculate sleep time to maintain frequency
        sleep_time = (required_period_ms - execution_time_ms) / 1000.0

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Missed deadline
            self.logger.warning(f"Missed timing deadline: {execution_time_ms:.2f}ms > {required_period_ms:.2f}ms")

        # Record actual period time
        actual_end = time.perf_counter()
        period_time_ms = (actual_end - start_time) * 1000
        self.period_times.append(period_time_ms)

        return result

    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.execution_times) > 0:
            avg_execution = sum(self.execution_times) / len(self.execution_times)
            max_execution = max(self.execution_times)
            min_execution = min(self.execution_times)
        else:
            avg_execution = max_execution = min_execution = 0

        if len(self.period_times) > 0:
            avg_period = sum(self.period_times) / len(self.period_times)
        else:
            avg_period = 0

        return {
            'current_frequency': self.current_frequency,
            'target_frequency': self.target_frequency,
            'avg_execution_ms': avg_execution,
            'max_execution_ms': max_execution,
            'min_execution_ms': min_execution,
            'avg_period_ms': avg_period,
            'missed_deadlines': sum(1 for t in self.execution_times if t > self.max_latency_ms)
        }

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
```

### Step 5: Main Application with Optimization

Create the main application file:

```python
# src/main.py

#!/usr/bin/env python3
"""
Edge Deployment Optimized Application
This application demonstrates optimized deployment techniques for Jetson platforms
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np
from resource_manager import ResourceManager, MemoryManager
from performance_optimizer import RealTimeOptimizer
from model_optimizer import ModelOptimizer
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedEdgeApplication(Node):
    def __init__(self):
        super().__init__('optimized_edge_app')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.resource_manager = ResourceManager()
        self.memory_manager = MemoryManager(max_memory_mb=1024)
        self.performance_optimizer = RealTimeOptimizer(target_frequency=20.0)

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize optimized model
        self.model = None
        self.initialize_optimized_model()

        # Processing state
        self.processing_enabled = True
        self.frame_count = 0

        # Start monitoring
        self.resource_manager.start_monitoring()
        self.performance_optimizer.start_performance_monitoring()

        self.get_logger().info('Optimized Edge Application initialized')

    def initialize_optimized_model(self):
        """Initialize optimized neural network model"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info("CUDA available, using GPU")
            else:
                self.device = torch.device('cpu')
                logger.warning("CUDA not available, using CPU")
                return

            # Load a pre-trained model (example: MobileNet for efficiency)
            import torchvision.models as models
            model = models.mobilenet_v2(pretrained=True)
            model.eval()

            # Create example input for tracing
            example_input = torch.randn(1, 3, 224, 224, device=self.device)

            # Optimize the model
            optimizer = ModelOptimizer(precision='fp16')
            self.model = optimizer.optimize_torch_model(
                model, example_input, "/tmp/optimized_model.ts"
            )

            if self.model is not None:
                self.model = self.model.to(self.device)
                logger.info("Model optimized and loaded successfully")
            else:
                logger.error("Failed to optimize model, falling back to CPU")
                self.model = model.cpu()
                self.device = torch.device('cpu')

        except Exception as e:
            logger.error(f"Error initializing optimized model: {e}")
            self.processing_enabled = False

    def image_callback(self, msg):
        """Process incoming image with optimization"""
        if not self.processing_enabled:
            return

        try:
            # Convert ROS image to tensor
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Preprocess image for model input
            tensor_image = self.preprocess_image(cv_image)

            # Process with optimized model
            with torch.no_grad():
                start_time = time.time()
                result = self.model(tensor_image)
                inference_time = (time.time() - start_time) * 1000  # ms

            # Process results
            self.process_inference_result(result, inference_time)

            # Update frame count
            self.frame_count += 1

            # Log performance periodically
            if self.frame_count % 50 == 0:
                self.log_performance()

        except Exception as e:
            logger.error(f"Error processing image: {e}")

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize image to model input size
        import cv2
        resized = cv2.resize(image, (224, 224))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device) / 255.0

        # Normalize with ImageNet means and stds
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        tensor = (tensor - imagenet_mean) / imagenet_std

        return tensor

    def process_inference_result(self, result, inference_time):
        """Process model inference results"""
        # Get top prediction
        probabilities = torch.nn.functional.softmax(result[0], dim=0)
        top_prob, top_cat = torch.topk(probabilities, 1)

        # Log results (in a real application, you'd use these for control)
        if top_prob.item() > 0.5:  # Confidence threshold
            self.get_logger().info(
                f"Prediction: {top_cat.item()}, Confidence: {top_prob.item():.3f}, "
                f"Inference time: {inference_time:.2f}ms"
            )

        # Publish a simple command based on inference
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward slowly
        cmd.angular.z = 0.0  # No rotation
        self.cmd_pub.publish(cmd)

    def log_performance(self):
        """Log performance statistics"""
        # Get resource stats
        resource_stats = self.resource_manager.get_resource_stats()

        # Get memory stats
        memory_stats = self.memory_manager.get_memory_stats()

        # Get performance stats
        perf_stats = self.performance_optimizer.get_performance_stats()

        # Log all stats
        self.get_logger().info(
            f"Performance - Freq: {perf_stats['current_frequency']:.1f}Hz, "
            f"Exec: {perf_stats['avg_execution_ms']:.2f}ms, "
            f"CPU: {resource_stats['cpu']['current']:.1f}%, "
            f"GPU: {resource_stats['gpu']['current']:.1f}%, "
            f"Mem: {memory_stats['allocated_mb']:.1f}MB"
        )

    def destroy_node(self):
        """Clean up resources"""
        self.resource_manager.stop_monitoring()
        self.performance_optimizer.stop_monitoring()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # Configure real-time settings
    app = OptimizedEdgeApplication()
    app.performance_optimizer.configure_real_time()

    try:
        rclpy.spin(app)
    except KeyboardInterrupt:
        app.get_logger().info('Shutting down optimized edge application')
    finally:
        app.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing and Validation

### Performance Testing Script

Create a comprehensive testing script:

```python
# test_performance.py

import subprocess
import time
import json
import csv
from datetime import datetime

def run_performance_test():
    """Run comprehensive performance test"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: Docker build time
    print("Testing Docker build performance...")
    start_time = time.time()
    try:
        subprocess.run([
            'docker', 'build',
            '-f', 'dockerfiles/edge_robot_app.Dockerfile',
            '-t', 'edge-robot-test:latest',
            '.'
        ], check=True, timeout=600)  # 10 minute timeout
        build_time = time.time() - start_time
        results['tests']['docker_build_time'] = build_time
        print(f"Docker build completed in {build_time:.2f}s")
    except subprocess.TimeoutExpired:
        print("Docker build timed out")
        results['tests']['docker_build_time'] = -1
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e}")
        results['tests']['docker_build_time'] = -1

    # Test 2: Resource utilization
    print("Testing resource utilization...")
    import psutil
    import GPUtil

    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent

    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_percent = gpus[0].load * 100
        gpu_memory_percent = gpus[0].memoryUtil * 100
    else:
        gpu_percent = gpu_memory_percent = 0

    results['tests']['idle_cpu_percent'] = cpu_percent
    results['tests']['idle_memory_percent'] = memory_percent
    results['tests']['idle_gpu_percent'] = gpu_percent
    results['tests']['idle_gpu_memory_percent'] = gpu_memory_percent

    # Test 3: Model optimization performance
    print("Testing model optimization...")
    import torch
    import time

    # Create a simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10)
    ).cuda()
    model.eval()

    example_input = torch.randn(1, 1000).cuda()

    # Test original model
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(example_input)
    original_time = (time.time() - start_time) / 100  # Average per inference

    # Test with torch.jit optimization
    traced_model = torch.jit.trace(model, example_input)
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = traced_model(example_input)
    traced_time = (time.time() - start_time) / 100  # Average per inference

    results['tests']['original_model_time_ms'] = original_time * 1000
    results['tests']['traced_model_time_ms'] = traced_time * 1000
    results['tests']['optimization_improvement'] = (original_time - traced_time) / original_time * 100

    # Save results
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Performance test completed. Results saved to performance_results.json")
    return results

if __name__ == "__main__":
    results = run_performance_test()

    # Print summary
    print("\n=== Performance Test Summary ===")
    for test, result in results['tests'].items():
        if isinstance(result, float):
            print(f"{test}: {result:.3f}")
        else:
            print(f"{test}: {result}")
```

## Power and Thermal Management

Create a power management script:

```python
# scripts/power_manager.py

#!/usr/bin/env python3
"""
Power and thermal management for Jetson edge deployment
"""

import subprocess
import time
import threading
import logging

class PowerManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.power_mode = 'balanced'

    def set_power_mode(self, mode):
        """Set Jetson power mode"""
        modes = {
            'max_performance': 'MAXN',
            'balanced': 'MODE_15W',  # Adjust based on your Jetson model
            'power_efficient': 'MODE_10W'
        }

        if mode in modes:
            try:
                subprocess.run(['sudo', 'nvpmodel', '-m', modes[mode]], check=True)
                self.power_mode = mode
                self.logger.info(f"Set power mode to {mode}")
                return True
            except subprocess.CalledProcessError:
                self.logger.error(f"Failed to set power mode {mode}")
                return False
        return False

    def get_temperature(self):
        """Get Jetson temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_mC = int(f.read().strip())
                return temp_mC / 1000.0  # Convert to Celsius
        except Exception:
            return 0.0

    def start_thermal_protection(self):
        """Start thermal monitoring and protection"""
        self.monitoring = True
        self.thermal_thread = threading.Thread(target=self._thermal_monitor_loop)
        self.thermal_thread.daemon = True
        self.thermal_thread.start()

    def _thermal_monitor_loop(self):
        """Thermal monitoring loop"""
        max_temp = 85.0  # Celsius
        hysteresis = 5.0

        while self.monitoring:
            current_temp = self.get_temperature()

            if current_temp > max_temp:
                self._activate_cooling()
            elif current_temp < (max_temp - hysteresis):
                self._restore_normal()

            time.sleep(2)  # Check every 2 seconds

    def _activate_cooling(self):
        """Activate cooling measures"""
        self.logger.warning(f"High temperature detected: {self.get_temperature():.1f}°C")
        # Reduce performance, activate fans, etc.
        self.set_power_mode('power_efficient')

    def _restore_normal(self):
        """Restore normal operation"""
        if self.power_mode != 'balanced':
            self.set_power_mode('balanced')

    def start_power_logging(self):
        """Start power consumption logging"""
        self.power_logging = True
        self.power_thread = threading.Thread(target=self._power_log_loop)
        self.power_thread.daemon = True
        self.power_thread.start()

    def _power_log_loop(self):
        """Power logging loop"""
        while self.power_logging:
            try:
                # Read power consumption (example for Jetson Xavier)
                with open('/sys/bus/i2c/devices/0-0040/hwmon/hwmon0/power1_input', 'r') as f:
                    power_mw = int(f.read().strip())
                    self.logger.info(f"Power consumption: {power_mw/1000:.2f}W")
            except Exception:
                pass  # Power monitoring not available on all Jetson models

            time.sleep(10)  # Log every 10 seconds

def main():
    power_manager = PowerManager()

    # Set to balanced mode initially
    power_manager.set_power_mode('balanced')

    # Start thermal protection
    power_manager.start_thermal_protection()

    # Start power logging
    power_manager.start_power_logging()

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down power manager")

if __name__ == "__main__":
    main()
```

## Deployment Scripts

Create deployment scripts for easy deployment:

```bash
#!/bin/bash
# scripts/deploy_jetson.sh

set -e  # Exit on error

echo "Starting Jetson deployment..."

# Check if running on Jetson
if ! [ -f /etc/nv_tegra_release ]; then
    echo "This script should be run on a Jetson device"
    exit 1
fi

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Docker is required but not installed"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found"
    exit 1
fi

# Set power mode for optimal performance
echo "Setting power mode..."
sudo nvpmodel -m MAXN

# Configure Jetson clocks
echo "Configuring clocks..."
sudo jetson_clocks

# Build Docker image
echo "Building Docker image..."
docker build -f dockerfiles/edge_robot_app.Dockerfile -t edge-robot-app:latest .

# Run the application with GPU access
echo "Starting application..."
docker run -it --rm \
    --gpus all \
    --privileged \
    --network host \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --volume /tmp:/tmp \
    --device /dev:/dev \
    --device /dev/i2c-1:/dev/i2c-1 \
    edge-robot-app:latest

echo "Deployment completed!"
```

```bash
#!/bin/bash
# scripts/build_cross_platform.sh

set -e

echo "Starting cross-platform build for Jetson..."

# Check for NVIDIA Container Runtime
if ! command -v nvidia-docker &> /dev/null; then
    echo "NVIDIA Docker runtime is required"
    exit 1
fi

# Build for Jetson architecture
echo "Building for aarch64 (Jetson)..."
docker buildx build \
    --platform linux/arm64 \
    -f dockerfiles/edge_robot_app.Dockerfile \
    -t edge-robot-app:latest \
    --load .

echo "Cross-platform build completed!"
```

## Troubleshooting

### Common Issues and Solutions

1. **Docker build fails with memory issues:**
   - Increase swap space on Jetson
   - Build with reduced parallelism: `docker build --memory=2g`
   - Use build cache effectively

2. **Model optimization fails:**
   - Check TensorRT version compatibility
   - Verify GPU memory availability
   - Use appropriate precision (FP16 vs INT8)

3. **High power consumption:**
   - Use power-efficient inference modes
   - Reduce processing frequency when possible
   - Implement adaptive performance scaling

4. **Thermal throttling:**
   - Improve cooling solution
   - Use power management scripts
   - Optimize algorithms for efficiency

## Lab Deliverables

Complete the following tasks to finish the lab:

1. **Containerize the application** using the provided Dockerfile
2. **Implement model optimization** with TensorRT
3. **Create resource management** system with monitoring
4. **Deploy and test** on Jetson hardware
5. **Document your results** including:
   - Performance improvements achieved
   - Resource utilization statistics
   - Power consumption measurements
   - Any challenges encountered and solutions

## Assessment Criteria

Your lab implementation will be assessed based on:
- **Functionality**: Does the optimized application work correctly?
- **Performance**: Are optimization improvements significant?
- **Efficiency**: How well does it manage resources?
- **Code Quality**: Is the code well-structured and documented?
- **Problem Solving**: How effectively did you optimize for edge constraints?

## Extensions (Optional)

For advanced students, consider implementing:
- **Adaptive precision scaling** based on available resources
- **Multi-model optimization** with shared resources
- **Edge-cloud hybrid deployment** strategies
- **Real-time performance prediction** and adaptation

## Summary

This lab provided hands-on experience with edge deployment optimization for Isaac-based robotic applications. You learned to containerize applications, optimize neural networks with TensorRT, manage system resources efficiently, and deploy applications with appropriate power and thermal management. These skills are essential for deploying robotic systems in resource-constrained edge environments.