---
sidebar_position: 5
---

# Lab 10: Basic Vision-Language Integration
## Overview
This lab provides hands-on experience with basic vision-language integration techniques, focusing on connecting visual perception with natural language understanding. You will implement fundamental approaches for grounding language in visual contexts and explore how robots can use both modalities together for more intelligent behavior.

## Learning Objectives
After completing this lab, you will be able to:
- Implement basic vision-language models that connect images with text
- Ground language expressions in visual scenes
- Create simple image captioning and visual question answering systems
- Evaluate vision-language model performance
- Integrate vision-language capabilities with Robotic Systems

## Prerequisites
- Completion of Module 1 (ROS 2 fundamentals)
- Completion of Module 2 (Simulation concepts)
- Completion of Module 3 (Isaac perception)
- Basic understanding of deep learning and PyTorch
- Familiarity with computer vision concepts

## Hardware and Software Requirements
### Required Hardware- GPU with CUDA support (NVIDIA RTX 3060 or better recommended)
- System with 16GB+ RAM
- Web camera or image dataset for testing

### Required Software- Python 3.8+ with PyTorch and torchvision
- Transformers library (Hugging Face)
- OpenCV and PIL for image processing
- ROS 2 Humble with vision packages
- Jupyter notebook or Python IDE

## Lab Setup
### Environment Configuration
1. **Install required packages:**
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers datasets
   pip install opencv-python pillow
   pip install jupyter notebook
   ```

2. **Verify GPU availability:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"Current GPU: {torch.cuda.get_device_name()}")
   ```

3. **Download sample dataset:**
   For this lab, we'll use a small sample dataset or synthetic data to demonstrate concepts.

## Implementation Steps
### Step 1: Basic Vision-Language Model
Create a simple model that can connect visual features with text embeddings:

```python
# vision_language_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BasicVisionLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()

        # Vision encoder (using pre-trained ResNet)
        self.vision_encoder = models.resnet18(pretrained=True)
        # Replace final layer for feature extraction
        num_features = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Linear(num_features, hidden_dim)

        # Text encoder (simple embedding + LSTM)
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )

        # Vision-language fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output heads for different tasks
        self.classification_head = nn.Linear(hidden_dim, 10)  # Example: 10 classes
        self.similarity_head = nn.Linear(hidden_dim, 1)      # For image-text matching

    def encode_image(self, image):
        """Encode image to feature vector"""
        features = self.vision_encoder(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text):
        """Encode text to feature vector"""
        embedded = self.text_embedding(text)
        encoded, (hidden, _) = self.text_encoder(embedded)
        # Use final hidden state
        text_features = hidden[-1]
        return F.normalize(text_features, dim=-1)

    def forward(self, image, text):
        """Forward pass combining image and text"""
        # Encode both modalities
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # Concatenate and fuse features
        combined_features = torch.cat([image_features, text_features], dim=-1)
        fused_features = self.fusion(combined_features)

        # Generate outputs
        classification_logits = self.classification_head(fused_features)
        similarity_score = self.similarity_head(fused_features)

        return classification_logits, similarity_score

# Example usage
def create_model():
    """Create and return a vision-language model"""
    model = BasicVisionLanguageModel(vocab_size=10000)  # Example vocab size
    return model
```

### Step 2: Image-Text Matching Implementation
Implement a simple image-text matching system:

```python
# image_text_matching.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_language_model import BasicVisionLanguageModel

class ImageTextMatcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vl_model = model

    def forward(self, images, texts):
        """Compute similarity between images and texts"""
        batch_size = images.size(0)

        # Encode all images and texts
        image_features = []
        text_features = []

        for i in range(batch_size):
            img_feat = self.vl_model.encode_image(images[i:i+1])
            txt_feat = self.vl_model.encode_text(texts[i:i+1])
            image_features.append(img_feat)
            text_features.append(txt_feat)

        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.t())

        return similarity_matrix

    def compute_loss(self, images, texts, labels):
        """Compute contrastive loss for image-text matching"""
        similarity_matrix = self(images, texts)

        # Labels should be diagonal (correct pairs)
        batch_size = images.size(0)
        target = torch.arange(batch_size).to(images.device)

        # Cross-entropy loss
        loss_i2t = F.cross_entropy(similarity_matrix, target)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), target)

        return (loss_i2t + loss_t2i) / 2
```

### Step 3: Simple Image Captioning
Create a basic image captioning system:

```python
# image_captioning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, max_length=20):
        super().__init__()

        # Vision encoder
        self.vision_encoder = models.resnet18(pretrained=True)
        num_features = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Linear(num_features, hidden_dim)

        # Text decoder
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_decoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.max_length = max_length

    def forward(self, image, target_captions=None):
        """Generate captions for images"""
        batch_size = image.size(0)

        # Encode image
        image_features = self.vision_encoder(image)  # [batch, hidden_dim]

        if target_captions is not None:
            # Training mode: use teacher forcing
            embedded = self.text_embedding(target_captions)
            decoder_output, _ = self.text_decoder(embedded)
            logits = self.output_projection(decoder_output)
            return logits
        else:
            # Inference mode: generate caption word by word
            return self.generate_caption(image_features)

    def generate_caption(self, image_features):
        """Generate caption using greedy decoding"""
        batch_size = image_features.size(0)

        # Start with start token (assuming token 1 is start token)
        current_tokens = torch.ones(batch_size, 1, dtype=torch.long) * 1

        all_tokens = [current_tokens]

        for _ in range(self.max_length):
            embedded = self.text_embedding(current_tokens)

            # For the first step, use image features as initial hidden state
            if len(all_tokens) == 1:
                h0 = image_features.unsqueeze(0).repeat(1, 1, 1)  # [1, batch, hidden]
                c0 = torch.zeros_like(h0)
                decoder_output, (h_n, c_n) = self.text_decoder(embedded, (h0, c0))
            else:
                decoder_output, (h_n, c_n) = self.text_decoder(embedded, (h_n, c_n))

            logits = self.output_projection(decoder_output)
            next_tokens = torch.argmax(logits, dim=-1)
            current_tokens = next_tokens[:, -1:]  # Take last prediction

            all_tokens.append(current_tokens)

            # Stop if end token is generated (assuming token 2 is end token)
            if torch.all(current_tokens == 2):
                break

        return torch.cat(all_tokens, dim=1)
```

### Step 4: Visual Question Answering
Implement a basic visual question answering system:

```python
# visual_qa.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualQAModel(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size, hidden_dim=512):
        super().__init__()

        # Vision encoder
        self.vision_encoder = models.resnet18(pretrained=True)
        num_features = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Linear(num_features, hidden_dim)

        # Question encoder (LSTM-based)
        self.question_embedding = nn.Embedding(vocab_size, hidden_dim // 2)
        self.question_encoder = nn.LSTM(
            hidden_dim // 2, hidden_dim // 2, batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Answer classifier
        self.answer_classifier = nn.Linear(hidden_dim, answer_vocab_size)

    def forward(self, image, question):
        """Answer questions about images"""
        # Encode image
        image_features = self.vision_encoder(image)

        # Encode question
        question_embedded = self.question_embedding(question)
        question_encoded, (hidden, _) = self.question_encoder(question_embedded)
        # Use last hidden state
        question_features = hidden[-1]

        # Fuse image and question features
        combined_features = torch.cat([image_features, question_features], dim=-1)
        fused_features = self.fusion(combined_features)

        # Generate answer
        answer_logits = self.answer_classifier(fused_features)

        return answer_logits

    def predict_answer(self, image, question, answer_vocab):
        """Predict the most likely answer"""
        logits = self(image, question)
        predicted_idx = torch.argmax(logits, dim=-1)

        # Convert to answer text
        answers = []
        for idx in predicted_idx:
            answer = answer_vocab.get(idx.item(), "unknown")
            answers.append(answer)

        return answers
```

### Step 5: Complete Vision-Language System
Create a complete system that integrates all components:

```python
# complete_vl_system.py

import torch
import torch.nn as nn
from vision_language_model import BasicVisionLanguageModel
from image_text_matching import ImageTextMatcher
from image_captioning import ImageCaptioner
from visual_qa import VisualQAModel

class CompleteVisionLanguageSystem(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size):
        super().__init__()

        # Initialize all components
        self.vision_language_model = BasicVisionLanguageModel(vocab_size)
        self.image_text_matcher = ImageTextMatcher(self.vision_language_model)
        self.image_captioner = ImageCaptioner(vocab_size)
        self.visual_qa = VisualQAModel(vocab_size, answer_vocab_size)

        self.task_specific_weights = nn.ParameterDict({
            'matching': nn.Parameter(torch.tensor(1.0)),
            'captioning': nn.Parameter(torch.tensor(1.0)),
            'qa': nn.Parameter(torch.tensor(1.0))
        })

    def forward(self, task, **kwargs):
        """Route to appropriate task"""
        if task == 'matching':
            return self.image_text_matcher(kwargs['images'], kwargs['texts'])
        elif task == 'captioning':
            return self.image_captioner(kwargs['image'], kwargs.get('target_captions', None))
        elif task == 'qa':
            return self.visual_qa(kwargs['image'], kwargs['question'])
        else:
            raise ValueError(f"Unknown task: {task}")

    def compute_total_loss(self, images, texts, questions, answers, captions):
        """Compute combined loss for all tasks"""
        # Image-text matching loss
        matching_loss = self.image_text_matcher.compute_loss(
            images, texts, labels=torch.arange(images.size(0)).to(images.device)
        )

        # Captioning loss (simplified)
        caption_logits = self.image_captioner(images, captions)
        caption_loss = F.cross_entropy(
            caption_logits.view(-1, caption_logits.size(-1)),
            captions.view(-1)
        )

        # QA loss
        qa_logits = self.visual_qa(images, questions)
        qa_loss = F.cross_entropy(qa_logits, answers)

        # Weighted combination
        total_loss = (
            self.task_specific_weights['matching'] * matching_loss +
            self.task_specific_weights['captioning'] * caption_loss +
            self.task_specific_weights['qa'] * qa_loss
        )

        return total_loss, {
            'matching_loss': matching_loss,
            'caption_loss': caption_loss,
            'qa_loss': qa_loss
        }
```

## Testing and Validation
### Basic Functionality Test
Create a test script to validate your implementations:

```python
# test_vision_language.py

import torch
from complete_vl_system import CompleteVisionLanguageSystem

def test_basic_functionality():
    """Test basic functionality of vision-language components"""

    # Parameters
    vocab_size = 10000
    answer_vocab_size = 500
    batch_size = 4
    image_channels = 3
    image_height = 224
    image_width = 224

    print("Testing Vision-Language System...")

    # Create model
    model = CompleteVisionLanguageSystem(vocab_size, answer_vocab_size)

    # Test image-text matching
    print("1. Testing Image-Text Matching...")
    images = torch.randn(batch_size, image_channels, image_height, image_width)
    texts = torch.randint(0, vocab_size, (batch_size, 10))  # 10 tokens per text

    matcher_output = model('matching', images=images, texts=texts)
    print(f"   Matching output shape: {matcher_output.shape}")

    # Test image captioning
    print("2. Testing Image Captioning...")
    caption_output = model('captioning', image=images[:1])  # Single image for generation
    print(f"   Caption output shape: {caption_output.shape}")

    # Test visual QA
    print("3. Testing Visual QA...")
    questions = torch.randint(0, vocab_size, (batch_size, 8))  # 8 tokens per question
    answers = torch.randint(0, answer_vocab_size, (batch_size,))

    qa_output = model('qa', image=images, question=questions)
    print(f"   QA output shape: {qa_output.shape}")

    # Test combined loss
    print("4. Testing Combined Loss...")
    total_loss, loss_components = model.compute_total_loss(
        images, texts, questions, answers, texts  # Using texts as dummy captions
    )
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss components: {loss_components}")

    print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
```

### Performance Evaluation
Create evaluation metrics for your vision-language system:

```python
# evaluate_vision_language.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class VisionLanguageEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_image_text_matching(self, model, test_loader):
        """Evaluate image-text matching performance"""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                images, texts, targets = batch
                outputs = model('matching', images=images, texts=texts)

                # For image-text matching, we want to find correct pairs
                # This is typically done by finding highest similarity scores
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_captioning(self, model, test_loader, vocab):
        """Evaluate image captioning performance"""
        model.eval()
        bleu_scores = []
        meteor_scores = []  # Simplified version

        with torch.no_grad():
            for batch in test_loader:
                images, target_captions = batch

                # Generate captions
                generated_captions = model('captioning', image=images)

                # Calculate BLEU scores (simplified)
                for gen_cap, target_cap in zip(generated_captions, target_captions):
                    bleu = self.calculate_bleu_score(gen_cap, target_cap)
                    bleu_scores.append(bleu)

        return {
            'avg_bleu': np.mean(bleu_scores),
            'std_bleu': np.std(bleu_scores)
        }

    def calculate_bleu_score(self, generated, reference, n=2):
        """Simplified BLEU score calculation"""
        # This is a very simplified version - in practice, use nltk.translate.bleu_score
        gen_tokens = generated.cpu().numpy()
        ref_tokens = reference.cpu().numpy()

        # Calculate n-gram overlap
        matches = 0
        total = 0

        for i in range(len(ref_tokens) - n + 1):
            if i + n <= len(gen_tokens):
                if tuple(ref_tokens[i:i+n]) in [tuple(gen_tokens[j:j+n])
                                              for j in range(len(gen_tokens) - n + 1)]:
                    matches += 1
            total += 1

        return matches / total if total > 0 else 0

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of the vision-language system"""
    evaluator = VisionLanguageEvaluator()

    print("Running comprehensive evaluation...")

    # This would typically load a test dataset
    # For this lab, we'll demonstrate the evaluation structure
    print("Evaluation framework ready.")
    print("To run full evaluation, implement data loading and call:")
    print("  evaluator.evaluate_image_text_matching(model, test_loader)")
    print("  evaluator.evaluate_captioning(model, test_loader, vocab)")

if __name__ == "__main__":
    run_comprehensive_evaluation()
```

## Integration with Robotics
### ROS 2 Integration Example
Show how to integrate vision-language capabilities with ROS 2:

```python
#!/usr/bin/env python3
# vision_language_ros_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
import numpy as np

class VisionLanguageROSNode(Node):
    def __init__(self):
        super().__init__('vision_language_ros_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load vision-language model (in practice, you'd load a trained model)
        # For this example, we'll use a dummy model
        self.model = self.load_model()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vl_command', self.command_callback, 10
        )

        # Publishers
        self.response_pub = self.create_publisher(String, '/vl_response', 10)
        self.gaze_pub = self.create_publisher(Point, '/robot_gaze_target', 10)

        # Internal state
        self.current_image = None
        self.current_command = None

        self.get_logger().info('Vision-Language ROS Node initialized')

    def load_model(self):
        """Load or initialize the vision-language model"""
        # In practice, load a pre-trained model
        # For this example, return a placeholder
        class DummyModel:
            def encode_image(self, image):
                # Return dummy features
                return torch.randn(1, 512)

        return DummyModel()

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Transform for model input
            tensor_image = self.image_transform(cv_image)
            tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension

            self.current_image = tensor_image

            # If we have both image and command, process them
            if self.current_command:
                self.process_vision_language_request()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming language command"""
        self.current_command = msg.data

        # If we have both image and command, process them
        if self.current_image is not None:
            self.process_vision_language_request()

    def process_vision_language_request(self):
        """Process combined vision and language input"""
        if self.current_image is None or self.current_command is None:
            return

        try:
            # In a real implementation, this would:
            # 1. Ground the language command in the visual scene
            # 2. Generate appropriate robot action or response
            # 3. Publish results

            # For this example, we'll simulate processing
            response = self.simulate_vision_language_processing(
                self.current_image, self.current_command
            )

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Reset for next request
            self.current_command = None

        except Exception as e:
            self.get_logger().error(f'Error in vision-language processing: {e}')

    def simulate_vision_language_processing(self, image, command):
        """Simulate vision-language processing"""
        # This would contain the actual vision-language model inference
        # For simulation, return a dummy response based on command keywords

        command_lower = command.lower()

        if 'red' in command_lower:
            return "I see a red object in the scene."
        elif 'person' in command_lower or 'human' in command_lower:
            return "I see a person in the scene."
        elif 'find' in command_lower or 'look' in command_lower:
            return "I'm analyzing the scene to find what you're looking for."
        else:
            return f"Processing command: {command}"

def main(args=None):
    rclpy.init(args=args)
    vl_node = VisionLanguageROSNode()

    try:
        rclpy.spin(vl_node)
    except KeyboardInterrupt:
        vl_node.get_logger().info('Shutting down Vision-Language ROS Node')
    finally:
        vl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab Deliverables
Complete the following tasks to finish the lab:

1. **Implement the basic vision-language model** with image and text encoding
2. **Create image-text matching functionality** with contrastive learning
3. **Implement simple image captioning** system
4. **Develop visual question answering** capability
5. **Integrate components into a complete system** with multiple tasks
6. **Test and validate** your implementations with sample data
7. **Document your results** including:
   - Model architecture and design decisions
   - Performance metrics achieved
   - Challenges encountered and solutions
   - Suggestions for improvement

## Assessment Criteria
Your lab implementation will be assessed based on:
- **Functionality**: Do all components work correctly?
- **Integration**: How well do the different components work together?
- **Performance**: Are the models efficient and accurate?
- **Code Quality**: Is the code well-structured and documented?
- **Problem Solving**: How effectively did you implement the vision-language concepts?

## Extensions (Optional)
For advanced students, consider implementing:
- **Attention mechanisms** for better vision-language alignment
- **Transformer-based models** for state-of-the-art performance
- **Multimodal pretraining** on larger datasets
- **Real-time processing** optimizations
- **Human evaluation** of generated captions or responses

## Troubleshooting
### Common Issues and Solutions
1. **Memory errors during training:**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Poor matching performance:**
   - Ensure proper normalization of embeddings
   - Try different loss functions (triplet loss, InfoNCE)
   - Increase training data diversity

3. **Caption generation issues:**
   - Check vocabulary coverage
   - Implement beam search for better generation
   - Add length normalization

4. **Integration problems:**
   - Verify data format compatibility
   - Check tensor dimensions throughout pipeline
   - Use consistent preprocessing across components

## Summary
This lab provided hands-on experience with fundamental vision-language integration techniques. You learned to connect visual perception with natural language understanding, creating systems that can ground language in visual contexts and respond intelligently to multimodal inputs. These capabilities are essential for advanced Human-Robot Interaction systems that can understand and respond to natural language commands in real-world environments.