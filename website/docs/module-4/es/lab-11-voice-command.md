---
sidebar_position: 6
---

# Lab 11: Voice Command Processing
## Overview
This lab focuses on implementing voice command processing systems for robotics applications. You will learn to capture, process, and understand spoken commands, integrating speech recognition with natural language understanding and robotic action execution. This is crucial for creating intuitive Human-Robot Interaction systems.

## Learning Objectives
After completing this lab, you will be able to:
- Implement real-time speech recognition systems
- Process and understand voice commands in context
- Integrate speech recognition with natural language processing
- Handle voice command ambiguity and errors
- Create robust voice command interfaces for robots

## Prerequisites
- Completion of Module 1 (ROS 2 fundamentals)
- Completion of Module 2 (Simulation concepts)
- Completion of Module 3 (Isaac perception)
- Basic understanding of signal processing
- Familiarity with Python audio processing libraries

## Hardware and Software Requirements
### Required Hardware- Microphone for audio input (USB or built-in)
- Speakers for audio feedback (optional)
- System with sufficient processing power for real-time audio processing

### Required Software- Python 3.8+ with required libraries
- Speech recognition libraries (speech_recognition, pyaudio)
- Audio processing libraries (librosa, scipy)
- Text-to-speech libraries (pyttsx3, espeak)
- ROS 2 Humble with audio processing packages

## Lab Setup
### Environment Configuration
1. **Install required packages:**
   ```bash
   pip install SpeechRecognition pyaudio
   pip install librosa scipy numpy
   pip install pyttsx3
   pip install torch torchaudio
   pip install transformers
   ```

2. **Test audio input:**
   ```python
   import pyaudio

   # Test audio input device
   p = pyaudio.PyAudio()
   print(f"Number of audio input devices: {p.get_device_count()}")

   for i in range(p.get_device_count()):
       info = p.get_device_info_by_index(i)
       if info['maxInputChannels'] > 0:
           print(f"Device {i}: {info['name']}")
   ```

3. **Verify speech recognition:**
   ```python
   import speech_recognition as sr

   # Test microphone access
   r = sr.Recognizer()
   with sr.Microphone() as source:
       print("Microphone test - say 'hello':")
       audio = r.listen(source, timeout=5)
       try:
           text = r.recognize_google(audio)
           print(f"Recognized: {text}")
       except sr.UnknownValueError:
           print("Could not understand audio")
       except sr.RequestError as e:
           print(f"Error: {e}")
   ```

## Implementation Steps
### Step 1: Basic Voice Command Recognition
Create a foundational voice command recognition system:

```python
# voice_recognition.py

import speech_recognition as sr
import pyaudio
import threading
import queue
import time
import logging
from typing import Callable, Optional

class VoiceCommandRecognizer:
    def __init__(self, callback: Optional[Callable] = None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Configuration
        self.recognizer.energy_threshold = 300  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True

        # Threading and queues for continuous recognition
        self.audio_queue = queue.Queue()
        self.listening = False
        self.continuous_thread = None

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def set_callback(self, callback: Callable):
        """Set callback function for recognized commands"""
        self.callback = callback

    def recognize_from_audio(self, audio_data):
        """Recognize speech from audio data"""
        try:
            # Use Google Web Speech API (requires internet)
            text = self.recognizer.recognize_google(audio_data)
            self.logger.info(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Error with speech recognition service: {e}")
            return None

    def listen_once(self):
        """Listen for a single voice command"""
        self.logger.info("Listening for command...")

        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                return self.recognize_from_audio(audio)
            except sr.WaitTimeoutError:
                self.logger.warning("No speech detected within timeout")
                return None

    def start_continuous_listening(self):
        """Start continuous voice command listening"""
        if self.listening:
            self.logger.warning("Already listening continuously")
            return

        self.listening = True
        self.continuous_thread = threading.Thread(target=self._continuous_listening_worker)
        self.continuous_thread.daemon = True
        self.continuous_thread.start()

    def stop_continuous_listening(self):
        """Stop continuous voice command listening"""
        self.listening = False
        if self.continuous_thread:
            self.continuous_thread.join(timeout=2)

    def _continuous_listening_worker(self):
        """Worker thread for continuous listening"""
        with self.microphone as source:
            while self.listening:
                try:
                    # Listen with timeout to allow for stopping
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                    # Recognize and process
                    text = self.recognize_from_audio(audio)
                    if text and self.callback:
                        self.callback(text)

                except sr.WaitTimeoutError:
                    # This is normal - just continue listening
                    continue
                except Exception as e:
                    self.logger.error(f"Error in continuous listening: {e}")
                    time.sleep(0.1)  # Brief pause before continuing

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        with self.microphone as source:
            self.logger.info("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.logger.info("Calibration complete")
```

### Step 2: Command Parser and Intent Recognition
Create a system to parse recognized speech and identify intents:

```python
# command_parser.py

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    intent: str
    entities: Dict[str, str]
    confidence: float
    original_text: str

class VoiceCommandParser:
    def __init__(self):
        # Define command patterns and intents
        self.command_patterns = {
            'navigation': [
                r'go to (the )?(?P<location>\w+)',
                r'move to (the )?(?P<location>\w+)',
                r'go to (the )?(?P<location>[\w\s]+?)\s*(?:room|area|spot|place)?',
                r'navigate to (the )?(?P<location>\w+)',
                r'bring me to (the )?(?P<location>\w+)'
            ],
            'object_interaction': [
                r'pick up (the )?(?P<object>[\w\s]+)',
                r'grab (the )?(?P<object>[\w\s]+)',
                r'take (the )?(?P<object>[\w\s]+)',
                r'get (the )?(?P<object>[\w\s]+)',
                r'bring me (the )?(?P<object>[\w\s]+)'
            ],
            'action': [
                r'follow (me|him|her)',
                r'stop',
                r'start',
                r'wait',
                r'help',
                r'what can you do'
            ],
            'question': [
                r'what is (this|that)',
                r'where is (the )?(?P<object>[\w\s]+)',
                r'how (many|much|long|big|tall|far) (is|are|does|do)',
                r'what time is it',
                r'what day is it'
            ]
        }

        # Location mappings
        self.location_synonyms = {
            'kitchen': ['kitchen', 'cooking area', 'cooking room'],
            'bedroom': ['bedroom', 'sleeping room', 'bed room'],
            'living room': ['living room', 'living area', 'sitting room', 'lounge'],
            'office': ['office', 'study', 'work room'],
            'bathroom': ['bathroom', 'restroom', 'toilet', 'bath'],
            'dining room': ['dining room', 'dining area', 'dining hall']
        }

        # Object mappings
        self.object_synonyms = {
            'water': ['water', 'bottle of water', 'water bottle'],
            'book': ['book', 'reading book', 'textbook'],
            'cup': ['cup', 'coffee cup', 'mug', 'glass'],
            'phone': ['phone', 'mobile', 'cell phone', 'smartphone'],
            'keys': ['keys', 'key', 'house keys', 'car keys']
        }

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse voice command and extract intent and entities"""
        text = text.lower().strip()

        # Check each intent type
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    entities = match.groupdict()

                    # Normalize entities using synonyms
                    normalized_entities = {}
                    for key, value in entities.items():
                        normalized_value = self._normalize_entity(key, value.strip())
                        normalized_entities[key] = normalized_value

                    # Calculate confidence based on pattern match strength
                    confidence = self._calculate_confidence(text, pattern, match)

                    return ParsedCommand(
                        intent=intent,
                        entities=normalized_entities,
                        confidence=confidence,
                        original_text=text
                    )

        # If no pattern matches, try to classify as general command
        return self._classify_general_command(text)

    def _normalize_entity(self, entity_type: str, value: str) -> str:
        """Normalize entity values using synonym mappings"""
        if entity_type == 'location':
            for canonical, synonyms in self.location_synonyms.items():
                if value in synonyms:
                    return canonical
        elif entity_type == 'object':
            for canonical, synonyms in self.object_synonyms.items():
                if value in synonyms:
                    return canonical
        return value

    def _calculate_confidence(self, text: str, pattern: str, match) -> float:
        """Calculate confidence score for the match"""
        # Simple confidence based on text coverage
        matched_text = match.group(0)
        confidence = len(matched_text) / len(text)

        # Boost confidence if it's a strong match
        if len(match.groups()) > 0:  # Has captured groups (entities)
            confidence *= 1.2

        return min(confidence, 1.0)

    def _classify_general_command(self, text: str) -> Optional[ParsedCommand]:
        """Classify commands that don't match specific patterns"""
        if any(word in text for word in ['hello', 'hi', 'hey', 'greetings']):
            return ParsedCommand(
                intent='greeting',
                entities={},
                confidence=0.8,
                original_text=text
            )
        elif any(word in text for word in ['thank', 'thanks', 'thank you']):
            return ParsedCommand(
                intent='acknowledgment',
                entities={},
                confidence=0.8,
                original_text=text
            )
        else:
            return ParsedCommand(
                intent='unknown',
                entities={},
                confidence=0.3,
                original_text=text
            )

    def get_suggested_commands(self) -> List[str]:
        """Get list of suggested commands for user guidance"""
        suggestions = [
            "Go to the kitchen",
            "Pick up the red cup",
            "What time is it?",
            "Where is my phone?",
            "Follow me",
            "Stop",
            "Help"
        ]
        return suggestions
```

### Step 3: Speech-to-Intent Processing Pipeline
Create a complete pipeline that processes speech to actionable robot commands:

```python
# speech_intent_pipeline.py

import asyncio
import threading
from typing import Dict, Any, Callable, Optional
from voice_recognition import VoiceCommandRecognizer
from command_parser import VoiceCommandParser, ParsedCommand

class SpeechIntentPipeline:
    def __init__(self, robot_controller_callback: Optional[Callable] = None):
        self.recognizer = VoiceCommandRecognizer()
        self.parser = VoiceCommandParser()
        self.robot_controller = robot_controller_callback

        # Callbacks
        self.command_callbacks = {
            'navigation': self._handle_navigation,
            'object_interaction': self._handle_object_interaction,
            'action': self._handle_action,
            'question': self._handle_question,
            'greeting': self._handle_greeting,
            'acknowledgment': self._handle_acknowledgment
        }

        # State management
        self.is_listening = False
        self.last_command = None
        self.command_history = []

    def start_listening(self):
        """Start the speech-to-intent pipeline"""
        if not self.is_listening:
            self.recognizer.set_callback(self._process_recognized_text)
            self.recognizer.start_continuous_listening()
            self.is_listening = True
            print("Voice command system started")

    def stop_listening(self):
        """Stop the speech-to-intent pipeline"""
        if self.is_listening:
            self.recognizer.stop_continuous_listening()
            self.is_listening = False
            print("Voice command system stopped")

    def _process_recognized_text(self, text: str):
        """Process recognized text through the pipeline"""
        print(f"Recognized: {text}")

        # Parse the command
        parsed_command = self.parser.parse_command(text)

        if parsed_command:
            print(f"Parsed - Intent: {parsed_command.intent}, "
                  f"Entities: {parsed_command.entities}, "
                  f"Confidence: {parsed_command.confidence:.2f}")

            # Store in history
            self.command_history.append(parsed_command)
            self.last_command = parsed_command

            # Handle the command based on intent
            self._handle_command(parsed_command)

    def _handle_command(self, parsed_command: ParsedCommand):
        """Handle parsed command based on intent"""
        if parsed_command.confidence < 0.5:
            print("Low confidence command - requesting clarification")
            self._request_clarification(parsed_command)
            return

        handler = self.command_callbacks.get(parsed_command.intent)
        if handler:
            try:
                handler(parsed_command)
            except Exception as e:
                print(f"Error handling command: {e}")
                self._handle_error(parsed_command, e)
        else:
            print(f"Unknown intent: {parsed_command.intent}")
            self._handle_unknown_command(parsed_command)

    def _handle_navigation(self, command: ParsedCommand):
        """Handle navigation commands"""
        location = command.entities.get('location', 'unknown')
        print(f"Navigating to {location}")

        if self.robot_controller:
            self.robot_controller('navigate', {'location': location})

    def _handle_object_interaction(self, command: ParsedCommand):
        """Handle object interaction commands"""
        obj = command.entities.get('object', 'unknown')
        print(f"Interacting with {obj}")

        if self.robot_controller:
            self.robot_controller('manipulate', {'object': obj, 'action': 'pick_up'})

    def _handle_action(self, command: ParsedCommand):
        """Handle action commands"""
        action = command.original_text
        print(f"Performing action: {action}")

        if self.robot_controller:
            self.robot_controller('action', {'command': action})

    def _handle_question(self, command: ParsedCommand):
        """Handle question commands"""
        question = command.original_text
        print(f"Processing question: {question}")

        if self.robot_controller:
            self.robot_controller('question', {'question': question})

    def _handle_greeting(self, command: ParsedCommand):
        """Handle greeting commands"""
        print("Hello! How can I help you?")

        if self.robot_controller:
            self.robot_controller('greeting', {'response': 'Hello! How can I help you?'})

    def _handle_acknowledgment(self, command: ParsedCommand):
        """Handle acknowledgment commands"""
        print("You're welcome!")

        if self.robot_controller:
            self.robot_controller('acknowledgment', {'response': "You're welcome!"})

    def _request_clarification(self, command: ParsedCommand):
        """Request clarification for low-confidence commands"""
        print(f"I'm not sure I understood. Did you mean: '{command.original_text}'?")

        if self.robot_controller:
            self.robot_controller('request_clarification', {
                'original_command': command.original_text,
                'suggested_commands': self.parser.get_suggested_commands()
            })

    def _handle_unknown_command(self, command: ParsedCommand):
        """Handle commands that couldn't be parsed"""
        print(f"I don't know how to handle: '{command.original_text}'")
        print("Try commands like:", ", ".join(self.parser.get_suggested_commands()))

        if self.robot_controller:
            self.robot_controller('unknown_command', {
                'command': command.original_text,
                'suggestions': self.parser.get_suggested_commands()
            })

    def _handle_error(self, command: ParsedCommand, error: Exception):
        """Handle errors during command processing"""
        print(f"Error processing command '{command.original_text}': {error}")

        if self.robot_controller:
            self.robot_controller('error', {
                'command': command.original_text,
                'error': str(error)
            })

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        self.recognizer.calibrate_microphone()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        return {
            'is_listening': self.is_listening,
            'command_history_length': len(self.command_history),
            'last_command': self.last_command.original_text if self.last_command else None,
            'suggested_commands': self.parser.get_suggested_commands()
        }
```

### Step 4: Advanced Voice Processing with Machine Learning
Implement more sophisticated voice command processing using ML models:

```python
# advanced_voice_processing.py

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, List, Tuple

class AdvancedVoiceProcessor:
    def __init__(self):
        # Initialize pre-trained speech recognition model
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.use_transformers = True
        except Exception as e:
            print(f"Could not load pre-trained model: {e}")
            print("Falling back to basic processing")
            self.use_transformers = False

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Preprocess audio data for model input"""
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_data = resampler(torch.from_numpy(audio_data)).numpy()

        return torch.tensor(audio_data).float()

    def recognize_speech_transformers(self, audio_data: np.ndarray) -> str:
        """Use pre-trained transformer model for speech recognition"""
        if not self.use_transformers:
            return None

        try:
            # Process audio
            input_values = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values

            # Recognize
            with torch.no_grad():
                logits = self.model(input_values).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription
        except Exception as e:
            print(f"Error in transformer-based recognition: {e}")
            return None

    def extract_audio_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, np.ndarray]:
        """Extract various audio features for command classification"""
        import librosa

        features = {}

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfccs'] = np.mean(mfccs, axis=1)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroids'] = np.mean(spectral_centroids)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)[0]
        features['zcr'] = np.mean(zcr)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma'] = np.mean(chroma, axis=1)

        return features

    def classify_command_type(self, audio_features: Dict[str, np.ndarray]) -> str:
        """Classify the type of command based on audio features"""
        # This would typically use a trained classifier
        # For this example, we'll use simple heuristics

        # Features that might indicate different command types
        avg_mfcc = np.mean(audio_features['mfccs'])
        avg_zcr = audio_features['zcr']
        avg_spectral = audio_features['spectral_centroids']

        # Simple rule-based classification
        if avg_zcr > 0.02:  # Higher zero crossing rate might indicate short commands
            return "short_command"
        elif avg_mfcc > -100:  # Average MFCC value
            return "long_command"
        else:
            return "unknown"

class VoiceCommandClassifier(nn.Module):
    """Neural network for classifying voice command types"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def create_voice_command_dataset():
    """Create a sample dataset for training voice command classifiers"""
    # This would typically load real audio data
    # For this example, we'll create synthetic data
    import numpy as np

    # Simulate audio features for different command types
    num_samples = 1000
    num_features = 20  # Number of audio features

    # Generate synthetic features for different command types
    navigation_features = np.random.normal(0.5, 0.2, (num_samples//4, num_features))
    object_features = np.random.normal(0.7, 0.2, (num_samples//4, num_features))
    action_features = np.random.normal(0.3, 0.2, (num_samples//4, num_features))
    question_features = np.random.normal(0.6, 0.2, (num_samples//4, num_features))

    X = np.vstack([navigation_features, object_features, action_features, question_features])
    y = np.hstack([
        np.zeros(num_samples//4),      # navigation
        np.ones(num_samples//4),       # object interaction
        np.full(num_samples//4, 2),    # action
        np.full(num_samples//4, 3)     # question
    ])

    return X, y
```

### Step 5: Voice Command System Integration
Create the complete voice command system with ROS 2 integration:

```python
#!/usr/bin/env python3
# voice_command_system.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import AudioData
from voice_recognition import VoiceCommandRecognizer
from command_parser import VoiceCommandParser, ParsedCommand
from speech_intent_pipeline import SpeechIntentPipeline
import threading
import time

class VoiceCommandSystem(Node):
    def __init__(self):
        super().__init__('voice_command_system')

        # Initialize voice processing components
        self.voice_pipeline = SpeechIntentPipeline(self.robot_command_callback)

        # Publishers
        self.command_pub = self.create_publisher(String, '/robot_commands', 10)
        self.navigation_pub = self.create_publisher(Pose, '/navigation_goal', 10)
        self.response_pub = self.create_publisher(String, '/voice_response', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio_input', self.audio_callback, 10
        )
        self.manual_command_sub = self.create_subscription(
            String, '/manual_voice_input', self.manual_command_callback, 10
        )

        # Timer for status updates
        self.status_timer = self.create_timer(5.0, self.publish_status)

        # State management
        self.system_active = True
        self.command_queue = []
        self.response_queue = []

        # Start voice recognition
        self.voice_pipeline.start_listening()

        self.get_logger().info('Voice Command System initialized')

    def robot_command_callback(self, intent: str, params: dict):
        """Callback for processed voice commands"""
        command_msg = String()

        if intent == 'navigate':
            # Publish navigation goal
            nav_pose = Pose()
            # In a real system, you'd convert location to coordinates
            self.navigation_pub.publish(nav_pose)
            command_msg.data = f"navigating_to_{params.get('location', 'unknown')}"

        elif intent == 'manipulate':
            command_msg.data = f"manipulating_{params.get('object', 'unknown')}"

        elif intent == 'action':
            command_msg.data = params.get('command', 'unknown_action')

        elif intent == 'question':
            command_msg.data = f"answering_{params.get('question', 'unknown_question')}"

        elif intent == 'greeting':
            self.speak_response(params.get('response', 'Hello!'))
            return

        elif intent == 'acknowledgment':
            self.speak_response(params.get('response', 'You are welcome!'))
            return

        elif intent == 'request_clarification':
            response = f"I didn't understand. Did you mean: {params.get('original_command', 'command')}?"
            self.speak_response(response)
            return

        elif intent == 'unknown_command':
            suggestions = ', '.join(params.get('suggestions', []))
            response = f"I don't know that command. Try: {suggestions}"
            self.speak_response(response)
            return

        self.command_pub.publish(command_msg)

    def audio_callback(self, msg):
        """Handle audio data from ROS topic"""
        # Convert audio message to format expected by speech recognizer
        # This is a simplified version - actual implementation would depend on audio format
        pass

    def manual_command_callback(self, msg):
        """Handle manually entered voice commands (for testing)"""
        # Process the text as if it were recognized speech
        self.get_logger().info(f"Manual command received: {msg.data}")

        # Parse and handle the command
        parsed = self.voice_pipeline.parser.parse_command(msg.data)
        if parsed:
            self.get_logger().info(f"Parsed command: {parsed.intent} with entities {parsed.entities}")
            self.voice_pipeline._handle_command(parsed)

    def speak_response(self, text: str):
        """Generate speech response"""
        # This would use text-to-speech in a real implementation
        self.get_logger().info(f"Speaking: {text}")

        response_msg = String()
        response_msg.data = text
        self.response_pub.publish(response_msg)

    def publish_status(self):
        """Publish system status periodically"""
        status = self.voice_pipeline.get_status()
        status_str = f"Listening: {status['is_listening']}, Commands: {status['command_history_length']}"

        status_msg = String()
        status_msg.data = f"STATUS: {status_str}"
        self.response_pub.publish(status_msg)

    def destroy_node(self):
        """Clean up before shutdown"""
        self.voice_pipeline.stop_listening()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_system = VoiceCommandSystem()

    try:
        rclpy.spin(voice_system)
    except KeyboardInterrupt:
        voice_system.get_logger().info('Shutting down Voice Command System')
    finally:
        voice_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing and Validation
### Basic Testing Script
Create a test script to validate your voice command system:

```python
# test_voice_command.py

import time
import threading
from speech_intent_pipeline import SpeechIntentPipeline

def mock_robot_controller(intent: str, params: dict):
    """Mock robot controller for testing"""
    print(f"Robot controller received: {intent} with params {params}")

def test_voice_pipeline():
    """Test the voice command pipeline"""
    print("Testing Voice Command Pipeline...")

    # Create pipeline with mock controller
    pipeline = SpeechIntentPipeline(mock_robot_controller)

    # Test different types of commands
    test_commands = [
        "go to the kitchen",
        "pick up the red cup",
        "what time is it",
        "follow me",
        "stop",
        "hello robot",
        "thank you"
    ]

    print("\nTesting command parsing:")
    for cmd in test_commands:
        print(f"\nTesting: '{cmd}'")
        parsed = pipeline.parser.parse_command(cmd)
        if parsed:
            print(f"  Intent: {parsed.intent}")
            print(f"  Entities: {parsed.entities}")
            print(f"  Confidence: {parsed.confidence:.2f}")
        else:
            print("  No parse result")

    # Test continuous listening (in a separate thread)
    print("\nStarting continuous listening test (5 seconds)...")
    pipeline.start_listening()

    # Wait for a few seconds
    time.sleep(5)

    # Stop listening
    pipeline.stop_listening()

    # Print status
    status = pipeline.get_status()
    print(f"\nFinal status: {status}")

    print("\nVoice pipeline test completed!")

def test_microphone_calibration():
    """Test microphone calibration"""
    from voice_recognition import VoiceCommandRecognizer

    print("Testing microphone calibration...")
    recognizer = VoiceCommandRecognizer()
    recognizer.calibrate_microphone()
    print("Calibration completed!")

if __name__ == "__main__":
    test_microphone_calibration()
    test_voice_pipeline()
```

### Performance Evaluation
Create evaluation metrics for voice command processing:

```python
# evaluate_voice_system.py

import time
import statistics
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class VoiceSystemMetrics:
    recognition_accuracy: float
    response_time_avg: float
    response_time_std: float
    command_success_rate: float
    false_positive_rate: float
    false_negative_rate: float

class VoiceSystemEvaluator:
    def __init__(self):
        self.metrics = []
        self.response_times = []
        self.correct_recognitions = 0
        self.total_commands = 0
        self.false_positives = 0
        self.false_negatives = 0

    def evaluate_recognition_performance(self, test_commands: List[Dict[str, str]]) -> VoiceSystemMetrics:
        """
        Evaluate recognition performance with known commands
        test_commands: List of {'audio_file': path, 'expected_text': text}
        """
        recognition_times = []
        correct_recognitions = 0

        for test_case in test_commands:
            start_time = time.time()

            # Simulate recognition (in real system, this would process audio file)
            recognized_text = self._simulate_recognition(test_case['audio_file'])
            recognition_time = time.time() - start_time

            recognition_times.append(recognition_time)

            if recognized_text.lower() == test_case['expected_text'].lower():
                correct_recognitions += 1

        accuracy = correct_recognitions / len(test_commands) if test_commands else 0
        avg_response_time = statistics.mean(recognition_times) if recognition_times else 0
        std_response_time = statistics.stdev(recognition_times) if len(recognition_times) > 1 else 0

        return VoiceSystemMetrics(
            recognition_accuracy=accuracy,
            response_time_avg=avg_response_time,
            response_time_std=std_response_time,
            command_success_rate=accuracy,  # Simplified
            false_positive_rate=0,  # Would need more complex evaluation
            false_negative_rate=0   # Would need more complex evaluation
        )

    def evaluate_command_understanding(self, test_commands: List[Dict[str, any]]) -> Dict[str, float]:
        """
        Evaluate command understanding performance
        test_commands: List of {'spoken_command': text, 'expected_intent': intent, 'expected_entities': entities}
        """
        correct_intents = 0
        correct_entities = 0
        total_commands = len(test_commands)

        for test_case in test_commands:
            from command_parser import VoiceCommandParser
            parser = VoiceCommandParser()

            parsed = parser.parse_command(test_case['spoken_command'])

            if parsed and parsed.intent == test_case['expected_intent']:
                correct_intents += 1

            # Check entity matching (simplified)
            expected_entities = test_case.get('expected_entities', {})
            if parsed and parsed.entities == expected_entities:
                correct_entities += 1

        return {
            'intent_accuracy': correct_intents / total_commands if total_commands > 0 else 0,
            'entity_accuracy': correct_entities / total_commands if total_commands > 0 else 0,
            'total_commands': total_commands
        }

    def _simulate_recognition(self, audio_file: str) -> str:
        """Simulate speech recognition (in real system, this would process the audio)"""
        # This is a placeholder - in real implementation, this would use actual recognition
        return "simulated recognition result"

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of the voice command system"""
    evaluator = VoiceSystemEvaluator()

    print("Running comprehensive voice system evaluation...")

    # Test recognition performance
    test_commands = [
        {'audio_file': 'command1.wav', 'expected_text': 'go to kitchen'},
        {'audio_file': 'command2.wav', 'expected_text': 'pick up cup'},
        {'audio_file': 'command3.wav', 'expected_text': 'what time is it'}
    ]

    recognition_metrics = evaluator.evaluate_recognition_performance(test_commands)
    print(f"Recognition Accuracy: {recognition_metrics.recognition_accuracy:.2f}")
    print(f"Average Response Time: {recognition_metrics.response_time_avg:.3f}s")

    # Test command understanding
    understanding_tests = [
        {
            'spoken_command': 'go to the kitchen',
            'expected_intent': 'navigation',
            'expected_entities': {'location': 'kitchen'}
        },
        {
            'spoken_command': 'pick up the red cup',
            'expected_intent': 'object_interaction',
            'expected_entities': {'object': 'red cup'}
        }
    ]

    understanding_metrics = evaluator.evaluate_command_understanding(understanding_tests)
    print(f"Intent Accuracy: {understanding_metrics['intent_accuracy']:.2f}")
    print(f"Entity Accuracy: {understanding_metrics['entity_accuracy']:.2f}")

    print("Evaluation completed!")

if __name__ == "__main__":
    run_comprehensive_evaluation()
```

## Lab Deliverables
Complete the following tasks to finish the lab:

1. **Implement the basic voice recognition system** with real-time processing
2. **Create command parsing and intent recognition** functionality
3. **Build the complete speech-to-intent pipeline** with error handling
4. **Implement advanced processing** with ML models (optional)
5. **Integrate with ROS 2** for robotic applications
6. **Test and validate** your implementations with sample commands
7. **Document your results** including:
   - Recognition accuracy achieved
   - Response times measured
   - Challenges encountered and solutions
   - Suggestions for improvement

## Assessment Criteria
Your lab implementation will be assessed based on:
- **Functionality**: Does the voice command system work correctly?
- **Accuracy**: How well does it recognize and understand commands?
- **Robustness**: How well does it handle noise and variations?
- **Integration**: How well is it integrated with Robotic Systems?
- **Code Quality**: Is the code well-structured and documented?

## Extensions (Optional)
For advanced students, consider implementing:
- **Wake word detection** to activate the system
- **Speaker identification** for personalized responses
- **Noise reduction** algorithms for better recognition
- **Multilingual support** for different languages
- **Emotion recognition** from voice patterns

## Troubleshooting
### Common Issues and Solutions
1. **Microphone access errors:**
   - Check microphone permissions in your OS
   - Verify microphone is not used by other applications
   - Try different audio input devices

2. **Poor recognition accuracy:**
   - Calibrate microphone for ambient noise
   - Speak clearly and at consistent distance
   - Check internet connection for cloud-based recognition

3. **High latency issues:**
   - Use local speech recognition models
   - Optimize audio processing pipeline
   - Reduce model complexity if needed

4. **Integration problems:**
   - Verify ROS 2 message formats
   - Check audio data encoding compatibility
   - Use appropriate sample rates and formats

## Summary
This lab provided hands-on experience with voice command processing for robotics applications. You learned to capture, process, and understand spoken commands, creating systems that can respond intelligently to natural voice interactions. These capabilities are essential for creating intuitive and accessible Human-Robot Interaction systems.