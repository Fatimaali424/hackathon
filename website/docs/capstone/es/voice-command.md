---
sidebar_position: 2
---

# Voice Command Processing
## Overview
The voice command processing system serves as the primary interface between users and the Autonomous Humanoid robot. This system enables natural language interaction, allowing users to communicate tasks and commands using everyday language rather than formal programming languages or specific command structures.

The voice command processing pipeline consists of several interconnected components: speech recognition, natural language understanding, command interpretation, and action mapping. Each component plays a crucial role in transforming spoken language into executable robot behaviors.

## System Architecture
### Voice Command Processing Pipeline
```
User Speaks Command
         │
         ▼
┌─────────────────┐
│   Audio Input   │
│   Acquisition   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│   & Filtering   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Speech-to-Text  │
│   Conversion    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Natural Lang   │
│   Processing    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Command Inter-  │
│ pretation &     │
│ Classification  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Intent-Action   │
│   Mapping       │
└─────────────────┘
         │
         ▼
│ Executable Robot Commands
```

## Core Components
### 1. Audio Input Acquisition
The audio input acquisition system handles the capture and initial processing of voice commands:

```python
import pyaudio
import numpy as np
import threading
import queue
import time

class AudioInputManager:
    def __init__(self,
                 sample_rate=16000,
                 chunk_size=1024,
                 channels=1,
                 sensitivity_threshold=500):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.sensitivity_threshold = sensitivity_threshold

        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()

        # Audio stream
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Energy detection parameters
        self.energy_threshold = 1000  # Initial threshold
        self.dynamic_threshold = True
        self.min_energy_threshold = 300

        # Recording parameters
        self.silence_duration = 1.0  # Seconds of silence to stop recording
        self.max_recording_duration = 10.0  # Max recording time

    def start_recording(self):
        """Start audio recording"""
        if self.stream is not None:
            return

        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _recording_worker(self):
        """Worker thread for audio recording"""
        silence_frames = 0
        recording_started = False
        recording_buffer = []
        start_time = time.time()

        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Calculate energy
                energy = np.sum(audio_chunk.astype(np.float32) ** 2) / len(audio_chunk)

                # Check for speech activity
                if energy > self.energy_threshold:
                    recording_started = True
                    recording_buffer.extend(audio_chunk.tolist())
                    silence_frames = 0
                elif recording_started:
                    # Accumulate silence
                    silence_frames += 1
                    recording_buffer.extend(audio_chunk.tolist())

                    # Check if enough silence to stop
                    silence_duration = (silence_frames * self.chunk_size) / self.sample_rate
                    if silence_duration > self.silence_duration:
                        # Stop recording and put audio in queue
                        audio_data = np.array(recording_buffer, dtype=np.int16)
                        self.audio_queue.put(audio_data)
                        break

                # Check maximum recording duration
                if time.time() - start_time > self.max_recording_duration:
                    audio_data = np.array(recording_buffer, dtype=np.int16)
                    self.audio_queue.put(audio_data)
                    break

            except Exception as e:
                print(f"Error in audio recording: {e}")
                break

    def get_audio_data(self):
        """Get recorded audio data"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
```

### 2. Speech Recognition Engine
The speech recognition engine converts audio to text using pre-trained models:

```python
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class SpeechRecognitionEngine:
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        # Load pre-trained model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Configuration
        self.sample_rate = 16000

    def preprocess_audio(self, audio_data):
        """Preprocess audio for speech recognition"""
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.float()

        # Resample if necessary
        if self.sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        return audio_tensor.unsqueeze(0)  # Add batch dimension

    def recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        # Preprocess audio
        input_tensor = self.preprocess_audio(audio_data)

        # Process with processor
        inputs = self.processor(
            input_tensor.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        # Recognize speech
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
```

### 3. Natural Language Understanding
The natural language understanding component interprets the recognized text and extracts meaning:

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CommandInterpretation:
    intent: str
    entities: Dict[str, str]
    confidence: float
    original_text: str
    normalized_text: str

class NaturalLanguageUnderstanding:
    def __init__(self):
        # Define command patterns and intents
        self.intent_patterns = {
            'navigation': [
                r'go to (the )?(?P<location>[\w\s]+?)(?: room| area| spot| place)?',
                r'move to (the )?(?P<location>[\w\s]+?)(?: room| area| spot| place)?',
                r'get to (the )?(?P<location>[\w\s]+?)(?: room| area| spot| place)?',
                r'go to (?:the )?(?P<location>[\w\s]+?)(?: please| now)?',
                r'navigate to (?:the )?(?P<location>[\w\s]+?)(?: please| now)?',
                r'bring me to (?:the )?(?P<location>[\w\s]+?)(?: please| now)?'
            ],
            'object_interaction': [
                r'(?:pick up|grab|get|take|fetch) (?:the )?(?P<object>[\w\s]+?)(?: from (?:the )?(?P<location>[\w\s]+?))?',
                r'(?:pick up|grab|get|take|fetch) (?:the )?(?P<object>[\w\s]+?)(?: there| over there)?',
                r'(?:pick up|grab|get|take|fetch) (?:the )?(?P<object>[\w\s]+?)(?: for me| please)?',
                r'bring me (?:the )?(?P<object>[\w\s]+?)(?: from (?:the )?(?P<location>[\w\s]+?))?',
                r'hand me (?:the )?(?P<object>[\w\s]+?)(?: from (?:the )?(?P<location>[\w\s]+?))?'
            ],
            'manipulation': [
                r'(?:pick up|lift|raise|hold) (?:the )?(?P<object>[\w\s]+?)',
                r'(?:put down|place|set) (?:the )?(?P<object>[\w\s]+?)(?: on (?:the )?(?P<surface>[\w\s]+?))?',
                r'(?:move|relocate) (?:the )?(?P<object>[\w\s]+?)(?: to (?:the )?(?P<location>[\w\s]+?))?',
                r'(?:open|close) (?:the )?(?P<object>[\w\s]+?)',
                r'(?:grasp|hold|squeeze) (?:the )?(?P<object>[\w\s]+?)'
            ],
            'action': [
                r'(?:stop|halt|pause|freeze)',
                r'(?:start|begin|go|proceed|continue)',
                r'(?:follow|escort|accompany) (?:me|him|her|them)',
                r'(?:wait|stand by|hold on|stay)',
                r'(?:help|assist|aid) (?:me|him|her|them)',
                r'(?:what can you do|what are you capable of|how can you help)'
            ],
            'question': [
                r'what is (?:this|that|the (?:\w+ ))(?P<object>[\w\s]+?)',
                r'where is (?:the )?(?P<object>[\w\s]+?)',
                r'where(?:\'s| is) (?:the )?(?P<object>[\w\s]+?)',
                r'how (?:many|much|long|far|big|small|tall|wide) (?:is|are|does|do) (?:the )?(?P<object>[\w\s]+?)',
                r'what time is it',
                r'what day is it',
                r'how are you',
                r'are you ready'
            ]
        }

        # Location synonyms mapping
        self.location_synonyms = {
            'kitchen': ['kitchen', 'cooking area', 'cooking room', 'food prep area'],
            'bedroom': ['bedroom', 'sleeping room', 'bed room', 'sleeping area'],
            'living room': ['living room', 'living area', 'sitting room', 'lounge', 'family room'],
            'office': ['office', 'study', 'work room', 'desk area'],
            'bathroom': ['bathroom', 'restroom', 'toilet', 'bath', 'washroom'],
            'dining room': ['dining room', 'dining area', 'dining hall', 'eat room'],
            'hallway': ['hallway', 'corridor', 'passage', 'hall', 'entry'],
            'garage': ['garage', 'car area', 'parking area'],
            'garden': ['garden', 'yard', 'outdoor area', 'patio', 'lawn']
        }

        # Object synonyms mapping
        self.object_synonyms = {
            'water': ['water', 'bottle of water', 'water bottle', 'drinking water'],
            'book': ['book', 'reading book', 'textbook', 'novel', 'magazine'],
            'cup': ['cup', 'coffee cup', 'mug', 'glass', 'drinking glass'],
            'phone': ['phone', 'mobile', 'cell phone', 'smartphone', 'cell'],
            'keys': ['keys', 'key', 'house keys', 'car keys', 'apartment keys'],
            'coffee': ['coffee', 'cup of coffee', 'hot drink', 'coffee mug'],
            'snack': ['snack', 'food', 'cookie', 'crackers', 'chips'],
            'medicine': ['medicine', 'pills', 'medication', 'drugs', 'prescription']
        }

    def interpret_command(self, text: str) -> Optional[CommandInterpretation]:
        """Interpret natural language command"""
        original_text = text
        text = text.lower().strip()

        # Normalize text
        normalized_text = self._normalize_text(text)

        # Try to match each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, normalized_text)
                if match:
                    entities = match.groupdict()

                    # Normalize entities using synonyms
                    normalized_entities = {}
                    for key, value in entities.items():
                        if value:
                            normalized_value = self._normalize_entity(key, value.strip())
                            normalized_entities[key] = normalized_value

                    # Calculate confidence based on match quality
                    confidence = self._calculate_confidence(text, pattern, match)

                    return CommandInterpretation(
                        intent=intent,
                        entities=normalized_entities,
                        confidence=confidence,
                        original_text=original_text,
                        normalized_text=normalized_text
                    )

        # If no pattern matches, try to classify as general command
        return self._classify_general_command(original_text)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())

        # Expand contractions (simple version)
        contractions = {
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "how's": "how is",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mightn't": "might not",
            "mustn't": "must not"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def _normalize_entity(self, entity_type: str, value: str) -> str:
        """Normalize entity value using synonym mapping"""
        if entity_type == 'location':
            for canonical, synonyms in self.location_synonyms.items():
                if value in synonyms:
                    return canonical
        elif entity_type == 'object' or entity_type == 'surface':
            for canonical, synonyms in self.object_synonyms.items():
                if value in synonyms:
                    return canonical

        return value

    def _calculate_confidence(self, text: str, pattern: str, match) -> float:
        """Calculate confidence score for the match"""
        # Calculate confidence based on text coverage
        matched_text = match.group(0)
        confidence = len(matched_text) / len(text) if len(text) > 0 else 0.0

        # Boost confidence if it has captured entities
        captured_groups = len([g for g in match.groups() if g is not None])
        if captured_groups > 0:
            confidence *= 1.2  # Boost for successful entity extraction

        return min(confidence, 1.0)  # Clamp to [0, 1]

    def _classify_general_command(self, text: str) -> CommandInterpretation:
        """Classify commands that don't match specific patterns"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening']):
            return CommandInterpretation(
                intent='greeting',
                entities={},
                confidence=0.8,
                original_text=text,
                normalized_text=text_lower
            )
        elif any(word in text_lower for word in ['thank', 'thanks', 'thank you', 'appreciate']):
            return CommandInterpretation(
                intent='acknowledgment',
                entities={},
                confidence=0.8,
                original_text=text,
                normalized_text=text_lower
            )
        elif any(word in text_lower for word in ['sorry', 'excuse me', 'pardon']):
            return CommandInterpretation(
                intent='apology',
                entities={},
                confidence=0.7,
                original_text=text,
                normalized_text=text_lower
            )
        else:
            return CommandInterpretation(
                intent='unknown',
                entities={},
                confidence=0.3,
                original_text=text,
                normalized_text=text_lower
            )
```

### 4. Command Interpretation and Classification
The command interpretation system classifies commands and extracts relevant entities:

```python
class CommandInterpreter:
    def __init__(self):
        self.nlu = NaturalLanguageUnderstanding()
        self.context_manager = ContextManager()

    def interpret(self, text: str, context: Dict = None) -> Dict:
        """Interpret command with context awareness"""
        # Get basic interpretation
        interpretation = self.nlu.interpret_command(text)

        # Enhance with context
        enhanced_interpretation = self._enhance_with_context(interpretation, context)

        return {
            'interpretation': enhanced_interpretation,
            'valid': self._validate_interpretation(enhanced_interpretation),
            'suggested_corrections': self._suggest_corrections(enhanced_interpretation, text)
        }

    def _enhance_with_context(self, interpretation: CommandInterpretation, context: Dict) -> CommandInterpretation:
        """Enhance interpretation with contextual information"""
        if not context:
            return interpretation

        enhanced_entities = interpretation.entities.copy()

        # Use context to disambiguate entities
        if 'current_location' in context and interpretation.intent == 'navigation':
            if 'location' in enhanced_entities:
                # Could use context to validate or correct location
                pass

        # Use context to infer missing information
        if interpretation.intent == 'object_interaction' and 'location' not in enhanced_entities:
            if 'current_location' in context:
                enhanced_entities['location'] = context['current_location']

        return CommandInterpretation(
            intent=interpretation.intent,
            entities=enhanced_entities,
            confidence=interpretation.confidence,
            original_text=interpretation.original_text,
            normalized_text=interpretation.normalized_text
        )

    def _validate_interpretation(self, interpretation: CommandInterpretation) -> bool:
        """Validate if interpretation makes sense"""
        if interpretation.confidence < 0.5:
            return False

        if interpretation.intent == 'navigation':
            # Validate location exists
            return 'location' in interpretation.entities and interpretation.entities['location']

        if interpretation.intent == 'object_interaction':
            # Validate object exists
            return 'object' in interpretation.entities and interpretation.entities['object']

        return True

    def _suggest_corrections(self, interpretation: CommandInterpretation, original_text: str) -> List[str]:
        """Suggest possible corrections or clarifications"""
        suggestions = []

        if interpretation.intent == 'unknown' or interpretation.confidence < 0.6:
            suggestions.append(f"I didn't understand '{original_text}'. Could you rephrase?")

        elif interpretation.intent == 'navigation' and 'location' not in interpretation.entities:
            suggestions.append("I heard a navigation command but didn't catch the destination. Where would you like me to go?")

        elif interpretation.intent == 'object_interaction' and 'object' not in interpretation.entities:
            suggestions.append("I heard an object interaction command but didn't catch what object. What would you like me to interact with?")

        return suggestions

class ContextManager:
    """Manages conversation and environmental context"""
    def __init__(self):
        self.conversation_history = []
        self.environment_context = {}
        self.user_preferences = {}
        self.robot_state = {}

    def update_context(self, user_input: str, robot_response: str, environment_state: Dict):
        """Update context with new information"""
        self.conversation_history.append({
            'user': user_input,
            'robot': robot_response,
            'timestamp': time.time()
        })

        self.environment_context.update(environment_state)

    def get_context(self) -> Dict:
        """Get current context for command interpretation"""
        return {
            'conversation_history': self.conversation_history[-5:],  # Last 5 exchanges
            'environment': self.environment_context,
            'user_preferences': self.user_preferences,
            'robot_state': self.robot_state
        }
```

## Voice Command Processing Pipeline
### Complete Voice Command System
```python
class VoiceCommandSystem:
    def __init__(self):
        # Initialize components
        self.audio_manager = AudioInputManager()
        self.speech_recognizer = SpeechRecognitionEngine()
        self.command_interpreter = CommandInterpreter()
        self.context_manager = ContextManager()

        # State management
        self.is_active = False
        self.listening = False
        self.wake_word = "robot"
        self.activation_threshold = 0.7

        # Callbacks
        self.command_callbacks = {}

    def start_listening(self):
        """Start listening for voice commands"""
        if not self.is_active:
            self.audio_manager.start_recording()
            self.is_active = True
            self.listening = True
            print("Voice command system activated")

    def stop_listening(self):
        """Stop listening for voice commands"""
        if self.is_active:
            self.audio_manager.stop_recording()
            self.is_active = False
            self.listening = False
            print("Voice command system deactivated")

    def process_audio_command(self, audio_data: np.ndarray) -> Dict:
        """Process audio command through complete pipeline"""
        # Step 1: Recognize speech
        recognized_text = self.speech_recognizer.recognize_speech(audio_data)

        # Step 2: Interpret command
        context = self.context_manager.get_context()
        interpretation_result = self.command_interpreter.interpret(recognized_text, context)

        # Step 3: Validate and handle
        interpretation = interpretation_result['interpretation']

        if interpretation_result['valid']:
            # Execute appropriate action based on intent
            action_result = self._execute_action(interpretation)

            # Update context
            self.context_manager.update_context(
                recognized_text,
                action_result.get('response', 'Action executed'),
                {}
            )

            return {
                'success': True,
                'interpretation': interpretation,
                'action_result': action_result,
                'corrections': []
            }
        else:
            # Request clarification
            return {
                'success': False,
                'interpretation': interpretation,
                'action_result': None,
                'corrections': interpretation_result['suggested_corrections']
            }

    def _execute_action(self, interpretation: CommandInterpretation) -> Dict:
        """Execute action based on command interpretation"""
        intent = interpretation.intent
        entities = interpretation.entities

        # Route to appropriate handler
        handlers = {
            'navigation': self._handle_navigation,
            'object_interaction': self._handle_object_interaction,
            'manipulation': self._handle_manipulation,
            'action': self._handle_action,
            'question': self._handle_question,
            'greeting': self._handle_greeting,
            'acknowledgment': self._handle_acknowledgment
        }

        handler = handlers.get(intent)
        if handler:
            return handler(entities, interpretation.original_text)
        else:
            return {
                'status': 'unknown_intent',
                'message': f'Unknown intent: {intent}',
                'command': interpretation.original_text
            }

    def _handle_navigation(self, entities: Dict, original_command: str) -> Dict:
        """Handle navigation commands"""
        location = entities.get('location', 'unknown')
        return {
            'action': 'navigate',
            'target': location,
            'original_command': original_command,
            'response': f'Navigating to {location}'
        }

    def _handle_object_interaction(self, entities: Dict, original_command: str) -> Dict:
        """Handle object interaction commands"""
        obj = entities.get('object', 'unknown')
        location = entities.get('location', 'current location')
        return {
            'action': 'interact_with_object',
            'object': obj,
            'location': location,
            'original_command': original_command,
            'response': f'Interacting with {obj} in {location}'
        }

    def _handle_manipulation(self, entities: Dict, original_command: str) -> Dict:
        """Handle manipulation commands"""
        obj = entities.get('object', 'unknown')
        action_type = original_command.split()[0] if original_command.split() else 'manipulate'
        return {
            'action': 'manipulate_object',
            'object': obj,
            'manipulation_type': action_type,
            'original_command': original_command,
            'response': f'{action_type.title()}ing {obj}'
        }

    def _handle_action(self, entities: Dict, original_command: str) -> Dict:
        """Handle action commands"""
        return {
            'action': 'execute_action',
            'command': original_command,
            'original_command': original_command,
            'response': f'Executing: {original_command}'
        }

    def _handle_question(self, entities: Dict, original_command: str) -> Dict:
        """Handle question commands"""
        return {
            'action': 'answer_question',
            'question': original_command,
            'entities': entities,
            'original_command': original_command,
            'response': f'I will answer your question: {original_command}'
        }

    def _handle_greeting(self, entities: Dict, original_command: str) -> Dict:
        """Handle greeting commands"""
        return {
            'action': 'greet',
            'original_command': original_command,
            'response': 'Hello! How can I assist you today?'
        }

    def _handle_acknowledgment(self, entities: Dict, original_command: str) -> Dict:
        """Handle acknowledgment commands"""
        return {
            'action': 'acknowledge',
            'original_command': original_command,
            'response': 'You\'re welcome!'
        }
```

## Integration with Robot Systems
### ROS 2 Integration
```python
#!/usr/bin/env python3
# voice_command_ros_integration.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import AudioData
from capstone_interfaces.msg import VoiceCommand, RobotAction

class VoiceCommandROSNode(Node):
    def __init__(self):
        super().__init__('voice_command_system')

        # Initialize voice command system
        self.voice_system = VoiceCommandSystem()

        # Publishers
        self.command_pub = self.create_publisher(RobotAction, '/robot_commands', 10)
        self.response_pub = self.create_publisher(String, '/voice_response', 10)
        self.navigation_pub = self.create_publisher(Pose, '/navigation_goal', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/microphone/audio_raw', self.audio_callback, 10
        )
        self.voice_command_sub = self.create_subscription(
            String, '/user_voice_input', self.voice_text_callback, 10
        )

        # Timer for periodic processing
        self.process_timer = self.create_timer(0.1, self.process_commands)

        # Start voice system
        self.voice_system.start_listening()

        self.get_logger().info('Voice Command System Node Initialized')

    def audio_callback(self, msg):
        """Handle audio data from microphone"""
        # Convert audio message to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16)

        # Process through voice command system
        result = self.voice_system.process_audio_command(audio_data)

        # Handle result
        self._handle_command_result(result)

    def voice_text_callback(self, msg):
        """Handle text-based voice commands (for testing)"""
        # For testing purposes, treat text as recognized speech
        # In real system, this would come from speech recognition

        # Create a mock interpretation for testing
        from dataclasses import dataclass
        @dataclass
        class MockInterpretation:
            intent: str
            entities: dict
            confidence: float
            original_text: str
            normalized_text: str

        interpretation = MockInterpretation(
            intent='navigation',
            entities={'location': 'kitchen'},
            confidence=0.9,
            original_text=msg.data,
            normalized_text=msg.data.lower()
        )

        result = {
            'success': True,
            'interpretation': interpretation,
            'action_result': {
                'action': 'navigate',
                'target': 'kitchen',
                'response': f'Navigating to kitchen'
            },
            'corrections': []
        }

        self._handle_command_result(result)

    def process_commands(self):
        """Periodic command processing"""
        # In a real system, this would process queued commands
        pass

    def _handle_command_result(self, result):
        """Handle command processing result"""
        if result['success']:
            action_result = result['action_result']

            # Publish robot command
            robot_cmd = RobotAction()
            robot_cmd.action_type = action_result['action']
            robot_cmd.target_location = action_result.get('target', '')
            robot_cmd.object_name = action_result.get('object', '')

            self.command_pub.publish(robot_cmd)

            # Publish response
            response_msg = String()
            response_msg.data = action_result['response']
            self.response_pub.publish(response_msg)

            self.get_logger().info(f"Command executed: {action_result['response']}")
        else:
            # Handle invalid interpretations
            for correction in result['corrections']:
                response_msg = String()
                response_msg.data = correction
                self.response_pub.publish(response_msg)

                self.get_logger().info(f"Suggestion: {correction}")

    def destroy_node(self):
        """Cleanup before shutdown"""
        self.voice_system.stop_listening()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceCommandROSNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        voice_node.get_logger().info('Shutting down Voice Command System')
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance and Optimization
### Real-time Processing Considerations
```python
class OptimizedVoiceCommandSystem(VoiceCommandSystem):
    """Optimized version for real-time performance"""

    def __init__(self):
        super().__init__()

        # Caching for frequently accessed data
        self.command_cache = {}
        self.max_cache_size = 100

        # Threading for non-blocking operations
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Performance monitoring
        self.processing_times = []
        self.max_processing_samples = 100

    def process_audio_command(self, audio_data: np.ndarray) -> Dict:
        """Optimized command processing with performance monitoring"""
        start_time = time.time()

        try:
            # Use cached results when possible
            audio_hash = hash(tuple(audio_data[:100]))  # Hash first 100 samples
            if audio_hash in self.command_cache:
                return self.command_cache[audio_hash]

            # Process normally
            result = super().process_audio_command(audio_data)

            # Cache result
            if len(self.command_cache) < self.max_cache_size:
                self.command_cache[audio_hash] = result

            return result

        finally:
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only recent samples
            if len(self.processing_times) > self.max_processing_samples:
                self.processing_times = self.processing_times[-self.max_processing_samples:]

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'avg_processing_time_ms': 0,
                'min_processing_time_ms': 0,
                'max_processing_time_ms': 0,
                'processing_samples': 0
            }

        avg_time = sum(self.processing_times) / len(self.processing_times)
        min_time = min(self.processing_times)
        max_time = max(self.processing_times)

        return {
            'avg_processing_time_ms': avg_time * 1000,
            'min_processing_time_ms': min_time * 1000,
            'max_processing_time_ms': max_time * 1000,
            'processing_samples': len(self.processing_times),
            'cache_hits': len([k for k, v in self.command_cache.items() if v is not None])
        }
```

## Error Handling and Recovery
### Robust Command Processing
```python
class RobustVoiceCommandSystem(OptimizedVoiceCommandSystem):
    """Robust version with comprehensive error handling"""

    def __init__(self):
        super().__init__()
        self.error_handlers = {
            'speech_recognition_error': self._handle_speech_recognition_error,
            'command_interpretation_error': self._handle_command_interpretation_error,
            'action_execution_error': self._handle_action_execution_error
        }

    def process_audio_command(self, audio_data: np.ndarray) -> Dict:
        """Process command with comprehensive error handling"""
        try:
            return super().process_audio_command(audio_data)
        except Exception as e:
            error_type = self._classify_error(e)
            return self.error_handlers[error_type](e, audio_data)

    def _classify_error(self, error: Exception) -> str:
        """Classify error type"""
        error_msg = str(error).lower()

        if 'speech' in error_msg or 'recognit' in error_msg:
            return 'speech_recognition_error'
        elif 'command' in error_msg or 'interpret' in error_msg:
            return 'command_interpretation_error'
        elif 'action' in error_msg or 'execut' in error_msg:
            return 'action_execution_error'
        else:
            return 'unknown_error'

    def _handle_speech_recognition_error(self, error: Exception, audio_data: np.ndarray) -> Dict:
        """Handle speech recognition errors"""
        return {
            'success': False,
            'interpretation': None,
            'action_result': None,
            'corrections': ['I couldn\'t understand your voice command. Could you speak more clearly?'],
            'error': f'Speech recognition error: {str(error)}'
        }

    def _handle_command_interpretation_error(self, error: Exception, audio_data: np.ndarray) -> Dict:
        """Handle command interpretation errors"""
        return {
            'success': False,
            'interpretation': None,
            'action_result': None,
            'corrections': ['I didn\'t understand that command. Could you rephrase it?'],
            'error': f'Command interpretation error: {str(error)}'
        }

    def _handle_action_execution_error(self, error: Exception, audio_data: np.ndarray) -> Dict:
        """Handle action execution errors"""
        return {
            'success': False,
            'interpretation': None,
            'action_result': None,
            'corrections': ['I encountered an error executing that command. Would you like me to try again?'],
            'error': f'Action execution error: {str(error)}'
        }
```

This voice command processing system provides a comprehensive foundation for natural language interaction with the Autonomous Humanoid robot. The system handles audio input, speech recognition, natural language understanding, and action mapping in a robust and efficient manner, suitable for real-time robotic applications.