# Feature Specification: Physical AI & Humanoid Robotics

**Feature Branch**: `002-physical-ai-humanoid-book`
**Created**: 2025-12-17
**Status**: Draft
**Input**: User description: "You are generating the full Specification document for an academic technical book titled:

\"Physical AI & Humanoid Robotics\"  📘 PROJECT CONTEXT

The book is an academic, technical publication authored as a Docusaurus documentation site and deployed to GitHub Pages.

Workflow and tooling:

Spec-Kit Plus for structured, spec-driven book development

Claude Code for iterative drafting, refinement, and verification

Version control and deployment via GitHub

The book focuses on Physical AI and Humanoid Robotics, emphasizing embodied intelligence operating in the real world.

📘 GLOBAL CONSTRAINTS (MANDATORY)

All specifications must enforce the following:

Total Word Count: 5,000–7,000 words (excluding references)

Citation Style: APA

Sources: Minimum 15 sources, at least 50% peer-reviewed

Plagiarism: 0% tolerance; all content must be original or properly cited

Writing Clarity: Flesch-Kincaid Grade 10–12

Deliverable Format: PDF export from Docusaurus with embedded citations

Verification: Every factual claim must be traceable to a credible source

Success Criteria

Zero plagiarism detected

Passes fact-checking review

All factual claims cited

Reproducible workflows and code examples

📘 SPECIFICATION OBJECTIVE

Convert the provided curriculum, modules, hardware details, and learning structure into a formal, enforceable Spec-Kit Plus specification.

The specification must:

Define scope, boundaries, and requirements

Be precise, testable, and unambiguous

Serve as the authoritative input for /sp.plan

📘 REQUIRED SPECIFICATION SECTIONS (NON-NEGOTIABLE)

Generate a single, complete specification document containing all of the following sections:

Title

Summary

Purpose of the Book

Scope

Target Audience (Computer Science / AI background)

Learning Themes

Physical AI & embodied intelligence

Humanoid robotics

ROS 2

Digital Twin simulation (Gazebo, Unity)

NVIDIA Isaac

Vision-Language-Action (VLA)

Module Specifications

Module 1: The Robotic Nervous System (ROS 2)

Module 2: The Digital Twin (Gazebo & Unity)

Module 3: The AI-Robot Brain (NVIDIA Isaac)

Module 4: Vision-Language-Action (VLA)

Each module specification must include:

Description

Key concepts

Skills gained

Weekly alignment

Deliverables / labs

Capstone Specification

Capstone title: The Autonomous Humanoid

Voice command → planning → navigation → perception → manipulation

Requirements

Success criteria

Evaluation rubric

Weekly Roadmap (Weeks 1–13)

Weekly topic

Learning objectives

Required tools/software

Lab or assignment

Learning Outcomes

Knowledge outcomes

Skill outcomes

Behavioral / competency outcomes

Hardware Specifications

Digital Twin workstation

Jetson Edge AI kit

Sensor suite

Robot lab options (budget → premium)

Sim-to-real architecture

Cloud-based alternative (\"Ether Lab\")

Include:

Structured tables

Minimum vs recommended specs

Rationale for each component

Lab Architecture Diagram (Textual Description)

Simulation rig

Jetson edge device

Sensors

Actuators

Cloud alternative

Data flow between components

Risks & Constraints

Latency trap (cloud → real robot)

GPU VRAM requirements

OS constraints (Ubuntu)"

## Summary

This specification defines the academic technical book "Physical AI & Humanoid Robotics" - a comprehensive educational resource that bridges theoretical concepts with practical implementation in embodied intelligence. The book provides a structured curriculum covering ROS 2, Digital Twin simulation, NVIDIA Isaac, and Vision-Language-Action systems, designed for computer science and AI practitioners.

## Purpose of the Book

The purpose of this book is to provide a comprehensive, academically rigorous resource for understanding and implementing Physical AI and Humanoid Robotics systems. It emphasizes the integration of perception, planning, control, and action in embodied agents that operate in the real world, combining theoretical foundations with practical implementation using industry-standard tools and platforms.

## Scope

### In Scope
- Physical AI and embodied intelligence concepts
- Humanoid robotics design and implementation
- ROS 2 as the robotic nervous system
- Digital Twin simulation using Gazebo and Unity
- NVIDIA Isaac for AI-powered robotics
- Vision-Language-Action (VLA) systems
- 13-week curriculum with weekly topics and assignments
- Hardware specifications for Digital Twin workstation, Jetson Edge AI kit, and sensor suite
- Capstone project: The Autonomous Humanoid with voice command → planning → navigation → perception → manipulation
- Learning outcomes for knowledge, skills, and behavioral competencies
- Lab architecture including simulation, edge devices, and cloud alternatives

### Out of Scope
- Basic programming fundamentals (assumes CS/AI background)
- Hardware manufacturing processes
- Detailed electronics circuit design
- Non-robotic AI applications (e.g., pure computer vision without embodiment)

## Target Audience

The target audience consists of:
- Graduate students in computer science, robotics, or AI
- Researchers in embodied intelligence and robotics
- Engineers developing robotic systems
- Educators teaching robotics and AI courses
- Practitioners with computer science or AI background seeking to understand Physical AI and humanoid robotics

## Learning Themes

### Physical AI & Embodied Intelligence
Understanding how AI systems can be embodied in physical agents that interact with the real world, including perception-action loops, sensorimotor integration, and real-world learning.

### Humanoid Robotics
Exploring the design, control, and implementation of humanoid robotic systems, including kinematics, dynamics, and human-like interaction capabilities.

### ROS 2
Mastering the Robot Operating System as the foundational nervous system for robotic applications, including communication, coordination, and middleware concepts.

### Digital Twin Simulation (Gazebo, Unity)
Creating and utilizing digital replicas of physical robotic systems for testing, validation, and development in virtual environments before real-world deployment.

### NVIDIA Isaac
Leveraging NVIDIA's AI platform for robotics, including perception, planning, and control algorithms optimized for edge computing.

### Vision-Language-Action (VLA)
Integrating visual perception, natural language understanding, and physical action to create intelligent robotic systems that can understand and respond to human commands.

## Module Specifications

### Module 1: The Robotic Nervous System (ROS 2)

**Description**: This module covers the Robot Operating System (ROS 2) as the foundational communication and coordination framework for robotic applications. Students learn about nodes, topics, services, actions, and the distributed computing model that enables complex robotic systems.

**Key Concepts**:
- ROS 2 architecture and middleware
- Node communication patterns (publish/subscribe, services, actions)
- Message and service definitions
- Parameter management and configuration
- Launch files and system composition
- Real-time considerations and Quality of Service (QoS)

**Skills Gained**:
- Design and implement ROS 2 nodes
- Create custom message and service types
- Configure and launch complex robotic systems
- Debug and monitor ROS 2 communications
- Implement distributed robotic applications

**Weekly Alignment**: Weeks 1-3 of the 13-week roadmap

**Deliverables/Labs**:
- Lab 1: Basic ROS 2 publisher/subscriber implementation
- Lab 2: Service and action client/server development
- Lab 3: Multi-node system integration and launch configuration
- Assignment: Design a simple robotic system using ROS 2 patterns

### Module 2: The Digital Twin (Gazebo & Unity)

**Description**: This module focuses on creating and utilizing digital replicas of physical robotic systems for simulation, testing, and validation. Students learn to build accurate virtual environments and test robotic behaviors before real-world deployment.

**Key Concepts**:
- Digital twin architecture and synchronization
- Physics simulation and modeling
- Sensor simulation and realistic environment creation
- Gazebo simulation framework and plugins
- Unity integration for advanced visualization
- Sim-to-real transfer challenges and solutions

**Skills Gained**:
- Create realistic simulation environments
- Model robotic systems with accurate physics
- Implement sensor simulation and validation
- Design sim-to-real transfer strategies
- Validate robotic behaviors in virtual environments

**Weekly Alignment**: Weeks 4-6 of the 13-week roadmap

**Deliverables/Labs**:
- Lab 1: Basic robot model creation and simulation in Gazebo
- Lab 2: Advanced sensor simulation and environment design
- Lab 3: Unity integration for enhanced visualization
- Assignment: Design a digital twin for a simple mobile robot

### Module 3: The AI-Robot Brain (NVIDIA Isaac)

**Description**: This module explores NVIDIA Isaac as the AI-powered brain for robotic systems, focusing on perception, planning, and control algorithms optimized for edge computing platforms like Jetson.

**Key Concepts**:
- NVIDIA Isaac platform architecture
- AI-powered perception (object detection, segmentation, pose estimation)
- Motion planning and trajectory generation
- Control algorithms for robotic systems
- Edge AI deployment and optimization
- Sensor fusion and state estimation

**Skills Gained**:
- Implement AI perception pipelines using Isaac
- Design motion planning algorithms
- Deploy AI models on edge computing platforms
- Integrate perception and control systems
- Optimize AI algorithms for real-time performance

**Weekly Alignment**: Weeks 7-9 of the 13-week roadmap

**Deliverables/Labs**:
- Lab 1: Basic perception pipeline with Isaac
- Lab 2: Motion planning and control implementation
- Lab 3: Edge deployment and optimization
- Assignment: Integrate perception and control for a manipulation task

### Module 4: Vision-Language-Action (VLA)

**Description**: This module integrates visual perception, natural language understanding, and physical action to create intelligent robotic systems capable of understanding and responding to human commands in real-world environments.

**Key Concepts**:
- Vision-Language-Action models and architectures
- Natural language processing for robotics
- Multimodal perception and reasoning
- Task planning from natural language commands
- Human-robot interaction and communication
- End-to-end learning for VLA systems

**Skills Gained**:
- Implement VLA models for robotic tasks
- Process natural language commands for robotic action
- Integrate vision and language processing
- Design human-robot interaction systems
- Create end-to-end trainable robotic systems

**Weekly Alignment**: Weeks 10-12 of the 13-week roadmap

**Deliverables/Labs**:
- Lab 1: Basic vision-language integration
- Lab 2: Natural language command processing
- Lab 3: VLA system implementation and testing
- Assignment: Create a voice-controlled robotic task

## Capstone Specification

### Capstone Title: The Autonomous Humanoid

**Description**: The capstone project integrates all concepts learned throughout the modules into a comprehensive autonomous humanoid robot system capable of receiving voice commands and executing complex tasks involving planning, navigation, perception, and manipulation.

**Voice Command → Planning → Navigation → Perception → Manipulation**:
1. **Voice Command**: System receives natural language commands from users
2. **Planning**: Task planning and motion planning for complex multi-step operations
3. **Navigation**: Autonomous navigation to target locations while avoiding obstacles
4. **Perception**: Object recognition, scene understanding, and state estimation
5. **Manipulation**: Physical interaction with objects to complete requested tasks

**Requirements**:
- Accept natural language voice commands
- Plan multi-step tasks combining navigation and manipulation
- Navigate safely in dynamic environments
- Recognize and interact with objects in the environment
- Execute manipulation tasks with precision
- Integrate all four modules' concepts cohesively

**Success Criteria**:
- Successfully interpret and execute 90% of voice commands within scope
- Navigate to target locations with 95% success rate
- Manipulate objects with 85% success rate
- Complete end-to-end tasks in less than 5 minutes on average
- System demonstrates robustness to environmental variations

**Evaluation Rubric**:
- Integration Quality (30%): How well modules are integrated and communicate
- Task Completion (30%): Success rate of completing requested tasks
- Robustness (20%): Performance under environmental variations
- Innovation (20%): Creative solutions and improvements to baseline approach

## Weekly Roadmap (Weeks 1–13)

| Week | Weekly Topic | Learning Objectives | Required Tools/Software | Lab or Assignment |
|------|--------------|-------------------|------------------------|-------------------|
| 1 | Introduction to Physical AI & ROS 2 Fundamentals | Understand Physical AI concepts; Install and configure ROS 2 | ROS 2 Humble Hawksbill, Ubuntu 22.04 | Lab 1: ROS 2 workspace setup and basic publisher/subscriber |
| 2 | ROS 2 Architecture & Communication | Master ROS 2 communication patterns; Design distributed systems | ROS 2 tools, custom message definitions | Lab 2: Service and action implementation |
| 3 | ROS 2 System Integration | Integrate multiple nodes; Launch complex systems | ROS 2 launch files, rqt tools | Lab 3: Multi-node system with launch configuration |
| 4 | Digital Twin Fundamentals & Gazebo | Create simulation environments; Model robots in Gazebo | Gazebo, URDF, XACRO | Lab 4: Basic robot model and simulation |
| 5 | Advanced Simulation & Sensor Modeling | Implement realistic sensors; Design complex environments | Gazebo plugins, sensor simulation | Lab 5: Advanced sensor simulation and environment |
| 6 | Unity Integration & Visualization | Integrate Unity for enhanced visualization; Sim-to-real concepts | Unity 3D, Isaac Sim | Lab 6: Unity integration with simulation |
| 7 | NVIDIA Isaac Platform & Perception | Install and configure Isaac; Implement basic perception | NVIDIA Isaac, Jetson platform | Lab 7: Basic perception pipeline |
| 8 | Isaac Motion Planning & Control | Design motion planning algorithms; Implement control systems | Isaac navigation, control libraries | Lab 8: Motion planning and control |
| 9 | Isaac Edge Deployment & Optimization | Deploy AI models on edge; Optimize for real-time performance | Jetson platform, TensorRT | Lab 9: Edge deployment and optimization |
| 10 | Vision-Language Integration | Combine visual and language processing; VLA concepts | Vision and NLP libraries | Lab 10: Basic VLA integration |
| 11 | Natural Language Processing for Robotics | Process voice commands; Map language to actions | Speech recognition, NLP tools | Lab 11: Voice command processing |
| 12 | VLA System Integration | Integrate all VLA components; Test end-to-end systems | All modules integrated | Lab 12: Complete VLA system implementation |
| 13 | Capstone Project Integration & Presentation | Integrate all modules; Present capstone project | Complete system integration | Capstone Project: Autonomous humanoid demonstration |

## Learning Outcomes

### Knowledge Outcomes
- Understand the theoretical foundations of Physical AI and embodied intelligence
- Master the architecture and components of modern robotic systems
- Comprehend the principles of digital twin technology and sim-to-real transfer
- Know the capabilities and limitations of AI-powered robotics platforms
- Understand the integration challenges between perception, planning, and action systems

### Skill Outcomes
- Design and implement ROS 2-based robotic systems
- Create and validate digital twin simulations
- Deploy AI algorithms on edge computing platforms
- Integrate vision, language, and action systems
- Troubleshoot and optimize complex robotic applications

### Behavioral / Competency Outcomes
- Apply systematic approaches to robotic system design
- Demonstrate proficiency in industry-standard robotics tools and platforms
- Integrate multiple technical domains effectively
- Communicate technical concepts clearly
- Work independently on complex technical projects

## Hardware Specifications

### Digital Twin Workstation
| Component | Minimum Spec | Recommended Spec | Rationale |
|-----------|--------------|------------------|-----------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i9 / AMD Ryzen 9 | Simulation requires high computational power |
| RAM | 16GB | 32GB+ | Complex simulations and multiple processes |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 4080 16GB | Physics simulation and rendering |
| Storage | 500GB SSD | 1TB+ NVMe SSD | Fast loading of simulation assets |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | ROS 2 compatibility |

### Jetson Edge AI Kit
| Component | Minimum Spec | Recommended Spec | Rationale |
|-----------|--------------|------------------|-----------|
| Platform | Jetson Nano | Jetson Orin AGX | AI inference performance |
| RAM | 4GB | 32GB | Complex AI model execution |
| Storage | 16GB eMMC | 64GB+ SD Card | Model storage and data processing |
| Power | 5V/4A AC | 19V/6.3A AC | Sufficient power for full performance |

### Sensor Suite
| Component | Minimum Spec | Recommended Spec | Rationale |
|-----------|--------------|------------------|-----------|
| RGB Camera | 720p, 30fps | 4K, 60fps | High-quality visual input |
| Depth Sensor | Basic stereo | Intel RealSense D455 | Accurate depth perception |
| IMU | Basic 9-DOF | High-precision IMU | Accurate orientation tracking |
| Microphone | Single array | Multi-array beamforming | Voice command recognition |

### Robot Lab Options
| Option | Components | Cost Range | Best For |
|--------|------------|------------|----------|
| Budget | Basic mobile base, single arm, limited sensors | $2,000-$5,000 | Individual learning |
| Standard | Differential drive, 5-DOF arm, full sensor suite | $8,000-$15,000 | Lab settings |
| Premium | Humanoid platform, full mobility, advanced sensors | $20,000-$50,000 | Advanced research |

## Lab Architecture Diagram (Textual Description)

### Simulation Rig
- High-performance workstation with NVIDIA GPU
- Running Gazebo for physics simulation
- Unity for advanced visualization
- Connected to development environment

### Jetson Edge Device
- NVIDIA Jetson platform for AI processing
- Connected to robot sensors and actuators
- Running perception and control algorithms
- Communicating with simulation via ROS 2

### Sensors and Actuators
- RGB-D cameras for visual perception
- IMU for orientation tracking
- Microphone array for voice commands
- Motors and servos for physical action

### Cloud Alternative ("Ether Lab")
- Remote access to shared hardware resources
- Virtualized simulation environments
- Web-based development interface
- Containerized robot applications

### Data Flow Between Components
- Sensor data flows from physical/virtual sensors to perception algorithms
- Perception outputs feed into planning and decision-making systems
- Control commands flow to actuators for physical action
- Simulation and real systems maintain synchronized state
- User commands enter through voice or text interfaces

## Risks & Constraints

### Latency Trap (Cloud → Real Robot)
**Risk**: Network latency between cloud-based AI and real robot causing unsafe behavior
**Mitigation**: Edge computing deployment, local safety systems, latency monitoring
**Impact**: High - safety and real-time performance concerns

### GPU VRAM Requirements
**Risk**: High VRAM requirements for complex AI models exceeding available resources
**Mitigation**: Model optimization, quantization, tiered model deployment
**Impact**: Medium - performance limitations but not safety critical

### OS Constraints (Ubuntu)
**Risk**: Limited compatibility with certain hardware or software components
**Mitigation**: Virtualization, containerization, dual-boot options
**Impact**: Low - development environment flexibility

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Academic Learner Completes Core Modules (Priority: P1)

Academic learners with computer science or AI background engage with the book's core modules to understand Physical AI and Humanoid Robotics concepts. They read content, follow tutorials, and complete hands-on exercises using provided tools and hardware specifications.

**Why this priority**: This represents the core value proposition of the book - providing comprehensive learning material that bridges theory and practice in Physical AI and Humanoid Robotics.

**Independent Test**: Can be fully tested by having a student complete one full module (e.g., Module 1: The Robotic Nervous System) and demonstrate understanding through lab exercises and deliverables.

**Acceptance Scenarios**:
1. **Given** a student has access to the book content and required tools, **When** they complete a module, **Then** they can implement the core concepts in practical exercises
2. **Given** a student has completed all four core modules, **When** they attempt the capstone project, **Then** they can successfully integrate concepts from all modules

---

### User Story 2 - Educator Adapts Content for Course Curriculum (Priority: P2)

Educators and instructors use the book as a textbook for courses on robotics, AI, or embodied intelligence. They follow the weekly roadmap, adapt lab assignments, and use the hardware specifications to set up classroom environments.

**Why this priority**: The book serves as a structured curriculum that can be adapted for formal education settings, expanding its impact beyond individual learners.

**Independent Test**: Can be tested by having an educator successfully map one module to their course schedule and implement the lab exercises with students.

**Acceptance Scenarios**:
1. **Given** an educator has access to the book, **When** they plan a course using the weekly roadmap, **Then** they can structure 13 weeks of content with appropriate labs and assignments
2. **Given** an educator has implemented the hardware specifications, **When** students engage with the content, **Then** they can complete hands-on exercises successfully

---

### User Story 3 - Researcher References Technical Content (Priority: P3)

Researchers in robotics and AI use the book as a reference for current state-of-the-art techniques in Physical AI, ROS 2, NVIDIA Isaac, and Vision-Language-Action systems. They access specific technical sections and implementation examples.

**Why this priority**: The book serves as a comprehensive technical reference for researchers building on these concepts in their own work.

**Independent Test**: Can be tested by having a researcher find and successfully implement a specific technical approach described in the book.

**Acceptance Scenarios**:
1. **Given** a researcher needs to implement a specific technique from the book, **When** they follow the documented approach, **Then** they can reproduce the results successfully
2. **Given** a researcher is comparing different approaches, **When** they reference the book's content, **Then** they can understand the trade-offs between techniques

---

## Edge Cases

- What happens when students have limited hardware access and must rely on simulation only? The curriculum should provide alternative simulation-only paths with equivalent learning outcomes.
- How does the system handle different operating system environments for the required tools? The book should provide clear installation guides and Docker/container solutions for cross-platform compatibility.
- What occurs when students have varying levels of prerequisite knowledge in robotics and AI? The book should include foundational review materials and clearly marked prerequisites for each module.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The book MUST contain 5,000–7,000 words of original, technical content focused on Physical AI and Humanoid Robotics
- **FR-002**: The book MUST include at least 15 sources with minimum 50% being peer-reviewed academic sources
- **FR-003**: All content MUST be written at Flesch-Kincaid Grade 10–12 readability level for academic audiences
- **FR-004**: The book MUST be authored as a Docusaurus documentation site and deployed to GitHub Pages
- **FR-005**: The book MUST include four core modules covering: Robotic Nervous System (ROS 2), Digital Twin, AI-Robot Brain (NVIDIA Isaac), and Vision-Language-Action
- **FR-006**: The book MUST include a capstone project enabling voice command → planning → navigation → perception → manipulation
- **FR-007**: The book MUST provide hardware specifications for Digital Twin workstation, Jetson Edge AI kit, and sensor suite
- **FR-008**: The book MUST include a 13-week roadmap with learning objectives, tools/software, and lab assignments
- **FR-009**: The book MUST provide minimum and recommended hardware specifications with rationale for each component
- **FR-010**: The book MUST include learning outcomes covering knowledge, skills, and behavioral competencies
- **FR-011**: The book MUST document the lab architecture including simulation rig, Jetson edge device, sensors, actuators, and cloud alternative
- **FR-012**: The book MUST include risk mitigation strategies for latency, GPU VRAM requirements, and OS constraints
- **FR-013**: The book MUST support PDF export with embedded citations in APA style
- **FR-014**: The book MUST ensure zero plagiarism with all content properly cited and original
- **FR-015**: The book MUST include reproducible workflows and code examples for all technical concepts
- **FR-016**: The book MUST include detailed module specifications with descriptions, key concepts, skills, and deliverables
- **FR-017**: The book MUST provide clear success criteria and evaluation rubrics for the capstone project
- **FR-018**: The book MUST include structured tables for hardware specifications comparing minimum vs recommended specs

### Key Entities

- **Book Content**: Academic technical content covering Physical AI and Humanoid Robotics topics, including text, code examples, diagrams, and references
- **Module**: Structured learning unit containing description, key concepts, skills gained, weekly alignment, deliverables, and labs
- **Hardware Specification**: Technical requirements for Digital Twin workstation, Jetson Edge AI kit, sensors, and lab infrastructure
- **Learning Outcome**: Measurable knowledge, skill, or competency that learners should achieve
- **Lab Exercise**: Hands-on practical activity that reinforces theoretical concepts with implementation
- **Capstone Project**: Comprehensive integration project demonstrating mastery of all modules
- **Weekly Roadmap**: Structured 13-week curriculum with topics, objectives, tools, and assignments

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The book contains between 5,000–7,000 words of original, technical content (measured by word count)
- **SC-002**: The book includes at least 15 sources with 50% (7-8) being peer-reviewed academic sources (measured by citation analysis)
- **SC-003**: All content maintains Flesch-Kincaid Grade 10–12 readability level (measured by readability analysis tools)
- **SC-004**: The book is successfully published as a Docusaurus site and accessible via GitHub Pages (measured by deployment verification)
- **SC-005**: Students can complete all four core modules and demonstrate understanding through lab exercises (measured by lab completion rates)
- **SC-006**: The capstone project successfully integrates voice command → planning → navigation → perception → manipulation (measured by project completion and functionality verification)
- **SC-007**: Zero plagiarism is detected in the final content (measured by plagiarism detection tools)
- **SC-008**: All factual claims in the book are supported by credible sources (measured by citation verification)
- **SC-009**: The 13-week roadmap provides clear learning objectives and assignments for each week (measured by curriculum completeness)
- **SC-010**: Students can successfully set up and use the hardware specifications provided (measured by setup success rates)
- **SC-011**: The capstone project achieves 85%+ success rate for core functionality (measured by demonstration evaluation)
- **SC-012**: Students demonstrate proficiency in integrating concepts from all four modules (measured by capstone evaluation)
