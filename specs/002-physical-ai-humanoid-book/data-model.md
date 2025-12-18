# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document defines the conceptual data models relevant to the Physical AI & Humanoid Robotics book. These models represent the key entities and relationships that will be covered in the content, serving as a reference for understanding the information architecture of robotic systems.

## Core Entities

### 1. Book Content Entity
- **Name**: BookContent
- **Description**: Represents the structured content of the academic book
- **Attributes**:
  - id: unique identifier for the book content
  - title: string (Physical AI & Humanoid Robotics)
  - version: string (semantic versioning)
  - wordCount: integer (5000-7000 range)
  - moduleCount: integer (4 core modules)
  - createdDate: date
  - lastUpdated: date
  - status: enum (draft, review, published)
  - citationCount: integer (minimum 15)
  - peerReviewedPercentage: float (minimum 0.5)

### 2. Module Entity
- **Name**: Module
- **Description**: Represents one of the four core modules in the book
- **Attributes**:
  - moduleId: unique identifier for the module
  - title: string (e.g., "Robotic Nervous System (ROS 2)")
  - description: text (detailed module description)
  - weekRange: string (e.g., "Weeks 1-3")
  - wordCount: integer (portion of total)
  - learningObjectives: array of strings
  - skillsGained: array of strings
  - deliverables: array of strings (labs, assignments)
  - toolsRequired: array of strings (software/hardware)
  - dependencies: array of Module IDs
  - parentBookId: reference to BookContent

### 3. Learning Outcome Entity
- **Name**: LearningOutcome
- **Description**: Represents measurable learning outcomes for the book
- **Attributes**:
  - outcomeId: unique identifier
  - type: enum (knowledge, skill, behavioral)
  - description: text (specific learning outcome)
  - targetAudience: enum (student, educator, researcher)
  - moduleIds: array of Module IDs
  - measurableCriteria: text (how to measure achievement)
  - assessmentMethod: string (how to assess)

### 4. Hardware Specification Entity
- **Name**: HardwareSpec
- **Description**: Represents hardware configurations for the book's projects
- **Attributes**:
  - specId: unique identifier
  - name: string (e.g., "Digital Twin Workstation")
  - category: enum (simulation, edge, sensor, robot, complete_system)
  - minimumSpec: object with components (CPU, RAM, GPU, etc.)
  - recommendedSpec: object with components
  - rationale: text (reasoning for specifications)
  - costRange: string (e.g., "$2,000-$5,000")
  - useCase: string (when to use this configuration)
  - compatibility: array of strings (OS, software compatibility)

### 5. Lab Exercise Entity
- **Name**: LabExercise
- **Description**: Represents hands-on exercises in the book
- **Attributes**:
  - labId: unique identifier
  - title: string (lab name)
  - description: text (detailed lab description)
  - moduleIds: array of Module IDs
  - difficultyLevel: enum (beginner, intermediate, advanced)
  - estimatedTime: string (e.g., "2-3 hours")
  - requiredTools: array of strings
  - requiredHardware: array of HardwareSpec IDs
  - learningObjectives: array of strings
  - deliverables: array of strings (what students produce)
  - successCriteria: text (how to know it's complete)
  - dependencies: array of LabExercise IDs

### 6. Citation Entity
- **Name**: Citation
- **Description**: Represents academic citations in the book
- **Attributes**:
  - citationId: unique identifier
  - type: enum (academic_paper, technical_documentation, standard, website)
  - title: string (title of source)
  - authors: array of strings
  - publicationDate: date
  - publisher: string
  - url: string (if available online)
  - doi: string (if academic paper)
  - peerReviewed: boolean (true if peer-reviewed)
  - apaFormatted: string (APA style citation)
  - contentReferences: array of strings (where in book this is referenced)
  - verificationStatus: enum (verified, pending, needs_review)

### 7. Capstone Project Entity
- **Name**: CapstoneProject
- **Description**: Represents the integrated capstone project
- **Attributes**:
  - projectId: unique identifier
  - title: string ("The Autonomous Humanoid")
  - description: text (detailed project description)
  - requirements: array of strings (technical requirements)
  - successCriteria: array of strings (success metrics)
  - evaluationRubric: object (grading criteria)
  - integrationPoints: array of strings (how modules connect)
  - voiceCommandFlow: string (voice → planning → navigation → perception → manipulation)
  - requiredHardware: array of HardwareSpec IDs
  - requiredSoftware: array of strings
  - estimatedCompletionTime: string (e.g., "4-6 weeks")

### 8. Weekly Roadmap Entity
- **Name**: WeeklyRoadmap
- **Description**: Represents the 13-week curriculum structure
- **Attributes**:
  - weekId: integer (1-13)
  - topic: string (weekly topic)
  - learningObjectives: array of strings
  - toolsSoftware: array of strings
  - labAssignment: string (lab or assignment name)
  - moduleIds: array of Module IDs
  - requiredReading: array of strings (book sections to read)
  - estimatedHours: integer (time commitment)

## Relationships

### BookContent to Module
- **Relationship**: One-to-Many
- **Description**: One book contains multiple modules
- **Cardinality**: 1 BookContent → 4+ Modules

### Module to LearningOutcome
- **Relationship**: Many-to-Many
- **Description**: Modules contribute to multiple learning outcomes
- **Cardinality**: M Modules ↔ N LearningOutcomes

### Module to LabExercise
- **Relationship**: One-to-Many
- **Description**: One module contains multiple lab exercises
- **Cardinality**: 1 Module → 3+ LabExercises

### LabExercise to HardwareSpec
- **Relationship**: Many-to-Many
- **Description**: Lab exercises may require multiple hardware specifications
- **Cardinality**: M LabExercises ↔ N HardwareSpecs

### BookContent to Citation
- **Relationship**: One-to-Many
- **Description**: One book contains multiple citations
- **Cardinality**: 1 BookContent → 15+ Citations

### Module to WeeklyRoadmap
- **Relationship**: Many-to-Many
- **Description**: Modules span multiple weeks in the roadmap
- **Cardinality**: M Modules ↔ N WeeklyRoadmap entries

### CapstoneProject to Module
- **Relationship**: Many-to-Many
- **Description**: Capstone project integrates concepts from all modules
- **Cardinality**: 1 CapstoneProject ↔ 4 Modules

## Validation Rules

### Book Content Validation
- Total word count must be between 5,000 and 7,000 words
- Minimum 15 citations required
- At least 50% of citations must be peer-reviewed
- All modules must be linked to the book content

### Module Validation
- Each module must have 1-3 learning outcomes
- Each module must have 1-3 lab exercises
- Module word counts must sum to total book target
- Module dependencies must form a valid sequence

### Citation Validation
- Each citation must have APA format verification
- Peer-reviewed status must be clearly marked
- All technical claims must have source citations
- Citation count must meet constitutional requirements

### Hardware Specification Validation
- Minimum and recommended specs must be clearly differentiated
- Rationale must be provided for each specification
- Cost estimates must be current and realistic
- Compatibility information must be accurate

### Lab Exercise Validation
- Difficulty levels must align with target audience
- Required time estimates must be realistic
- Success criteria must be measurable
- Dependencies must be properly ordered

## State Transitions

### Content Development States
- **Draft**: Initial content creation
- **Reviewed**: Peer review completed
- **Validated**: Technical accuracy verified
- **Published**: Final version ready for deployment

### Citation Verification States
- **Identified**: Source found but not verified
- **Verified**: Source accuracy confirmed
- **Cited**: Properly formatted in content
- **Cross-referenced**: Linked to specific content sections

### Lab Exercise States
- **Designed**: Exercise concept created
- **Tested**: Exercise validated in environment
- **Documented**: Instructions complete
- **Reviewed**: Peer review completed