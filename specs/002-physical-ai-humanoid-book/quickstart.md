# Quick Start Guide: Physical AI & Humanoid Robotics

## Overview
This quick start guide provides a rapid introduction to the Physical AI & Humanoid Robotics book project. It outlines the initial setup, development workflow, and basic structure to help you get started quickly with creating and validating the book content.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (primary), with support for other Linux distributions
- **RAM**: Minimum 16GB (32GB+ recommended for simulation)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: NVIDIA RTX with CUDA support (RTX 3060 8GB minimum, RTX 4080+ recommended)
- **Storage**: 500GB+ SSD (1TB+ recommended for simulation assets)
- **Network**: Stable internet connection for package downloads and research

### Software Dependencies
```bash
# Essential development tools
sudo apt update
sudo apt install python3 python3-pip nodejs npm git

# ROS 2 Humble Hawksbill setup
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash

# Node.js for Docusaurus (if not installed via package manager)
# Follow NodeSource setup for your Ubuntu version

# Python dependencies for development
pip3 install flake8 pytest
```

## Initial Project Setup

### 1. Clone and Initialize the Repository
```bash
# Clone the repository
git clone <repository-url>
cd humanoid-robot-book

# Initialize the project structure
mkdir -p website/docs/{module-1,module-2,module-3,module-4,capstone}
mkdir -p website/src/{components,pages,css}
mkdir -p website/static/{img,assets}

# Initialize Docusaurus if not already done
cd website
npm init docusaurus@latest . classic
```

### 2. Set Up Development Environment
```bash
# Navigate to website directory
cd website

# Install dependencies
npm install

# Create initial documentation structure
touch docs/{intro.md,modules.md,capstone.md}
```

### 3. Configure Docusaurus for the Book
Edit `docusaurus.config.js` with the following configuration:

```javascript
// docusaurus.config.js
module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Embodied Intelligence in the Real World',
  favicon: 'img/favicon.ico',

  url: 'https://your-username.github.io',
  baseUrl: '/humanoid-robot-book/',
  organizationName: 'your-username',
  projectName: 'humanoid-robot-book',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-username/humanoid-robot-book/tree/main/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book Content',
        },
        {
          href: 'https://github.com/your-username/humanoid-robot-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Content',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
    },
  },
};
```

## Development Workflow

### 1. Creating Book Content
Content follows the 4-module structure:

#### Module 1: The Robotic Nervous System (ROS 2)
- Create files in `website/docs/module-1/`
- Follow ROS 2 concepts and architecture
- Include code examples and diagrams
- Target word count: ~1,200-1,500 words

#### Module 2: The Digital Twin (Gazebo & Unity)
- Create files in `website/docs/module-2/`
- Focus on simulation and digital twin concepts
- Include Gazebo and Unity integration examples
- Target word count: ~1,200-1,500 words

#### Module 3: The AI-Robot Brain (NVIDIA Isaac)
- Create files in `website/docs/module-3/`
- Cover perception, planning, and control with Isaac
- Include edge deployment examples
- Target word count: ~1,200-1,500 words

#### Module 4: Vision-Language-Action (VLA)
- Create files in `website/docs/module-4/`
- Integrate vision, language, and action systems
- Focus on HRI and multimodal systems
- Target word count: ~1,000-1,300 words

#### Capstone: The Autonomous Humanoid
- Create files in `website/docs/capstone/`
- Integrate all previous modules
- Focus on voice command to manipulation pipeline
- Target word count: ~800-1,200 words

### 2. Adding Citations
All content must follow APA citation style:

```markdown
According to recent research, embodied AI systems show significant performance improvements when deployed with proper sim-to-real transfer techniques (Smith et al., 2023).

References:

Smith, J., Johnson, A., & Williams, B. (2023). Embodied Intelligence in Robotic Systems. *Journal of Robotics*, 15(3), 45-67. https://doi.org/10.1234/example
```

### 3. Validation Process
Before committing content, validate:

#### Content Validation
```bash
# Check for broken links
npm run build
npx docusaurus-links-validator

# Validate word count (install wc utility if not present)
find docs/ -name "*.md" -exec cat {} \; | wc -w
# Should be between 5000-7000 words for content (excluding references)

# Check readability (install textstat if needed)
# python -c "import textstat; print(textstat.flesch_kincaid_grade(open('path/to/file.md').read()))"
# Should be between 10-12
```

#### Technical Validation
```bash
# Build the site to catch any formatting errors
npm run build

# Run a local server to preview changes
npm start

# Verify all code examples execute correctly in ROS 2 environment
```

## Basic Content Creation

### Creating a New Chapter
1. Create a new markdown file in the appropriate module directory:
```bash
touch website/docs/module-1/new-chapter.md
```

2. Add frontmatter and content:
```markdown
---
sidebar_position: 2
title: "ROS 2 Communication Patterns"
---

# ROS 2 Communication Patterns

## Overview
This chapter covers the fundamental communication patterns in ROS 2...

## Publisher-Subscriber Pattern
The publisher-subscriber pattern allows for asynchronous communication between nodes...

### Code Example
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
```

### Creating Lab Exercises
Create hands-on exercises in each module directory:

```markdown
---
sidebar_position: 10
title: "Lab 1: Basic ROS 2 Publisher/Subscriber"
---

# Lab 1: Basic ROS 2 Publisher/Subscriber

## Objectives
- Implement a basic publisher node
- Create a subscriber node
- Test communication between nodes

## Prerequisites
- ROS 2 Humble installed
- Basic Python knowledge

## Steps
1. Create a new package...
2. Implement publisher...
3. Implement subscriber...
4. Test communication...

## Deliverables
- Working publisher-subscriber system
- Documentation of findings

## Success Criteria
- Messages successfully transmitted between nodes
- Understanding of ROS 2 communication patterns
```

## Quality Assurance

### Automated Checks
```bash
# Run these checks before committing
npm run build  # Check for build errors
npx markdown-link-check **/*.md  # Check for broken links
# Run plagiarism detection tools
# Validate APA citations
```

### Manual Reviews
- Verify all technical claims against primary sources
- Ensure readability meets Flesch-Kincaid Grade 10-12
- Confirm all citations follow APA style
- Test all code examples in environment

## Deployment to GitHub Pages

### 1. Prepare for Deployment
```bash
# Build the static site
npm run build

# Verify the build succeeds
ls -la build/
```

### 2. Deploy
```bash
# Deploy to GitHub Pages
GIT_USER=<Your GitHub username> \
  CURRENT_BRANCH=main \
  USE_SSH=true \
  npm run deploy
```

### 3. Verify Deployment
- Check the deployed site at `https://<Your GitHub username>.github.io/humanoid-robot-book/`
- Verify all links work correctly
- Test navigation through all modules

## Troubleshooting

### Common Issues
1. **Docusaurus build fails**: Check for syntax errors in markdown files
2. **ROS 2 commands not found**: Ensure ROS 2 environment is sourced
3. **Links not working**: Verify all relative paths are correct
4. **Citations not formatted properly**: Double-check APA style compliance

### Quick Fixes
```bash
# Refresh ROS 2 environment
source /opt/ros/humble/setup.bash

# Clean and rebuild Docusaurus
rm -rf build/
npm run build

# Clear browser cache if preview doesn't update
```

## Next Steps
1. Follow the detailed 13-week roadmap in the specification
2. Implement content for each module according to the weekly breakdown
3. Create lab exercises and assignments
4. Validate content against all constitutional requirements
5. Complete the capstone project integration
6. Prepare for final publication and deployment