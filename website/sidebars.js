// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: '📚 Introduction',
      collapsed: false,
      items: [
        'intro',
      ],
    },
    {
      type: 'category',
      label: '🔌 Module 1: The Robotic Nervous System (ROS 2)',
      collapsible: true,
      collapsed: false,
      items: [
        'module-1/index',
        'module-1/ros2-fundamentals',
        'module-1/ros2-architecture',
        'module-1/ros2-integration',
        {
          type: 'category',
          label: 'Labs',
          collapsible: true,
          collapsed: true,
          items: [
            'module-1/lab-1-publisher-subscriber',
            'module-1/lab-2-services-actions',
            'module-1/lab-3-multi-node',
          ],
        },
        'module-1/assignment',
        // Additional module 1 pages will be added here
      ],
    },
    {
      type: 'category',
      label: '🔄 Module 2: The Digital Twin (Gazebo & Unity)',
      collapsible: true,
      collapsed: true,
      items: [
        'module-2/index',
        'module-2/gazebo-simulation',
        'module-2/unity-integration',
        'module-2/sim-to-real',
        {
          type: 'category',
          label: 'Labs',
          collapsible: true,
          collapsed: true,
          items: [
            'module-2/lab-4-robot-model',
            'module-2/lab-5-sensor-simulation',
            'module-2/lab-6-unity-integration',
          ],
        },
        'module-2/assignment',
        // Additional module 2 pages will be added here
      ],
    },
    {
      type: 'category',
      label: '🧠 Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      collapsible: true,
      collapsed: true,
      items: [
        'module-3/index',
        'module-3/isaac-platform',
        'module-3/motion-planning',
        'module-3/edge-deployment',
        {
          type: 'category',
          label: 'Labs',
          collapsible: true,
          collapsed: true,
          items: [
            'module-3/lab-7-perception-pipeline',
            'module-3/lab-8-motion-control',
            'module-3/lab-9-edge-deployment',
          ],
        },
        'module-3/assignment',
        'module-3/hardware',
        'module-3/learning-outcomes',
      ],
    },
    {
      type: 'category',
      label: '👁️ Module 4: Vision-Language-Action (VLA)',
      collapsible: true,
      collapsed: true,
      items: [
        'module-4/index',
        'module-4/vision-language',
        'module-4/nlp-robotics',
        'module-4/human-robot-interaction',
        {
          type: 'category',
          label: 'Labs',
          collapsible: true,
          collapsed: true,
          items: [
            'module-4/lab-10-vision-language',
            'module-4/lab-11-voice-command',
            'module-4/lab-12-vla-system',
          ],
        },
        'module-4/assignment',
        'module-4/hardware',
        'module-4/learning-outcomes',
      ],
    },
    {
      type: 'category',
      label: '🏆 Capstone: The Autonomous Humanoid',
      collapsible: true,
      collapsed: true,
      items: [
        'capstone/index',
        'capstone/overview',
        'capstone/voice-command',
        'capstone/planning',
        'capstone/navigation',
        'capstone/perception',
        'capstone/manipulation',
        'capstone/integration',
        'capstone/evaluation',
        'capstone/conclusion',
        'capstone/diagrams',
      ],
    },
    {
      type: 'category',
      label: '📚 Reference Materials',
      collapsible: true,
      collapsed: true,
      items: [
        'references',
        'weekly-roadmap',
        'educator-guide',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
