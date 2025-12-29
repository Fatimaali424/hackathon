// Translation configuration file

const translationConfig = {
  // Supported languages with their ISO codes
  supportedLanguages: {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'pt': 'Portuguese',
    'it': 'Italian',
  },

  // Translation service configuration
  translationServices: {
    google: {
      enabled: true,
      apiKey: process.env.GOOGLE_TRANSLATE_API_KEY || null,
      rateLimit: 100 // requests per minute
    },
    deepl: {
      enabled: true,
      apiKey: process.env.DEEPL_API_KEY || null,
      rateLimit: 50 // requests per minute
    },
    openai: {
      enabled: true,
      apiKey: process.env.OPENAI_API_KEY || null,
      model: 'gpt-4-turbo' // or 'gpt-3.5-turbo' for lower cost
    }
  },

  // Content that should not be translated
  doNotTranslate: [
    // Code blocks and inline code
    '```.*?```',
    '`.*?`',
    // File paths and code elements
    '\\b\\w+\\.\\w+\\b', // file extensions
    // URLs and links
    'https?://[\\w\\./-]+',
    // Technical terms that should remain in English
    'ROS',
    'Gazebo',
    'Unity',
    'NVIDIA Isaac',
    'VLA',
    'AI',
    'API',
    'JSON',
    'XML',
    'HTML',
    'CSS',
    'JavaScript',
    'Python',
    'C++',
    // Variable names and constants
    '[A-Z_][A-Z0-9_]*',
    // Special formatting
    '\\{\\{.*?\\}\\}',
    '\\[.*?\\]\\(.*?\\)', // markdown links
  ],

  // Settings for translation process
  settings: {
    batchSize: 10, // Number of files to process in parallel
    retryAttempts: 3,
    timeout: 30000, // 30 seconds timeout per translation request
    preserveFormatting: true,
    maxTextLength: 5000, // Maximum length of text to send in one request
  },

  // Custom terminology mappings for technical content
  terminology: {
    'physical ai': 'physical AI',
    'humanoid robotics': 'Humanoid Robotics',
    'embodied intelligence': 'Embodied Intelligence',
    'digital twin': 'Digital Twin',
    'sim-to-real': 'sim-to-real',
    'motion planning': 'Motion Planning',
    'human-robot interaction': 'Human-Robot Interaction',
    'robotic systems': 'Robotic Systems',
    'perception pipeline': 'Perception Pipeline',
    'vision-language-action': 'Vision-Language-Action',
    'autonomous humanoid': 'Autonomous Humanoid',
  }
};

module.exports = translationConfig;