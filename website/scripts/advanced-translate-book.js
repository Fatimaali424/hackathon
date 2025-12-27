const fs = require('fs').promises;
const path = require('path');
const { createWriteStream } = require('fs');
const { pipeline } = require('stream/promises');
const translationConfig = require('./translation-config');

// For actual translation, you would need to install these packages:
// npm install @google-cloud/translate @opentelemetry/api deepl-node

class BookTranslator {
  constructor(targetLanguage, service = 'mock') {
    this.targetLanguage = targetLanguage;
    this.service = service;
    this.config = translationConfig;

    // Initialize translation service
    this.translationService = this.initializeService(service);
  }

  initializeService(serviceType) {
    switch (serviceType) {
      case 'google':
        try {
          const { Translate } = require('@google-cloud/translate').v2;
          return new Translate({
            projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
            key: process.env.GOOGLE_TRANSLATE_API_KEY
          });
        } catch (e) {
          console.warn('Google Translate not available:', e.message);
          return null;
        }
      case 'deepl':
        try {
          const deepl = require('deepl-node');
          const authKey = process.env.DEEPL_API_KEY;
          return new deepl.Translator(authKey);
        } catch (e) {
          console.warn('DeepL not available:', e.message);
          return null;
        }
      case 'openai':
        try {
          const { OpenAIApi, Configuration } = require('openai');
          const config = new Configuration({
            apiKey: process.env.OPENAI_API_KEY,
          });
          return new OpenAIApi(config);
        } catch (e) {
          console.warn('OpenAI not available:', e.message);
          return null;
        }
      default:
        // Mock service for demonstration
        return {
          translateText: async (text) => {
            console.log(`Mock translation to ${this.targetLanguage}: ${text.substring(0, 30)}...`);
            return text; // Return original text for mock
          }
        };
    }
  }

  // Read all markdown files from a directory recursively
  async readMarkdownFiles(dir, fileList = []) {
    const files = await fs.readdir(dir);

    for (const file of files) {
      const filePath = path.join(dir, file);
      const stat = await fs.stat(filePath);

      if (stat.isDirectory()) {
        await this.readMarkdownFiles(filePath, fileList);
      } else if (path.extname(file) === '.md') {
        fileList.push(filePath);
      }
    }

    return fileList;
  }

  // Parse markdown content into translatable and non-translatable parts
  parseMarkdownContent(content) {
    const parts = [];
    let currentText = '';
    let i = 0;

    while (i < content.length) {
      const char = content[i];

      // Check for code blocks (``` or ~~~)
      if ((char === '`' || char === '~') && content.substring(i, i + 3) === '```') {
        // Save any accumulated text
        if (currentText) {
          parts.push({ type: 'text', content: currentText });
          currentText = '';
        }

        // Find the end of the code block
        const endMarker = content.substring(i, i + 3);
        const endIdx = content.indexOf(endMarker, i + 3);
        const codeBlockEnd = endIdx !== -1 ? endIdx + 3 : content.length;

        parts.push({
          type: 'code',
          content: content.substring(i, codeBlockEnd),
          language: this.extractCodeLanguage(content.substring(i, i + 30))
        });
        i = codeBlockEnd;
        continue;
      }

      // Check for inline code (`...`)
      if (char === '`') {
        // Save any accumulated text
        if (currentText) {
          parts.push({ type: 'text', content: currentText });
          currentText = '';
        }

        // Find the end of inline code
        const endIdx = content.indexOf('`', i + 1);
        const inlineCodeEnd = endIdx !== -1 ? endIdx + 1 : content.length;

        parts.push({ type: 'inlineCode', content: content.substring(i, inlineCodeEnd) });
        i = inlineCodeEnd;
        continue;
      }

      // Check for links [text](url)
      if (char === '[') {
        const linkEnd = this.findLinkEnd(content, i);
        if (linkEnd !== -1) {
          // Save any accumulated text
          if (currentText) {
            parts.push({ type: 'text', content: currentText });
            currentText = '';
          }

          const linkContent = content.substring(i, linkEnd);
          const { text: linkText, url } = this.parseLink(linkContent);

          parts.push({
            type: 'link',
            original: linkContent,
            text: linkText,
            url: url
          });
          i = linkEnd;
          continue;
        }
      }

      // Check for headers
      if (char === '#' && (i === 0 || content[i - 1] === '\n')) {
        // Save any accumulated text
        if (currentText) {
          parts.push({ type: 'text', content: currentText });
          currentText = '';
        }

        // Find the end of the header line
        const endIdx = content.indexOf('\n', i);
        const headerEnd = endIdx !== -1 ? endIdx : content.length;

        parts.push({
          type: 'header',
          content: content.substring(i, headerEnd)
        });
        i = endIdx !== -1 ? endIdx + 1 : endIdx;
        continue;
      }

      currentText += char;
      i++;
    }

    // Add any remaining text
    if (currentText) {
      parts.push({ type: 'text', content: currentText });
    }

    return parts;
  }

  extractCodeLanguage(codeBlockStart) {
    // Extract language from ```language marker
    const match = codeBlockStart.match(/```(\w+)/);
    return match ? match[1] : 'unknown';
  }

  findLinkEnd(content, startIndex) {
    let bracketCount = 1;
    let i = startIndex + 1;

    // Find the matching closing bracket for the link text
    while (i < content.length && bracketCount > 0) {
      if (content[i] === '[') bracketCount++;
      else if (content[i] === ']') bracketCount--;
      i++;
    }

    if (bracketCount !== 0) return -1; // Malformed link

    // Now look for the opening parenthesis for the URL
    if (content[i] !== '(') return -1;

    // Find the closing parenthesis for the URL
    let parenCount = 1;
    i++;
    const urlStart = i;

    while (i < content.length && parenCount > 0) {
      if (content[i] === '(') parenCount++;
      else if (content[i] === ')') parenCount--;
      i++;
    }

    if (parenCount !== 0) return -1; // Malformed link

    return i;
  }

  parseLink(linkContent) {
    const linkMatch = linkContent.match(/\[([^\]]*)\]\(([^)]*)\)/);
    if (linkMatch) {
      return { text: linkMatch[1], url: linkMatch[2] };
    }
    return { text: '', url: '' };
  }

  // Reconstruct markdown content from parsed parts
  reconstructMarkdown(parts) {
    return parts.map(part => {
      switch (part.type) {
        case 'text':
          return part.content;
        case 'code':
          return part.content;
        case 'inlineCode':
          return part.content;
        case 'link':
          return `[${part.text}](${part.url})`;
        case 'header':
          return part.content;
        default:
          return part.content;
      }
    }).join('');
  }

  // Translate text content, preserving non-translatable parts
  async translatePart(part) {
    if (!this.translationService) {
      throw new Error(`Translation service not available for ${this.service}`);
    }

    switch (part.type) {
      case 'text':
      case 'header':
        // Only translate text content, apply terminology mapping
        let translatedContent = part.content;

        // Apply custom terminology mappings
        Object.entries(this.config.terminology).forEach(([key, value]) => {
          const regex = new RegExp(`\\b${key}\\b`, 'gi');
          translatedContent = translatedContent.replace(regex, value);
        });

        try {
          const result = await this.translationService.translateText(translatedContent, this.targetLanguage);
          return { ...part, content: result };
        } catch (error) {
          console.error(`Translation error for part:`, error.message);
          return part; // Return original part if translation fails
        }

      case 'code':
      case 'inlineCode':
      case 'link':
        // Don't translate code blocks, inline code, or URLs
        return part;

      default:
        return part;
    }
  }

  // Translate a markdown file
  async translateFile(filePath) {
    try {
      console.log(`Translating: ${filePath}`);

      const content = await fs.readFile(filePath, 'utf8');

      // Separate frontmatter if present
      const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
      let frontmatter = '';
      let body = content;

      if (frontmatterMatch) {
        frontmatter = frontmatterMatch[1];
        body = frontmatterMatch[2];
      }

      // Parse and translate the body
      const parts = this.parseMarkdownContent(body);
      const translatedParts = [];

      for (const part of parts) {
        const translatedPart = await this.translatePart(part);
        translatedParts.push(translatedPart);
      }

      const translatedBody = this.reconstructMarkdown(translatedParts);

      // Reconstruct the full content
      const translatedContent = frontmatter
        ? `---\n${frontmatter}\n---\n${translatedBody}`
        : translatedBody;

      // Create translated file path
      const dir = path.dirname(filePath);
      const fileName = path.basename(filePath, '.md');
      const langDir = path.join(dir, this.targetLanguage);

      // Create language directory if it doesn't exist
      await fs.mkdir(langDir, { recursive: true });

      const translatedFilePath = path.join(langDir, `${fileName}.md`);

      // Write translated content to new file
      await fs.writeFile(translatedFilePath, translatedContent);

      console.log(`✓ Translated: ${filePath} -> ${translatedFilePath}`);

      return translatedFilePath;
    } catch (error) {
      console.error(`✗ Error translating ${filePath}:`, error);
      throw error;
    }
  }

  // Main translation method
  async translateBook() {
    console.log(`🚀 Starting translation to ${this.targetLanguage} using ${this.service} service...`);

    // Read all markdown files from the docs directory
    const docsDir = path.join(__dirname, '..', 'docs');
    const markdownFiles = await this.readMarkdownFiles(docsDir);

    console.log(`📋 Found ${markdownFiles.length} markdown files to translate`);

    // Process files in batches to manage API limits
    const batchSize = this.config.settings.batchSize;

    for (let i = 0; i < markdownFiles.length; i += batchSize) {
      const batch = markdownFiles.slice(i, i + batchSize);

      console.log(`📦 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(markdownFiles.length / batchSize)}`);

      await Promise.all(batch.map(file => this.translateFile(file)));

      // Add delay between batches to respect rate limits
      if (i + batchSize < markdownFiles.length) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
      }
    }

    // Update Docusaurus config for the new language
    await this.updateDocusaurusConfig();

    console.log(`✅ Translation to ${this.targetLanguage} completed!`);
  }

  // Update Docusaurus config to include the new language
  async updateDocusaurusConfig() {
    const configPath = path.join(__dirname, '..', 'docusaurus.config.js');
    let configContent = await fs.readFile(configPath, 'utf8');

    // Check if the language is already configured
    if (configContent.includes(`'${this.targetLanguage}'`)) {
      console.log(`⚠️  Language ${this.targetLanguage} already configured in docusaurus.config.js`);
      return;
    }

    // Add the new language to the i18n configuration
    if (configContent.includes('i18n: {')) {
      // Find the i18n configuration and add the new language
      const i18nRegex = /(i18n:\s*{[^}]*locales:\s*\[)([^\]]*)(\])/;
      const match = configContent.match(i18nRegex);

      if (match) {
        const currentLocales = match[2];
        let newLocales = currentLocales;

        if (!currentLocales.includes(`'${this.targetLanguage}'`)) {
          newLocales = currentLocales ? `${currentLocales}, '${this.targetLanguage}'` : `'${this.targetLanguage}'`;
        }

        configContent = configContent.replace(i18nRegex, `$1${newLocales}$3`);
      } else {
        // If no locales array found, add one
        configContent = configContent.replace(
          /(i18n:\s*{)/,
          `$1\n    locales: ['en', '${this.targetLanguage}'],`
        );
      }
    } else {
      // If no i18n config exists, add it
      const i18nConfig = `
  i18n: {
    defaultLocale: 'en',
    locales: ['en', '${this.targetLanguage}'],
  },`;

      // Add i18n config before the closing brace of the main config object
      configContent = configContent.replace(
        /(\s*},\s*\n\s*\);\s*$)/,
        `${i18nConfig}$1`
      );
    }

    await fs.writeFile(configPath, configContent);
    console.log(`📝 Updated docusaurus.config.js to include ${this.targetLanguage}`);
  }
}

// Command-line interface
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error('Usage: node advanced-translate-book.js <target-language-code> [service]');
    console.error('Example: node advanced-translate-book.js es google');
    console.error('Available services: google, deepl, openai, mock');
    console.error('Available languages:', Object.keys(translationConfig.supportedLanguages).join(', '));
    process.exit(1);
  }

  const targetLanguage = args[0];
  const service = args[1] || 'mock';

  if (!translationConfig.supportedLanguages[targetLanguage]) {
    console.error(`❌ Unsupported language: ${targetLanguage}`);
    console.error('Available languages:', Object.keys(translationConfig.supportedLanguages).join(', '));
    process.exit(1);
  }

  const translator = new BookTranslator(targetLanguage, service);
  translator.translateBook().catch(console.error);
}

module.exports = BookTranslator;