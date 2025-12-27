const fs = require('fs').promises;
const path = require('path');
const { createWriteStream } = require('fs');
const { pipeline } = require('stream/promises');

// Function to read all markdown files from a directory recursively
async function readMarkdownFiles(dir, fileList = []) {
  const files = await fs.readdir(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = await fs.stat(filePath);

    if (stat.isDirectory()) {
      await readMarkdownFiles(filePath, fileList);
    } else if (path.extname(file) === '.md') {
      fileList.push(filePath);
    }
  }

  return fileList;
}

// Function to translate text using a translation API
// Using a mock translation function for demonstration
// In a real implementation, you would use a translation API like Google Translate, DeepL, etc.
async function translateText(text, targetLanguage) {
  // This is a mock translation function
  // In a real implementation, you would use an actual translation API
  console.log(`Translating to ${targetLanguage}: ${text.substring(0, 50)}...`);

  // For demonstration purposes, we'll return the original text
  // In a real implementation, replace this with actual translation API call
  return text;
}

// Function to translate a markdown file
async function translateMarkdownFile(filePath, targetLanguage) {
  try {
    const content = await fs.readFile(filePath, 'utf8');

    // Split content into frontmatter and body
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);

    let frontmatter = '';
    let body = content;

    if (frontmatterMatch) {
      frontmatter = frontmatterMatch[1];
      body = frontmatterMatch[2];
    }

    // Translate the body content
    // Note: We should be careful not to translate code blocks, links, or other special markdown elements
    const translatedBody = await translateText(body, targetLanguage);

    // Create translated content
    const translatedContent = frontmatter
      ? `---\n${frontmatter}\n---\n${translatedBody}`
      : translatedBody;

    // Create translated file path
    const dir = path.dirname(filePath);
    const fileName = path.basename(filePath, '.md');
    const translatedFilePath = path.join(dir, `${fileName}.${targetLanguage}.md`);

    // Write translated content to new file
    await fs.writeFile(translatedFilePath, translatedContent);

    console.log(`Translated: ${filePath} -> ${translatedFilePath}`);

    return translatedFilePath;
  } catch (error) {
    console.error(`Error translating ${filePath}:`, error);
    throw error;
  }
}

// Main translation function
async function translateBook(targetLanguage) {
  console.log(`Starting translation to ${targetLanguage}...`);

  // Read all markdown files from the docs directory
  const docsDir = path.join(__dirname, '..', 'docs');
  const markdownFiles = await readMarkdownFiles(docsDir);

  console.log(`Found ${markdownFiles.length} markdown files to translate`);

  // Process each file
  for (const file of markdownFiles) {
    await translateMarkdownFile(file, targetLanguage);
  }

  console.log(`Translation to ${targetLanguage} completed!`);
}

// Function to translate with actual API (Google Translate example)
async function translateWithGoogle(text, targetLanguage, sourceLanguage = 'en') {
  // This is a placeholder - you would need to install and use google-translate-api
  // npm install google-translate-api-x
  // const translate = require('google-translate-api-x');

  try {
    // In a real implementation:
    // const result = await translate(text, { to: targetLanguage, from: sourceLanguage });
    // return result.text;

    // For now, return original text as placeholder
    return text;
  } catch (error) {
    console.error('Translation API error:', error);
    return text; // Return original text if translation fails
  }
}

// Function to translate with DeepL API
async function translateWithDeepL(text, targetLanguage) {
  // This is a placeholder - you would need to install and use deepl-node
  // npm install deepl-node
  // const deepl = require('deepl-node');

  try {
    // In a real implementation:
    // const translator = new deepl.Translator(process.env.DEEPL_API_KEY);
    // const result = await translator.translateText(text, null, targetLanguage);
    // return result.text;

    // For now, return original text as placeholder
    return text;
  } catch (error) {
    console.error('DeepL API error:', error);
    return text; // Return original text if translation fails
  }
}

// Enhanced translation function that handles markdown properly
async function translateMarkdownContent(content, targetLanguage, translationMethod = 'mock') {
  // Split content into sections to preserve code blocks, links, etc.
  const sections = splitMarkdownIntoTranslatableSections(content);

  const translatedSections = [];

  for (const section of sections) {
    if (section.type === 'translatable') {
      let translatedText;
      switch (translationMethod) {
        case 'google':
          translatedText = await translateWithGoogle(section.content, targetLanguage);
          break;
        case 'deepl':
          translatedText = await translateWithDeepL(section.content, targetLanguage);
          break;
        case 'mock':
        default:
          translatedText = section.content; // Placeholder
      }
      translatedSections.push(translatedText);
    } else {
      // Don't translate non-translatable sections (code blocks, links, etc.)
      translatedSections.push(section.content);
    }
  }

  return translatedSections.join('');
}

// Function to split markdown into translatable and non-translatable sections
function splitMarkdownIntoTranslatableSections(content) {
  const sections = [];
  let currentSection = '';
  let inCodeBlock = false;
  let inInlineCode = false;
  let i = 0;

  while (i < content.length) {
    const char = content[i];
    const nextChar = content[i + 1];
    const nextTwoChars = content.substring(i, i + 3);

    // Check for code block start/end (``` or ~~~)
    if ((char === '`' || char === '~') && nextTwoChars === '```') {
      if (currentSection) {
        sections.push({ type: inCodeBlock ? 'non-translatable' : 'translatable', content: currentSection });
        currentSection = '';
      }
      inCodeBlock = !inCodeBlock;

      // Add the code block marker to non-translatable section
      let codeBlockEnd = content.indexOf(nextTwoChars, i + 3);
      if (codeBlockEnd === -1) codeBlockEnd = content.length;
      codeBlockEnd += 3; // Include the closing markers

      sections.push({ type: 'non-translatable', content: content.substring(i, codeBlockEnd) });
      i = codeBlockEnd;
      continue;
    }

    // Check for inline code start/end (`...`)
    if (char === '`' && !inCodeBlock) {
      if (currentSection) {
        sections.push({ type: inInlineCode ? 'non-translatable' : 'translatable', content: currentSection });
        currentSection = '';
      }
      inInlineCode = !inInlineCode;

      // Find the closing backtick
      let inlineCodeEnd = content.indexOf('`', i + 1);
      if (inlineCodeEnd === -1) inlineCodeEnd = content.length;

      sections.push({ type: 'non-translatable', content: content.substring(i, inlineCodeEnd + 1) });
      i = inlineCodeEnd + 1;
      continue;
    }

    // Check for links [text](url)
    if (char === '[' && !inCodeBlock && !inInlineCode) {
      const linkEnd = findLinkEnd(content, i);
      if (linkEnd !== -1) {
        if (currentSection) {
          sections.push({ type: 'translatable', content: currentSection });
          currentSection = '';
        }

        const linkContent = content.substring(i, linkEnd);
        const [linkText, linkUrl] = parseLink(linkContent);

        // Translate only the link text, not the URL
        sections.push({ type: 'non-translatable', content: `[${linkText}](${linkUrl})` });
        i = linkEnd;
        continue;
      }
    }

    currentSection += char;
    i++;
  }

  if (currentSection) {
    sections.push({ type: inCodeBlock || inInlineCode ? 'non-translatable' : 'translatable', content: currentSection });
  }

  return sections;
}

// Helper function to find the end of a markdown link
function findLinkEnd(content, startIndex) {
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

// Helper function to parse a markdown link
function parseLink(linkContent) {
  const linkMatch = linkContent.match(/\[([^\]]*)\]\(([^)]*)\)/);
  if (linkMatch) {
    return [linkMatch[1], linkMatch[2]];
  }
  return ['', ''];
}

// Export functions for use in other modules
module.exports = {
  translateBook,
  translateMarkdownFile,
  translateMarkdownContent,
  splitMarkdownIntoTranslatableSections,
  translateWithGoogle,
  translateWithDeepL
};

// If this script is run directly, execute the translation
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error('Usage: node translate-book.js <target-language-code>');
    console.error('Example: node translate-book.js es (for Spanish)');
    process.exit(1);
  }

  const targetLanguage = args[0];
  translateBook(targetLanguage).catch(console.error);
}