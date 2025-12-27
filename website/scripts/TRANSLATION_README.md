# Book Translation System

This system allows you to translate the entire Physical AI & Humanoid Robotics book into any language using AI translation services.

## Installation

First, install the required dependencies:

```bash
cd website
npm install @google-cloud/translate deepl-node openai
```

## Configuration

Set up your API keys for translation services in environment variables:

### Google Translate
```bash
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
export GOOGLE_TRANSLATE_API_KEY="your-api-key"
```

### DeepL
```bash
export DEEPL_API_KEY="your-api-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Command Line
```bash
# Translate to Spanish using mock service (for testing)
npm run translate:es

# Translate to French using Google Translate
node scripts/advanced-translate-book.js fr google

# Translate to German using DeepL
node scripts/advanced-translate-book.js de deepl

# Translate to any language with a specific service
node scripts/advanced-translate-book.js <language-code> <service>
```

### Available Languages
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi
- `pt` - Portuguese
- `it` - Italian

### Available Services
- `google` - Google Cloud Translation API
- `deepl` - DeepL API
- `openai` - OpenAI GPT models
- `mock` - Mock translation for testing (default)

## How It Works

1. **Markdown Parsing**: The system carefully parses markdown files, identifying translatable text while preserving code blocks, links, and technical terms.

2. **Smart Translation**: Technical terms and code elements are preserved during translation to maintain accuracy.

3. **Batch Processing**: Files are processed in batches to respect API rate limits.

4. **Language Directories**: Translated files are stored in language-specific subdirectories.

5. **Docusaurus Integration**: The system automatically updates the Docusaurus configuration to include the new language.

## File Structure

After translation, files are organized as:
```
docs/
├── module-1/
│   ├── index.md
│   ├── index.es.md (Spanish translation)
│   ├── index.fr.md (French translation)
│   └── ...
├── module-2/
│   ├── index.md
│   ├── index.es.md
│   └── ...
└── ...
```

## Customization

You can customize the translation behavior by modifying `scripts/translation-config.js`:

- Add new languages
- Configure terminology mappings
- Adjust rate limits
- Modify content that should not be translated

## Limitations

- Translation quality depends on the selected service
- Technical terms might need manual review
- Complex markdown formatting may require post-processing
- API costs apply when using commercial services