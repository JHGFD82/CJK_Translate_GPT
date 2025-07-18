# CJK_Translate_GPT
Translate text interchangably between Chinese, Japanese, Korean, and English from either source PDFs or inputted text.

## Important Note
This application is designed exclusively for Princeton University faculty members (or delegates) with access to the AI Sandbox. You must have a valid Princeton University AI Sandbox API key to use this tool.

## Features
- Translate between Chinese, Japanese, Korean, and English in any direction
- Extract text from PDF files and translate
- Support for custom text input
- Save translations to text files or PDF files
- Auto-save functionality with timestamped filenames
- Page range selection for PDF processing
- Abstract context support for better translations

## Requirements
- Python 3.7+
- Princeton University AI Sandbox access
- Valid AI Sandbox API key (Princeton University faculty only)

## Usage

### Basic Translation
```bash
# Translate PDF from Chinese to English
python main.py heller CE -i document.pdf

# Translate custom text from Japanese to Korean
python main.py smith JK -c
```

### Save Output Options
```bash
# Save translation to a specific text file
python main.py heller CE -i document.pdf -o translation.txt

# Save translation to a PDF file
python main.py heller CE -i document.pdf -o translation.pdf

# Auto-save with timestamp in the same directory as input
python main.py heller CE -i document.pdf --auto-save
```

### Advanced Options
```bash
# Translate specific pages with abstract context
python main.py smith CE -i document.pdf -p 5-10 -a

# Translate single page and save to PDF
python main.py heller JE -i document.pdf -p 3 -o page3_translation.pdf
```

### Token Usage Tracking
```bash
# View usage report for a professor
python main.py heller --usage-report

# View daily usage for a specific date
python main.py heller --daily-usage 2024-07-14

# View today's usage
python main.py smith --daily-usage
```

### Progressive Saving
For large documents or unreliable connections, use progressive saving to save each page immediately after translation:

```bash
# Save each page as it's translated (useful for error recovery)
python main.py heller CE -i large_document.txt --progressive-save -o output.txt

# Combine with auto-save for timestamped progressive saving
python main.py smith JK -i document.txt --progressive-save --auto-save
```

**Note:** When using `--progressive-save`, the translation is saved incrementally as each page is processed, which helps prevent data loss if the process is interrupted. Progressive saving only supports text file output (.txt) - PDF output is not compatible with progressive saving.

### Error Handling and Retries
The system includes robust error handling for API failures:

- **Automatic Retries**: If the API returns an error, the system will automatically retry the translation
- **Error Recovery**: If retries fail, the system will insert a translation error message and continue to the next page
- **Graceful Degradation**: Processing continues even when individual pages fail, ensuring you get translations for successful pages

This ensures that temporary API issues or problematic pages don't stop the entire translation process.

## Language Codes
- `C` = Chinese
- `J` = Japanese
- `K` = Korean
- `E` = English

Examples:
- `CE` = Chinese to English
- `JK` = Japanese to Korean
- `EJ` = English to Japanese
- `KC` = Korean to Chinese

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up professor configuration in `.env` file:

   Copy the example file and configure professors:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add professor configurations:
   ```env
   # Professor Configuration
   PROF_1_NAME=heller
   PROF_1_KEY=your_api_key_here
   PROF_1_BACKUP_KEY=your_backup_key_here
   
   PROF_2_NAME=smith
   PROF_2_KEY=another_api_key_here
   PROF_2_BACKUP_KEY=another_backup_key_here
   ```
   
   **Getting your AI Sandbox API key:**
   Princeton University faculty can obtain their AI Sandbox API key through OIT.
   
   **Professor Selection:**
   Use professor names as configured in the `.env` file. The system supports:
   - Simple names (just last names work fine: `heller`, `smith`)
   - Full names (if they contain spaces, use quotes: `"john smith"`)
   - Safe names (spaces converted to underscores: `john_smith`)
   
   Run `python show_config.py` to see available professors and their safe names.

## Token Usage Tracking

Each professor has separate token usage tracking and monthly budgets:
- Usage data is stored in `data/token_usage_[professor_name].json` files
- Monthly limits are configurable in the pricing configuration
- View reports using `--usage-report` and `--daily-usage` options

## Output File Formats

### Text Files (.txt)
- UTF-8 encoded
- Preserves original formatting
- Ideal for further processing

### PDF Files (.pdf)
- Professional formatting
- Supports CJK characters
- Suitable for sharing and archiving

### Auto-Save Naming
When using `--auto-save`, files are saved with the format:
```
[input_filename]_[source_lang]to[target_lang]_[timestamp].[extension]
```

Example: `document_ChinesetoEnglish_20250710_152416.txt`
