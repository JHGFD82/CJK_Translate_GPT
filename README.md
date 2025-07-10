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
python src/main.py CE -i document.pdf

# Translate custom text from Japanese to Korean
python src/main.py JK -c
```

### Save Output Options
```bash
# Save translation to a specific text file
python src/main.py CE -i document.pdf -o translation.txt

# Save translation to a PDF file
python src/main.py CE -i document.pdf -o translation.pdf

# Auto-save with timestamp in the same directory as input
python src/main.py CE -i document.pdf --auto-save
```

### Advanced Options
```bash
# Translate specific pages with abstract context
python src/main.py CE -i document.pdf -p 5-10 -a

# Translate single page and save to PDF
python src/main.py JE -i document.pdf -p 3 -o page3_translation.pdf
```

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

2. Set up your AI Sandbox API key (choose one method):

   **Method 1: Environment Variable**
   ```bash
   export AI_SANDBOX_KEY="your_ai_sandbox_key_here"
   ```

   **Method 2: .env file (recommended)**
   Copy the example file and add your AI Sandbox API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and replace `your_actual_ai_sandbox_key_here` with your actual AI Sandbox API key:
   ```env
   AI_SANDBOX_KEY=your_actual_ai_sandbox_key_here
   ```
   
   Note: The `.env` file should not be committed to version control for security reasons.
   
   **Getting your AI Sandbox API key:**
   Princeton University faculty can obtain their AI Sandbox API key through OIT.

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
