# CJK Translation GPT - AI Coding Assistant Instructions

## Project Overview
Academic translation tool for Princeton University faculty to translate between Chinese, Japanese, Korean, and English using Azure OpenAI API. Features professor-specific API key management, per-professor token tracking, and privacy-friendly configuration.

## Architecture Pattern
**Multi-Professor Service Architecture**: Each professor has isolated API keys and token tracking through environment-based configuration.

- **Entry Point**: `main.py` → `src/cli.py` → `CJKTranslator` class
- **Core Services**: `TranslationService` (Azure OpenAI), `TokenTracker` (usage tracking), `PDFProcessor` (text extraction), `FileOutputHandler` (output formatting)
- **Configuration**: Environment variables (`.env`) for professor configs, `pricing_config.json` for model pricing

## Professor Configuration System
The system uses a specific environment variable pattern:
```bash
PROF_[ID]_NAME=professor_name     # Display name
PROF_[ID]_KEY=primary_api_key     # Primary Azure OpenAI key
PROF_[ID]_BACKUP_KEY=backup_key   # Fallback key
```

**Safe Name Conversion**: Professor names are converted to safe filenames using `make_safe_filename()` - spaces become underscores, special chars removed. This affects:
- Token usage files: `data/token_usage_{safe_name}.json`
- CLI argument parsing and validation
- Error message formatting

## Token Tracking Architecture
**Per-Professor Isolation**: Each professor gets separate token usage tracking:
- Files: `data/token_usage_{safe_name}.json`
- Structure: `{total_usage, model_usage, daily_usage, session_history}`
- Pricing: Loaded from `src/pricing_config.json` with configurable units (default: per 1M tokens)
- Budget tracking: Monthly limits with percentage warnings

**Migration Pattern**: Legacy `token_usage.json` is automatically migrated to professor-specific files on first run.

## Key Development Workflows

### Adding New Professors
1. Add to `.env`: `PROF_N_NAME=name`, `PROF_N_KEY=key`, `PROF_N_BACKUP_KEY=backup`
2. Run `python show_config.py` to verify configuration
3. Token tracking files auto-created on first use

### Testing CLI Changes
```bash
# Test basic functionality
python -m src.cli professor_name --help
python -m src.cli professor_name --usage-report

# Test with actual professor (requires .env)
python -m src.cli conlan CE -c  # Custom text translation
```

### Language Code Pattern
Two-character codes: `CE` (Chinese→English), `JK` (Japanese→Korean), etc.
- Parsed in `parse_language_code()` using `LANGUAGE_MAP` from `config.py`
- Validation ensures source ≠ target, valid language codes only

## Critical Implementation Details

### Error Handling Pattern
- **API Failures**: Automatic retries with exponential backoff in `TranslationService`
- **Progressive Saving**: Text-only (no PDF/Word support), saves each page immediately
- **Graceful Degradation**: Failed pages get error messages, processing continues

### File Output Strategy
- **Text Files**: Direct UTF-8 output with proper paragraph breaks
- **PDF Files**: Uses `reportlab` with CJK font support from `fonts/` directory
- **Word Documents**: Uses `python-docx` with CJK font support, professional formatting (1.5 line spacing, proper margins)
- **Auto-save**: Timestamped filenames with language codes, placed in source file directory
- **Absolute Path Resolution**: Input files converted to absolute paths to ensure output placement regardless of execution directory

### PDF Processing Specifics
- **CJK Optimization**: Custom `LAParams` in `PDFProcessor` for better CJK text extraction
- **Context Preservation**: Previous page context (65% of content) passed to translation
- **Page Range Support**: 1-based user input converted to 0-based internally

### Output Format Support
- **Text Files (.txt)**: Direct UTF-8 output, supports progressive saving
- **PDF Files (.pdf)**: Uses `reportlab` with CJK font validation via `_get_cjk_font()`
- **Word Documents (.docx)**: Uses `python-docx` with CJK font selection via `_get_docx_font()`
- **Path Resolution**: Input files converted to absolute paths in CLI for consistent output placement
- **Progressive Save Limitation**: Only text format supports progressive saving; PDF/Word fall back to text

### File Path Handling
- **Input Path Resolution**: `os.path.abspath()` applied to input files in `translate_pdf()` method
- **Output Path Resolution**: User-specified output paths converted to absolute paths in CLI `run()` method
- **Directory Placement**: Output files placed in same directory as source file via `generate_output_filename()`

## External Dependencies
- **Azure OpenAI**: Uses `SANDBOX_ENDPOINT` and `SANDBOX_API_VERSION` from config
- **Font Management**: Custom fonts in `fonts/` directory for PDF and Word output
- **Document Generation**: `reportlab` for PDF, `python-docx` for Word documents
- **Princeton-Specific**: API keys from Princeton's AI Sandbox service

## Common Patterns to Follow
- **Safe Name Usage**: Always use `make_safe_filename()` for file operations
- **Professor Context**: Pass professor name to `TokenTracker` and `TranslationService`
- **Configuration Loading**: Use `load_professor_config()` for env var parsing
- **Error Messages**: Include available professors and suggest both full/safe names
- **Logging**: Use structured logging with professor context where applicable

## Testing Without API Keys
Use `python show_config.py` to validate professor configuration without making API calls.
