# CJK Translate GPT - Token Usage Tracking

## Overview

The CJK Translation script includes comprehensive token usage tracking to help you monitor your API usage and costs. This feature tracks token consumption across all API calls and provides detailed reports on usage patterns and costs.

**Important:** The script now requires a `pricing_config.json` file to function. This file serves as the single source of truth for all model pricing and configuration data.

## Features

### 1. Automatic Token Tracking
- All API calls are automatically tracked
- Records input tokens, output tokens, and total tokens
- Calculates costs based on current pricing from `pricing_config.json`
- Stores usage data persistently in `token_usage.json`

### 2. Centralized Configuration
- All model pricing stored in `pricing_config.json`
- Script will not run without valid pricing configuration
- Automatic model discovery from pricing configuration
- Consistent pricing across all modules

### 3. Usage Reports
- View comprehensive usage reports
- See breakdowns by model, date, and session
- Track costs for better budget management

### 4. Pricing Management
- Update pricing information via command line
- Easily update rates when Princeton changes them
- Support for multiple models with fallback handling

## Usage Commands

### View Overall Usage Report
```bash
python main.py --usage-report
```

### View Daily Usage
```bash
# Today's usage
python main.py --daily-usage

# Specific date usage
python main.py --daily-usage 2025-07-11
```

### Update Model Pricing
```bash
python main.py --update-pricing gpt-4o 2.75 11.00
```

## Current Pricing (as of July 2025)

Based on the Princeton KB article: https://princeton.service-now.com/service?id=kb_article&sys_id=KB0014337

| Model Name | Input (Per 1M tokens) | Output (Per 1M tokens) |
|------------|---------------------|----------------------|
| o3-mini | $1.21 | $4.84 |
| gpt-4o-mini | $0.165 | $0.66 |
| gpt-4o | $2.75 | $11.00 |
| gpt-35-turbo-16k | $3.00 | $4.00 |
| Mistral-Small | $1.00 | $3.00 |
| Meta-Llama-3-1-8B-Instruct | $3.000 | $0.61 |
| Meta-Llama-3-1-70B-Instruct | $2.68 | $3.54 |

## File Structure

### Required Files:
- `src/pricing_config.json` - **REQUIRED** - Central pricing configuration and model definitions
- `src/token_tracker.py` - Core token tracking functionality
- `src/config.py` - Centralized configuration management

### Auto-Generated Files:
- `token_usage.json` - Usage data (created automatically)

### Updated Files:
- `src/translation_service.py` - Integrated token tracking
- `src/cli.py` - Added usage commands
- `src/pdf_processor.py` - Added PDF processing utilities
- `src/file_output.py` - Added file naming utilities

## Data Storage

### pricing_config.json (Required)
This file is **required** for the script to function and contains:
- Model pricing information (input/output costs per 1M tokens)
- Configuration settings (pricing unit, monthly limits)
- Available model definitions

**The script will fail to start without this file.**

Example structure:
```json
{
  "config": {
    "pricing_unit": 1000000,
    "monthly_limit": 250.0
  },
  "models": {
    "gpt-4o": {
      "input": 2.75,
      "output": 11.0
    },
    "gpt-4o-mini": {
      "input": 0.165,
      "output": 0.66
    }
  }
}
```

### token_usage.json (Auto-generated)
This file stores all usage data and includes:
- Total usage across all sessions
- Usage breakdown by model
- Daily usage statistics
- Individual session history

**This file is created automatically and should not be manually edited.**

## Example Usage Workflow

### First Time Setup:
1. **Ensure `src/pricing_config.json` exists** (required):
   ```json
   {
     "config": {
       "pricing_unit": 1000000,
       "monthly_limit": 250.0
     },
     "models": {
       "gpt-4o": {"input": 2.75, "output": 11.0},
       "gpt-4o-mini": {"input": 0.165, "output": 0.66}
     }
   }
   ```

### Normal Usage:
1. **Translate a document:**
   ```bash
   python main.py CE -i document.pdf -o translation.txt
   ```

2. **Check usage after translation:**
   ```bash
   python main.py --usage-report
   ```

3. **Update pricing when Princeton changes rates:**
   ```bash
   python main.py --update-pricing gpt-4o 3.00 12.00
   ```

4. **Check daily usage:**
   ```bash
   python main.py --daily-usage
   ```

## Sample Output

### Usage Report
```
============================================================
TOKEN USAGE REPORT
============================================================
Total Tokens Used: 15,450
  • Input Tokens: 10,200
  • Output Tokens: 5,250
Total Cost: $0.5125

Model Breakdown:
----------------------------------------
gpt-4o:
  • Calls: 3
  • Tokens: 12,000
  • Cost: $0.4620
gpt-4o-mini:
  • Calls: 2
  • Tokens: 3,450
  • Cost: $0.0505

Today's Usage (2025-07-11):
----------------------------------------
Tokens: 3,450
Cost: $0.0505
============================================================
```

## Important Notes

1. **pricing_config.json is required** - The script will not run without this file
2. **Token tracking is automatic** - No need to manually track usage
3. **Pricing updates persist** - Updated pricing is saved to `pricing_config.json`
4. **Historical data preserved** - All usage history is maintained in `token_usage.json`
5. **Cost calculations** - Automatically calculates costs based on current pricing
6. **Multiple models supported** - Tracks usage across all available models
7. **Centralized configuration** - All model and pricing data managed in one place
8. **Error handling** - Script provides clear error messages for missing/invalid config

## Troubleshooting

If you encounter issues:

### Script won't start:
1. **Check that `src/pricing_config.json` exists** - This file is required
2. **Verify pricing_config.json has valid JSON** - Check for syntax errors
3. **Ensure pricing_config.json has required sections** - Must have "config" and "models" sections
4. **Check that models section is not empty** - At least one model must be defined

### Usage tracking issues:
1. Check that `token_usage.json` has write permissions
2. Verify the model name in pricing matches the actual model used by the API
3. Ensure monthly_limit is set in pricing_config.json

### Common error messages:
- "Pricing configuration file not found" - Create `src/pricing_config.json`
- "Invalid JSON in pricing configuration" - Fix JSON syntax errors
- "Missing required 'models' section" - Add models section to config
- "No models configured" - Add at least one model to the models section
