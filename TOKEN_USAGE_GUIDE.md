# CJK Translate GPT - Token Usage Tracking

## Overview

The CJK Translation script now includes comprehensive token usage tracking to help you monitor your API usage and costs. This feature tracks token consumption across all API calls and provides detailed reports on usage patterns and costs.

## Features

### 1. Automatic Token Tracking
- All API calls are automatically tracked
- Records input tokens, output tokens, and total tokens
- Calculates costs based on current pricing
- Stores usage data persistently in `token_usage.json`

### 2. Usage Reports
- View comprehensive usage reports
- See breakdowns by model, date, and session
- Track costs for better budget management

### 3. Pricing Management
- Store and update pricing information
- Easily update rates when Princeton changes them
- Support for multiple models

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

### New Files Added:
- `src/token_tracker.py` - Core token tracking functionality
- `src/pricing_config.json` - Current pricing configuration
- `token_usage.json` - Usage data (created automatically)

### Updated Files:
- `src/translation_service.py` - Integrated token tracking
- `src/cli.py` - Added usage commands

## Data Storage

### token_usage.json
This file stores all usage data and includes:
- Total usage across all sessions
- Usage breakdown by model
- Daily usage statistics
- Individual session history

### pricing_config.json
Contains current pricing for all supported models. Update this file when Princeton changes their pricing.

## Example Usage Workflow

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

1. **Token tracking is automatic** - No need to manually track usage
2. **Pricing updates persist** - Updated pricing is saved for future calculations
3. **Historical data preserved** - All usage history is maintained
4. **Cost calculations** - Automatically calculates costs based on current pricing
5. **Multiple models supported** - Tracks usage across all available models

## Troubleshooting

If you encounter issues:
1. Check that `token_usage.json` has write permissions
2. Verify `pricing_config.json` exists and has valid pricing data
3. Ensure the model name in pricing matches the actual model used by the API
