# PU AI Sandbox Toolkit - Token Usage Tracking

## Overview

The PU AI Sandbox Toolkit includes comprehensive token usage tracking to help you monitor API usage and costs across all translation and OCR operations. Each professor's usage is tracked in isolation, with data split by calendar month so that the active file stays small and the $250/month Princeton budget limit is always front-and-center.

**Important:** The toolkit requires a `model_catalog.json` file to function. This file is the single source of truth for all model pricing and configuration.

## Features

### 1. Automatic Token Tracking
- All API calls are automatically tracked
- Records input tokens, output tokens, and total tokens per call
- Calculates costs based on current pricing from `model_catalog.json`
- Separate tracking file per professor, per calendar month

### 2. Per-Month File Isolation
- The active file covers the **current month only** — it never grows indefinitely
- At the start of a new month the previous file is automatically archived
- Past months are stored in `data/archives/{professor}/{YYYY-MM}.json`
- All-time totals are computed on demand by aggregating all archive files

### 3. Usage Reports
- Current month report with budget status
- Per-model and per-day breakdowns
- View any archived month's full report
- Optional all-time aggregate across every archived month

### 4. Pricing Management
- Update pricing via command line — changes are saved to `model_catalog.json`
- Easily update rates when Princeton changes them

## Usage Commands

### Current month report + budget status
```bash
python main.py heller usage report
```

### Report for a specific archived month
```bash
python main.py heller usage report 2025-07
```

### Current month + all-time totals
```bash
python main.py heller usage report --all-time
```

### List all archived months
```bash
python main.py heller usage months
```

### Daily usage
```bash
# Today's usage
python main.py heller usage daily

# Specific date
python main.py heller usage daily 2026-02-14
```

### Update model pricing
```bash
python main.py --update-pricing gpt-4o 2.75 11.00
```

## Current Pricing (as of March 2026)

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

### Required Files
- `src/model_catalog.json` — **REQUIRED** — central pricing configuration and model definitions
- `src/tracking/token_tracker.py` — core token tracking functionality
- `src/config.py` — centralized configuration management

### Auto-Generated Files
```
data/
  token_usage_{name}.json          ← active file, current month only
  archives/
    {name}/
      2026-02.json                 ← one file per past month (auto-created)
      2026-03.json
      ...
```

Each JSON file (active or archive) is self-contained for its month:
```json
{
  "month": "2026-03",
  "total_usage": { ... },
  "model_usage":  { ... },
  "daily_usage":  { ... },
  "session_history": [ ... ]
}
```

## Data Storage

### model_catalog.json (Required)
Contains model pricing and global config. **The toolkit will not run without this file.**

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

### token_usage_{name}.json (Auto-generated)
Stores the **current month's** usage data only. Automatically archived and reset at the start of each new month. Do not manually edit this file.

## Example Usage Workflow

### First Time Setup
1. Ensure `src/model_catalog.json` exists (see structure above).
2. Add professor config to `.env` — usage files are created automatically on first run.

### Normal Usage
1. **Translate a document:**
   ```bash
   python main.py heller translate CE -i document.pdf -o translation.txt
   ```

2. **Check current month usage + budget:**
   ```bash
   python main.py heller usage report
   ```

3. **Review a past month:**
   ```bash
   python main.py heller usage months          # see what's archived
   python main.py heller usage report 2026-02  # pull that month's report
   ```

4. **Update pricing when Princeton changes rates:**
   ```bash
   python main.py --update-pricing gpt-4o 3.00 12.00
   ```

## Sample Output

### Current Month Report
```
============================================================
TOKEN USAGE REPORT - PROFESSOR HELLER
============================================================

Current Month (2026-03):
----------------------------------------
Total Tokens Used: 3,731
  • Input Tokens:  2,716
  • Output Tokens: 1,015
Total Cost: $0.0186

Model Breakdown (this month):
----------------------------------------
gpt-4o-2024-08-06:
  • Calls:  2
  • Tokens: 3,731
  • Cost:   $0.0186

Monthly Budget (2026-03):
----------------------------------------
Monthly Limit: $250.00
Used:          $0.0186 (0.0%)
Remaining:     $249.98
============================================================
```

### Archived Month Report
```
============================================================
TOKEN USAGE REPORT - PROFESSOR HELLER
============================================================

Archived Month (2026-02):
----------------------------------------
Total Tokens Used: 223,722
  • Input Tokens:  150,297
  • Output Tokens: 73,425
Total Cost: $0.9050
API Calls:  34

Model Breakdown:
----------------------------------------
gpt-4o-2024-08-06:
  • Calls:  18
  • Tokens: 40,052
  • Cost:   $0.2449

Daily Breakdown:
----------------------------------------
2026-02-24: 136,905 tokens  $0.2009  (11 calls)
2026-02-27: 83,086 tokens  $0.6854  (21 calls)
============================================================
```

## Important Notes

1. **model_catalog.json is required** — the toolkit will not run without it
2. **Token tracking is automatic** — no manual steps needed
3. **Active file = current month only** — totals never accumulate across months in one file
4. **Month rollover is automatic** — the first run of a new month archives the previous file and starts fresh
5. **All-time totals on demand** — use `usage report --all-time` to aggregate across all archives
6. **Pricing updates persist** — saved to `model_catalog.json`
7. **Per-professor isolation** — each professor has their own active file and archive folder

## Troubleshooting

### Toolkit won't start
1. Check that `src/model_catalog.json` exists — it is required
2. Verify `model_catalog.json` has valid JSON (check for syntax errors)
3. Ensure `model_catalog.json` has both `"config"` and `"models"` sections
4. Confirm the `"models"` section is not empty

### Usage tracking issues
1. Check that `data/` has write permissions
2. Verify the model name in pricing matches the model name returned by the API
3. Ensure `monthly_limit` is set in `model_catalog.json`

### Common error messages
- `"Model catalog file not found"` — create `src/model_catalog.json`
- `"Invalid JSON in model catalog file"` — fix JSON syntax errors
- `"Missing required 'models' section"` — add models section to config
- `"No models configured"` — add at least one model to the models section
- `"No archive found for YYYY-MM"` — that month has no data; run `usage months` to see what's available
