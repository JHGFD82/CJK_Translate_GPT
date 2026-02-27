#!/usr/bin/env python3
"""
CJK Translation Script

This script extracts text from a PDF file and translates it between different languages using the GPT engine. 
It supports translation between Chinese, Japanese, Korean, and English in any direction.
This script is for Princeton University use only and will only function using a valid API key to the AI Sandbox.

Usage:
    python main.py [professor_name] [command] [options]
    
Examples:
    python main.py heller translate CE -i document.pdf              # Translate PDF from Chinese to English
    python main.py smith translate JK -c                            # Translate custom text from Japanese to Korean
    python main.py heller translate CE -i document.pdf -o translation.txt  # Save translation to file
    python main.py heller usage report                              # Show usage report for professor's account
    python main.py --list-models                                    # List all available models (no professor needed)
    
Professor Configuration:
    Professor names and API keys are configured in .env file using IDs (1, 2, 3, etc.)
    Each professor has separate usage tracking and monthly budgets
"""

if __name__ == '__main__':
    from src.cli import main
    main()
