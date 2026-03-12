#!/usr/bin/env python3
"""
PU AI Sandbox

This script extracts text from a PDF file and translates it between different languages using the GPT engine. 
It supports translation between Chinese, Japanese, Korean, and English in any direction.
This script is for Princeton University use only and will only function using a valid API key to the AI Sandbox.

Usage:
    python main.py [professor_name] [command] [options]
    
Examples:
    python main.py heller translate CE -i document.pdf              # Translate PDF from Chinese to English
    python main.py smith translate JK -c                            # Translate custom text from Japanese to Korean
    python main.py heller translate CE -i document.pdf -o translation.txt  # Save translation to file
    python main.py heller transcribe E -i image.jpg -o extracted.txt -m gpt-4o-mini  # OCR image to English text using specified model
    python main.py smith translate CE -i page2.jpg                  # Translate text from image file (using OCR) from Chinese to English
    python main.py heller usage report                              # Show usage report for professor's account
    python main.py --list-models                                    # List all available models (no professor needed)
    
Professor Configuration:
    Professor names and API keys are configured in .env
    Each professor has separate usage tracking and monthly budgets
"""

if __name__ == '__main__':
    from src.cli import main
    main()
