#!/usr/bin/env python3
"""
CJK Translation Script

This script extracts text from a PDF file and translates it between different languages using the GPT engine. 
It supports translation between Chinese, Japanese, Korean, and English in any direction.
This script is for Princeton University use only and will only function using a valid API key to the AI Sandbox.

Usage:
    python main.py CE -i document.pdf
    python main.py JK -c
    python main.py CE -i document.pdf -o translation.txt
"""

if __name__ == '__main__':
    from src.cli import main
    main()
