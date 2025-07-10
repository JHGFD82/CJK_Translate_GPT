"""
Configuration constants for the CJK Translation script.
"""

from typing import Dict, List

# Language mapping
LANGUAGE_MAP: Dict[str, str] = {
    'C': 'Chinese',
    'J': 'Japanese', 
    'K': 'Korean',
    'E': 'English'
}

# Available models
AVAILABLE_MODELS: List[str] = [
    "o3-mini", 
    "gpt-4o-mini", 
    "gpt-4o", 
    "gpt-35-turbo-16k", 
    "Meta-Llama-3-1-70B-Instruct-htzs", 
    "Meta-Llama-3-1-8B-Instruct-nwxcg", 
    "Mistral-small-zgjes"
]

# Default model
DEFAULT_MODEL: str = "gpt-4o"

# API configuration
SANDBOX_API_VERSION: str = "2025-03-01-preview"
SANDBOX_ENDPOINT: str = "https://api-ai-sandbox.princeton.edu/"

# Translation parameters
TRANSLATION_TEMPERATURE: float = 0.5
TRANSLATION_MAX_TOKENS: int = 1000
TRANSLATION_TOP_P: float = 0.5

# PDF margins (in points)
PDF_MARGINS = {
    'left': 72,
    'right': 72,
    'top': 72,
    'bottom': 18
}

# Context percentage for previous page
CONTEXT_PERCENTAGE: float = 0.65
