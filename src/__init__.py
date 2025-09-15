"""
CJK Translation Package

This package provides tools for translating documents and custom text
between Chinese, Japanese, Korean, and English using the Princeton AI Sandbox.

Supported input formats:
- PDF documents (.pdf)
- Word documents (.docx)
- Text files (.txt)
- Custom text input

Supported output formats:
- Plain text (.txt)
- PDF documents (.pdf)
- Word documents (.docx)

Features:
- Professor-specific API key management
- Token usage tracking and cost reporting
- Page range selection for all document types
- Progressive saving for error recovery
- Custom font support for CJK text
- Multi-format document processing
"""

from .cli import main, CJKTranslator
from .translation_service import TranslationService
from .file_output import FileOutputHandler
from .pdf_processor import PDFProcessor
from .docx_processor import DocxProcessor
from .txt_processor import TxtProcessor
from .base_text_processor import BaseTextProcessor
from .token_tracker import TokenTracker
from .utils import (
    parse_language_code,
    load_professor_config,
    get_api_key,
    make_safe_filename,
    validate_page_nums
)

__version__ = "3.0.0"
__author__ = "Jeff Heller"
__email__ = "jsheller@princeton.edu"

__all__ = [
    # Main entry points
    "main",
    "CJKTranslator",
    
    # Core services
    "TranslationService",
    "TokenTracker",
    
    # File processing
    "FileOutputHandler",
    "PDFProcessor",
    "DocxProcessor", 
    "TxtProcessor",
    "BaseTextProcessor",
    
    # Utilities
    "parse_language_code",
    "load_professor_config",
    "get_api_key",
    "make_safe_filename",
    "validate_page_nums"
]
