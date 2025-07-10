"""
CJK Translation Package

This package provides tools for translating PDF documents and custom text
between Chinese, Japanese, Korean, and English using the Princeton AI Sandbox.
"""

from .cli import main
from .translation_service import TranslationService
from .file_output import FileOutputHandler
from .pdf_processor import PDFProcessor

__version__ = "2.0.0"
__author__ = "Jeff Heller"
__email__ = "jsheller@princeton.edu"

__all__ = [
    "main",
    "TranslationService", 
    "FileOutputHandler",
    "PDFProcessor"
]
