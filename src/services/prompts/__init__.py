"""Prompt schema classes for all services.

Each spec class holds the parameters for a prompt and assembles the final
strings by referencing named constants from the fragment library.  Services
create a spec, then call its methods to obtain (system_prompt, user_prompt).
"""

from .translation import TranslationPromptSpec
from .ocr import OcrPromptSpec
from .image_translation import ImageTranslationPromptSpec
from .transcription_review import TranscriptionReviewPromptSpec

__all__ = [
    "TranslationPromptSpec",
    "OcrPromptSpec",
    "ImageTranslationPromptSpec",
    "TranscriptionReviewPromptSpec",
]
