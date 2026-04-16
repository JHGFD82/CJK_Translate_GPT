"""Shared constants for all processor modules."""

from ..settings import DEFAULT_PAGE_SIZE

__all__ = ["DEFAULT_PAGE_SIZE"]

# Supported image file extensions for OCR and vision processing
IMAGE_EXTENSIONS: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
