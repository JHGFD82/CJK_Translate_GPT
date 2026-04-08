"""Shared constants for all processor modules."""

# Default target number of characters per logical "page" when splitting documents
DEFAULT_PAGE_SIZE: int = 2000

# Supported image file extensions for OCR and vision processing
IMAGE_EXTENSIONS: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
