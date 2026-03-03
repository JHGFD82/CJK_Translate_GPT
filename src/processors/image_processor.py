"""
Image file processing utilities for the CJK Translation script.
"""

import os
import base64
from mimetypes import guess_type

from .constants import IMAGE_EXTENSIONS


class ImageProcessor():
    """Handles extraction of text from image files."""

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if a file is an image file based on its extension."""
        return file_path.lower().endswith(IMAGE_EXTENSIONS)

    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """Validate that a file is a valid image file."""
        if not ImageProcessor.is_image_file(file_path):
            return False

        if not os.path.exists(file_path):
            return False

        return True

    # Base 64 encode local image and return text to be included in AI prompt
    def local_image_to_data_url(self, file_path: str):
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(file_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"
