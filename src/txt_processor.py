"""
Text file processing utilities for the CJK Translation script.
"""

import logging
from typing import List, TextIO
import os

from .base_text_processor import BaseTextProcessor


class TxtProcessor(BaseTextProcessor):
    """Handles extraction of text from plain text files."""
    
    def extract_raw_content(self, file_obj: TextIO) -> str:
        """Extract raw text content from a text file."""
        return file_obj.read().strip()
    
    def get_file_type_name(self) -> str:
        """Get a human-readable name for the file type this processor handles."""
        return "text file"
    
    @staticmethod
    def process_txt(file_obj: TextIO) -> List[str]:
        """
        Extract text from a plain text file and return as list of pages.
        
        Args:
            file_obj: Text file object
            
        Returns:
            List of strings, each representing a "page" of content
        """
        try:
            processor = TxtProcessor()
            content = processor.extract_raw_content(file_obj)
            
            if not content:
                logging.warning("No text content found in text file")
                return [""]

            # For simple text files, treat the entire content as one "page"
            return [content]
                
        except Exception as e:
            logging.error(f"Error processing text file: {e}")
            raise Exception(f"Failed to process text file: {e}")
    
    @staticmethod
    def process_txt_with_pages(file_obj: TextIO, target_page_size: int = 2000) -> List[str]:
        """
        Extract text from a plain text file and split into logical pages based on content size.
        
        Args:
            file_obj: Text file object
            target_page_size: Target number of characters per "page"
            
        Returns:
            List of strings, each representing a logical "page" of content
        """
        try:
            processor = TxtProcessor()
            content = processor.extract_raw_content(file_obj)
            
            if not content:
                logging.warning("No text content found in text file")
                return [""]
            
            # Parse content into paragraphs and split into pages
            paragraphs = processor.parse_text_into_paragraphs(content)
            pages = processor.split_text_into_pages(paragraphs, target_page_size)
            
            logging.info(f"Split text file into {len(pages)} logical pages")
            return pages
                
        except Exception as e:
            logging.error(f"Error processing text file: {e}")
            raise Exception(f"Failed to process text file: {e}")
    
    @staticmethod
    def is_txt_file(file_path: str) -> bool:
        """Check if a file is a text file based on its extension."""
        return file_path.lower().endswith('.txt')
    
    @staticmethod
    def validate_txt_file(file_path: str) -> bool:
        """Validate that a file is a readable text file."""
        if not TxtProcessor.is_txt_file(file_path):
            return False
        
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                TxtProcessor.process_txt(f)
            return True
        except Exception:
            return False
