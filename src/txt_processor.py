"""
Text file processing utilities for the CJK Translation script.
"""

import logging
from typing import List, TextIO
import os


class TxtProcessor:
    """Handles extraction of text from plain text files."""
    
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
            # Read the entire file content
            content = file_obj.read().strip()
            
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
            # Read the entire file content
            content = file_obj.read().strip()
            
            if not content:
                logging.warning("No text content found in text file")
                return [""]
            
            # Split by double newlines (paragraph breaks) first
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if not paragraphs:
                # If no paragraph breaks, split by single newlines
                paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            
            if not paragraphs:
                # If still no content, return the raw content
                return [content]
            
            # Split into logical pages based on content size
            pages: List[str] = []
            current_page: List[str] = []
            current_size = 0
            
            for paragraph in paragraphs:
                para_size = len(paragraph)
                
                # If adding this paragraph would exceed target size and we have content, start new page
                if current_size + para_size > target_page_size and current_page:
                    pages.append('\n\n'.join(current_page))
                    current_page = [paragraph]
                    current_size = para_size
                else:
                    current_page.append(paragraph)
                    current_size += para_size + 2  # +2 for the '\n\n' separator
            
            # Add the last page if it has content
            if current_page:
                pages.append('\n\n'.join(current_page))
            
            # If no pages were created (all paragraphs were very small), create one page
            if not pages:
                pages = ['\n\n'.join(paragraphs)]
            
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
