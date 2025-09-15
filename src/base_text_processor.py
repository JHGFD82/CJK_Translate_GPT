"""
Base text processing utilities for the CJK Translation script.
Contains shared functionality for processing text-based documents.
"""

import logging
from typing import List, Any
from abc import ABC, abstractmethod


class BaseTextProcessor(ABC):
    """Base class for text-based document processors."""
    
    @staticmethod
    def split_text_into_pages(paragraphs: List[str], target_page_size: int = 2000) -> List[str]:
        """
        Split a list of paragraphs into logical pages based on content size.
        
        Args:
            paragraphs: List of text paragraphs
            target_page_size: Target number of characters per "page"
            
        Returns:
            List of strings, each representing a logical "page" of content
        """
        if not paragraphs:
            logging.warning("No paragraphs provided for page splitting")
            return [""]
        
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
        
        return pages
    
    @staticmethod
    def parse_text_into_paragraphs(content: str) -> List[str]:
        """
        Parse raw text content into a list of paragraphs.
        
        Args:
            content: Raw text content
            
        Returns:
            List of paragraph strings
        """
        if not content.strip():
            return []
        
        # Split by double newlines (paragraph breaks) first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # If no paragraph breaks, split by single newlines
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        if not paragraphs:
            # If still no content, return the raw content as a single paragraph
            return [content.strip()]
        
        return paragraphs
    
    @abstractmethod
    def extract_raw_content(self, file_obj: Any) -> str:
        """
        Extract raw text content from the file object.
        This method must be implemented by subclasses.
        
        Args:
            file_obj: File object (type varies by processor)
            
        Returns:
            Raw text content as a string
        """
        pass
    
    @abstractmethod
    def get_file_type_name(self) -> str:
        """
        Get a human-readable name for the file type this processor handles.
        
        Returns:
            String description of the file type
        """
        pass
