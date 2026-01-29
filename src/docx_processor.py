"""
Word document processing utilities for the CJK Translation script.
"""

import logging
from typing import List, BinaryIO
import os

from .base_text_processor import BaseTextProcessor


class DocxProcessor(BaseTextProcessor):
    """Handles extraction of text from Word documents."""
    
    def extract_raw_content(self, file_obj: BinaryIO) -> str:
        """Extract raw text content from a Word document."""
        try:
            from docx import Document
            
            # Load the document
            doc = Document(file_obj)
            
            # Extract text from all paragraphs
            paragraphs: List[str] = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:  # Only add non-empty paragraphs
                    paragraphs.append(text)
            
            # Join paragraphs with double newlines
            return '\n\n'.join(paragraphs) if paragraphs else ""
                
        except ImportError:
            raise ImportError(
                "python-docx is required to process Word documents. "
                "Install it with: pip install python-docx"
            )
    
    def get_file_type_name(self) -> str:
        """Get a human-readable name for the file type this processor handles."""
        return "Word document"
    """Handles extraction of text from Word documents."""
    
    @staticmethod
    def process_docx(file_obj: BinaryIO) -> List[str]:
        """
        Extract text from a Word document and return as list of pages.
        
        Args:
            file_obj: Binary file object of the Word document
            
        Returns:
            List of strings, each representing a "page" of content
        """
        try:
            processor = DocxProcessor()
            content = processor.extract_raw_content(file_obj)
            
            if not content:
                logging.warning("No text content found in Word document")
                return [""]
            
            # For Word documents, treat the entire document as one "page"
            return [content]
                
        except Exception as e:
            logging.error(f"Error processing Word document: {e}")
            raise Exception(f"Failed to process Word document: {e}")
    
    @staticmethod
    def process_docx_with_pages(file_obj: BinaryIO, target_page_size: int = 2000) -> List[str]:
        """
        Extract text from a Word document and split into logical pages based on content size.
        
        Args:
            file_obj: Binary file object of the Word document
            target_page_size: Target number of characters per "page"
            
        Returns:
            List of strings, each representing a logical "page" of content
        """
        try:
            processor = DocxProcessor()
            content = processor.extract_raw_content(file_obj)
            
            if not content:
                logging.warning("No text content found in Word document")
                return [""]
            
            # Parse content into paragraphs and split into pages
            paragraphs = processor.parse_text_into_paragraphs(content)
            pages = processor.split_text_into_pages(paragraphs, target_page_size)
            
            logging.info(f"Split Word document into {len(pages)} logical pages")
            return pages
                
        except Exception as e:
            logging.error(f"Error processing Word document: {e}")
            raise Exception(f"Failed to process Word document: {e}")
    
    @staticmethod
    def is_docx_file(file_path: str) -> bool:
        """Check if a file is a .docx Word document based on its extension.
        Note: .doc files (older Word format) are not supported."""
        return file_path.lower().endswith('.docx')
    
    @staticmethod
    def validate_docx_file(file_path: str) -> bool:
        """Validate that a file is a readable Word document."""
        if not DocxProcessor.is_docx_file(file_path):
            return False
        
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'rb') as f:
                DocxProcessor.process_docx(f)
            return True
        except Exception:
            return False
