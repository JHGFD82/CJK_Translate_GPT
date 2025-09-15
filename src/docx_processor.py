"""
Word document processing utilities for the CJK Translation script.
"""

import logging
from typing import List, BinaryIO
import os


class DocxProcessor:
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
            from docx import Document
            
            # Load the document
            doc = Document(file_obj)
            
            # Extract text from all paragraphs
            paragraphs: List[str] = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:  # Only add non-empty paragraphs
                    paragraphs.append(text)
            
            # Combine paragraphs into pages
            # For Word documents, we'll treat the entire document as one "page"
            # since there's no clear page boundary concept like in PDFs
            if paragraphs:
                full_text = '\n\n'.join(paragraphs)
                return [full_text]
            else:
                logging.warning("No text content found in Word document")
                return [""]
                
        except ImportError:
            raise ImportError(
                "python-docx is required to process Word documents. "
                "Install it with: pip install python-docx"
            )
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
            from docx import Document
            
            # Load the document
            doc = Document(file_obj)
            
            # Extract text from all paragraphs
            paragraphs: List[str] = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:  # Only add non-empty paragraphs
                    paragraphs.append(text)
            
            if not paragraphs:
                logging.warning("No text content found in Word document")
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
            
            logging.info(f"Split Word document into {len(pages)} logical pages")
            return pages
                
        except ImportError:
            raise ImportError(
                "python-docx is required to process Word documents. "
                "Install it with: pip install python-docx"
            )
        except Exception as e:
            logging.error(f"Error processing Word document: {e}")
            raise Exception(f"Failed to process Word document: {e}")
    
    @staticmethod
    def is_docx_file(file_path: str) -> bool:
        """Check if a file is a Word document based on its extension."""
        return file_path.lower().endswith(('.docx', '.doc'))
    
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
