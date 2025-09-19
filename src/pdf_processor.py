"""
PDF processing utilities for the CJK Translation script.

This module provides functionality to extract and process text from PDF files,
including support for CJK (Chinese, Japanese, Korean) text extraction.
"""

from typing import Iterator, BinaryIO
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTFigure, LTChar, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
import re


def generate_process_text(abstract_text: str, page_text: str, previous_page: str, context_percentage: float = 0.65) -> str:
    """Generate text for processing with context."""
    context = abstract_text if abstract_text else previous_page[int(len(previous_page) * context_percentage):]
    if context:
        context = f"--Context: \n{context}"
    return f"--Current Page: \n{page_text}\n{context}"


class PDFProcessor:
    """
    Handles PDF processing operations, including text extraction and cleaning.

    This class uses pdfminer.six to parse PDF files and extract text content,
    with optimizations for handling CJK text and vertical text layouts.
    """

    def __init__(self):
        """
        Initialize the PDFProcessor with custom layout analysis parameters.

        The parameters are optimized for better handling of CJK text, including
        vertical text detection and improved character grouping.
        """
        self.rsrcmgr = PDFResourceManager()
        # Improved LAParams for better CJK text extraction
        self.laparams = LAParams(
            char_margin=0.5,  # Increase margin to better group characters
            line_margin=0.5,  # Increase line margin
            word_margin=0.1,  # Reduce word margin to avoid breaking CJK characters
            detect_vertical=True,  # Enable vertical text detection for CJK
            all_texts=False,  # Only extract text, not non-text elements
            boxes_flow=None,  # Use None for better CJK handling
        )
        self.device = PDFPageAggregator(self.rsrcmgr, laparams=self.laparams)
        self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing problematic characters and formatting issues.

        Args:
            text: The raw text extracted from the PDF layout.

        Returns:
            A cleaned version of the text with unwanted characters removed.
        """
        if not text:
            return ""
        
        # Remove null characters and other control characters
        cleaned_text = text.replace('\x00', '').replace('\ufeff', '')
        
        # Remove CID references like (cid:123) which appear when character mapping fails
        cleaned_text = re.sub(r'\(cid:\d+\)', '', cleaned_text)
        
        # Remove excessive whitespace but preserve line breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def process_pdf(self, file_handle: BinaryIO) -> Iterator[PDFPage]:
        """
        Process a PDF file and return an iterator over its pages.

        Args:
            file_handle: A binary file object representing the PDF file.

        Returns:
            An iterator over PDFPage objects.
        """
        return PDFPage.get_pages(file_handle)
    
    def parse_layout(self, layout: LTPage) -> str:
        """
        Parse the layout tree of a PDF page and extract text content.

        Args:
            layout: The LTPage object representing the layout of a PDF page.

        Returns:
            A string containing the extracted and cleaned text from the page.
        """
        result: list[str] = []
        stack = list(layout)  # Using a list as a stack

        while stack:
            lt_obj = stack.pop(0)
            if isinstance(lt_obj, LTTextLine):
                text = lt_obj.get_text()
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    result.append(cleaned_text)
            elif isinstance(lt_obj, (LTChar, LTTextContainer)):
                text = lt_obj.get_text()
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    result.append(cleaned_text)
            elif isinstance(lt_obj, (LTFigure, LTTextBox)):
                stack.extend(list(lt_obj))  # Add children to the stack

        # Join with line breaks to preserve document structure
        final_text = '\n'.join(result)
        # Clean up excessive line breaks but preserve paragraph structure
        final_text = re.sub(r'\n\s*\n', '\n\n', final_text)  # Preserve paragraph breaks
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)  # Remove excessive line breaks
        return final_text.strip()
    
    def process_page(self, page: PDFPage) -> str:
        """
        Process a single PDF page and extract its text content.

        Args:
            page: The PDFPage object representing the page to process.

        Returns:
            A string containing the extracted and cleaned text from the page.
        """
        self.interpreter.process_page(page)
        layout = self.device.get_result()
        return self.parse_layout(layout)
