"""
PDF processing utilities for the CJK Translation script.
"""

from typing import Iterator, BinaryIO, Optional, Tuple
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTFigure, LTChar, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
import re


def extract_page_nums(page_nums_str: Optional[str]) -> Tuple[int, int]:
    """Extract the start and end page numbers from the given string."""
    if page_nums_str is None:
        return 0, 0  # Process all pages
    
    if '-' in page_nums_str:
        start_page, end_page = map(int, page_nums_str.split('-'))
        return start_page - 1, end_page - 1
    else:
        page_num = int(page_nums_str)
        if page_num <= 0:
            raise ValueError(f"{page_nums_str} is not a valid page number.")
        return page_num - 1, page_num - 1


def generate_process_text(abstract_text: str, page_text: str, previous_page: str, context_percentage: float = 0.65) -> str:
    """Generate text for processing with context."""
    context = abstract_text if abstract_text else previous_page[int(len(previous_page) * context_percentage):]
    if context:
        context = f"--Context: \n{context}"
    return f"--Current Page: \n{page_text}\n{context}"


class PDFProcessor:
    """Handles PDF processing operations."""
    
    def __init__(self):
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
        """Clean extracted text by removing problematic characters and CID references."""
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
        """Process a PDF file and return pages iterator."""
        return PDFPage.get_pages(file_handle)
    
    def parse_layout(self, layout: LTPage) -> str:
        """Parse the layout tree and extract text."""
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
        """Process a single page and extract text."""
        self.interpreter.process_page(page)
        layout = self.device.get_result()
        return self.parse_layout(layout)
