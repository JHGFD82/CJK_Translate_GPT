"""
PDF processing utilities for the CJK Translation script.
"""

from typing import Iterator, BinaryIO
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTFigure, LTChar, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


class PDFProcessor:
    """Handles PDF processing operations."""
    
    def __init__(self):
        self.rsrcmgr = PDFResourceManager()
        self.laparams = LAParams()
        self.device = PDFPageAggregator(self.rsrcmgr, laparams=self.laparams)
        self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
    
    def process_pdf(self, file_handle: BinaryIO) -> Iterator[PDFPage]:
        """Process a PDF file and return pages iterator."""
        return PDFPage.get_pages(file_handle)
    
    def parse_layout(self, layout: LTPage) -> str:
        """Parse the layout tree and extract text."""
        result: list[str] = []
        stack = list(layout)  # Using a list as a stack

        while stack:
            lt_obj = stack.pop(0)
            if isinstance(lt_obj, (LTTextLine, LTChar, LTTextContainer)):
                result.append(lt_obj.get_text())
            elif isinstance(lt_obj, (LTFigure, LTTextBox)):
                stack.extend(list(lt_obj))  # Add children to the stack

        return "".join(result)
    
    def process_page(self, page: PDFPage) -> str:
        """Process a single page and extract text."""
        self.interpreter.process_page(page)
        layout = self.device.get_result()
        return self.parse_layout(layout)
