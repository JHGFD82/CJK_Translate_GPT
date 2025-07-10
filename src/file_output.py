"""
File output utilities for the CJK Translation script.
"""

import logging
from pathlib import Path
from typing import Optional

from .config import PDF_MARGINS


class FileOutputHandler:
    """Handles saving translations to various file formats."""
    
    @staticmethod
    def save_to_text_file(content: str, output_path: str) -> None:
        """Save content to a text file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f'Translation saved to text file: {output_path}')
            print(f"\nTranslation saved to: {output_path}")
        except Exception as e:
            logging.error(f'Error saving to text file: {e}')
            print(f"Error saving to text file: {e}")
    
    @staticmethod
    def save_to_pdf(content: str, output_path: str) -> None:
        """Save content to a PDF file using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path, 
                pagesize=letter,
                rightMargin=PDF_MARGINS['right'],
                leftMargin=PDF_MARGINS['left'],
                topMargin=PDF_MARGINS['top'],
                bottomMargin=PDF_MARGINS['bottom']
            )
            
            # Create story (content container)
            story: list[Flowable] = []
            styles = getSampleStyleSheet()
            
            # Configure style for CJK characters
            try:
                normal_style = ParagraphStyle(
                    'Normal',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=18,  # 1.5 leading (12pt * 1.5 = 18pt)
                    spaceAfter=12,
                    encoding='utf-8'
                )
            except:
                normal_style = styles['Normal']
            
            # Split content into paragraphs and add to story
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    clean_text = para.strip()
                    try:
                        p = Paragraph(clean_text, normal_style)
                        story.append(p)
                        story.append(Spacer(1, 12))
                    except:
                        # Fallback for problematic characters
                        clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
                        p = Paragraph(clean_text, normal_style)
                        story.append(p)
                        story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            logging.info(f'Translation saved to PDF file: {output_path}')
            print(f"\nTranslation saved to PDF: {output_path}")
            
        except ImportError:
            logging.warning('reportlab not installed. Falling back to text file.')
            print("Warning: reportlab not installed. Saving as text file instead.")
            text_output_path = output_path.replace('.pdf', '.txt')
            FileOutputHandler.save_to_text_file(content, text_output_path)
        except Exception as e:
            logging.error(f'Error saving to PDF: {e}')
            print(f"Error saving to PDF: {e}")
            print("Falling back to text file...")
            text_output_path = output_path.replace('.pdf', '.txt')
            FileOutputHandler.save_to_text_file(content, text_output_path)
    
    @staticmethod
    def save_translation_output(content: str, input_file: Optional[str], output_file: Optional[str], 
                              auto_save: bool, source_lang: str, target_lang: str) -> None:
        """Save translation output to file based on user preferences."""
        if not content.strip():
            print("No content to save.")
            return
        
        # Determine output file path
        if output_file:
            output_path = output_file
        elif auto_save and input_file:
            from .utils import generate_output_filename
            output_path = generate_output_filename(input_file, source_lang, target_lang, '.txt')
        else:
            # No saving requested
            return
        
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file type and save accordingly
        if output_path.lower().endswith('.pdf'):
            FileOutputHandler.save_to_pdf(content, output_path)
        else:
            # Default to text file
            if not output_path.lower().endswith('.txt'):
                output_path += '.txt'
            FileOutputHandler.save_to_text_file(content, output_path)
