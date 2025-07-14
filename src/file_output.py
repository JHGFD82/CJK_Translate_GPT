"""
File output utilities for the CJK Translation script.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import PDF_MARGINS


def generate_output_filename(input_file: str, source_lang: str, target_lang: str, extension: str = '.txt') -> str:
    """Generate an output filename based on input file and languages."""
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_path.stem}_{source_lang}to{target_lang}_{timestamp}{extension}"
    return str(input_path.parent / output_name)


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
    def save_to_pdf(content: str, output_path: str, custom_font: Optional[str] = None, target_lang: Optional[str] = None) -> None:
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
            
            # Configure font - use Times-Roman for English unless custom font specified
            if not custom_font and target_lang == 'English':
                font_name = 'Times-Roman'
                logging.info(f"Using Times-Roman for English translation (target_lang={target_lang})")
            else:
                font_name = FileOutputHandler._get_cjk_font(custom_font)
                logging.info(f"Using CJK font logic (custom_font={custom_font}, target_lang={target_lang})")
            
            try:
                normal_style = ParagraphStyle(
                    'CJKNormal',
                    parent=styles['Normal'],
                    fontName=font_name,
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
                        # For CJK characters, we need to handle the text more carefully
                        # Replace line breaks within paragraphs with spaces
                        clean_text = clean_text.replace('\n', ' ')
                        p = Paragraph(clean_text, normal_style)
                        story.append(p)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        # More graceful fallback for problematic characters
                        logging.warning(f"Error processing paragraph, trying fallback: {e}")
                        try:
                            # Try with a simpler style
                            simple_style = ParagraphStyle(
                                'SimpleCJK',
                                parent=styles['Normal'],
                                fontName='Helvetica',
                                fontSize=12,
                                leading=18,
                                spaceAfter=12
                            )
                            p = Paragraph(clean_text, simple_style)
                            story.append(p)
                            story.append(Spacer(1, 12))
                        except Exception:
                            # Ultimate fallback: just add as plain text
                            logging.warning("Using ultimate fallback for problematic text")
                            p = Paragraph(clean_text.encode('ascii', 'ignore').decode('ascii'), styles['Normal'])
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
            print(f"Error generating PDF: {e}")
            print("Falling back to text file for reliable CJK character support...")
            text_output_path = output_path.replace('.pdf', '.txt')
            FileOutputHandler.save_to_text_file(content, text_output_path)
    
    @staticmethod
    def save_translation_output(content: str, input_file: Optional[str], output_file: Optional[str], 
                              auto_save: bool, source_lang: str, target_lang: str, custom_font: Optional[str] = None) -> None:
        """Save translation output to file based on user preferences."""
        if not content.strip():
            print("No content to save.")
            return
        
        # Determine output file path
        if output_file:
            output_path = output_file
        elif auto_save and input_file:
            output_path = generate_output_filename(input_file, source_lang, target_lang, '.txt')
        else:
            # No saving requested
            return
        
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file type and save accordingly
        if output_path.lower().endswith('.pdf'):
            FileOutputHandler.save_to_pdf(content, output_path, custom_font, target_lang)
        else:
            # Default to text file
            if not output_path.lower().endswith('.txt'):
                output_path += '.txt'
            FileOutputHandler.save_to_text_file(content, output_path)
    
    @staticmethod
    def _get_cjk_font(custom_font: Optional[str] = None) -> str:
        """Get an appropriate font for CJK characters from the fonts directory."""
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import os
            
            # Check fonts from the local fonts directory
            fonts_dir = os.path.join(os.path.dirname(__file__), '..', 'fonts')
            if os.path.exists(fonts_dir):
                # If a custom font is specified, try it first
                if custom_font:
                    custom_font_path = os.path.join(fonts_dir, f"{custom_font}.ttf")
                    if os.path.exists(custom_font_path):
                        try:
                            if custom_font not in pdfmetrics.getRegisteredFontNames():
                                pdfmetrics.registerFont(TTFont(custom_font, custom_font_path))  # type: ignore
                            logging.info(f"Using custom CJK font: {custom_font}")
                            return custom_font
                        except Exception as e:
                            logging.warning(f"Failed to register custom font {custom_font}: {e}")
                            print(f"Warning: Custom font '{custom_font}' failed to load. Using default font selection.")
                    else:
                        logging.warning(f"Custom font file not found: {custom_font_path}")
                        print(f"Warning: Custom font '{custom_font}.ttf' not found in fonts/ directory. Using default font selection.")
                
                # Preferred CJK fonts (in order of preference) - map to available fonts
                preferred_fonts = [
                    ('Arial Unicode.ttf', 'ArialUnicode'),         # Your available Arial Unicode
                    ('AppleGothic.ttf', 'AppleGothic'),           # Apple Gothic (good for CJK)
                    ('AppleMyungjo.ttf', 'AppleMyungjo'),         # Apple Myungjo (good for CJK)
                ]
                
                # First, try preferred fonts
                for preferred_font in preferred_fonts:
                    font_path = os.path.join(fonts_dir, preferred_font)
                    if os.path.exists(font_path):
                        font_name = os.path.splitext(preferred_font)[0]
                        try:
                            if font_name not in pdfmetrics.getRegisteredFontNames():
                                pdfmetrics.registerFont(TTFont(font_name, font_path))  # type: ignore
                            logging.info(f"Using preferred CJK font: {font_name}")
                            return font_name
                        except Exception as e:
                            logging.debug(f"Failed to register preferred font {font_name}: {e}")
                            continue
                
                # If no preferred fonts, use any available .ttf file (reportlab only supports TTF, not OTF)
                for font_file in os.listdir(fonts_dir):
                    if font_file.endswith('.ttf'):
                        font_path = os.path.join(fonts_dir, font_file)
                        font_name = os.path.splitext(font_file)[0]  # Use filename without extension
                        try:
                            if font_name not in pdfmetrics.getRegisteredFontNames():
                                pdfmetrics.registerFont(TTFont(font_name, font_path))  # type: ignore
                            logging.info(f"Using available CJK font: {font_name}")
                            return font_name
                        except Exception as e:
                            logging.debug(f"Failed to register font {font_name}: {e}")
                            continue
            
            # No compatible fonts found - give clear guidance
            logging.warning("No CJK fonts found in fonts/ directory.")
            print("Warning: No CJK fonts available for PDF generation.")
            print("To fix: Add CJK .ttf fonts to the 'fonts/' directory in this project.")
            print("Note: Only .ttf fonts are supported. OTF fonts will not work with reportlab.")
            print("Recommended CJK fonts:")
            print("  - Arial Unicode MS (Microsoft)")
            print("  - Source Han Sans (Adobe): https://github.com/adobe-fonts/source-han-sans")
            print("  - Apple system fonts (AppleGothic, AppleMyungjo)")
            print("Alternative: Save as .txt file for proper CJK character display.")
            
            return 'Times-Roman'  # Fallback to reportlab default (Times New Roman equivalent)
            
        except Exception as e:
            logging.debug(f"Error checking CJK fonts: {e}")
            return 'Times-Roman'
