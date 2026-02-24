"""Font resolution utilities for output generation."""

import logging
from pathlib import Path
from typing import Optional


def _fonts_dir() -> Path:
    """Return the project-level fonts directory."""
    return Path(__file__).resolve().parents[2] / 'fonts'


def get_pdf_font(custom_font: Optional[str] = None) -> str:
    """Get an appropriate font for CJK characters in PDF output."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        fonts_dir = _fonts_dir()
        if fonts_dir.exists():
            if custom_font:
                custom_font_path = fonts_dir / f"{custom_font}.ttf"
                if custom_font_path.exists():
                    try:
                        custom_font_name = f"CustomFont_{custom_font}"
                        if custom_font_name not in pdfmetrics.getRegisteredFontNames():
                            pdfmetrics.registerFont(TTFont(custom_font_name, str(custom_font_path)))  # type: ignore
                        logging.info(f"Using custom CJK font: {custom_font_name}")
                        return custom_font_name
                    except Exception as error:
                        logging.warning(f"Failed to register custom font {custom_font}: {error}")
                        print(f"Warning: Custom font '{custom_font}' failed to load. Using default font selection.")
                else:
                    logging.warning(f"Custom font file not found: {custom_font_path}")
                    print(f"Warning: Custom font '{custom_font}.ttf' not found in fonts/ directory. Using default font selection.")

            preferred_fonts = [
                ('Arial Unicode.ttf', 'ArialUnicode'),
                ('AppleGothic.ttf', 'AppleGothic'),
                ('AppleMyungjo.ttf', 'AppleMyungjo'),
            ]

            for font_filename, font_name in preferred_fonts:
                font_path = fonts_dir / font_filename
                if font_path.exists():
                    try:
                        if font_name not in pdfmetrics.getRegisteredFontNames():
                            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))  # type: ignore
                        logging.info(f"Using preferred CJK font: {font_name} ({font_filename})")
                        return font_name
                    except Exception as error:
                        logging.warning(f"Failed to register preferred font {font_name}: {error}")
                        continue

            for font_path in fonts_dir.glob('*.ttf'):
                safe_font_name = font_path.stem.replace('-', '_').replace(',', '_').replace(' ', '_')
                try:
                    if safe_font_name not in pdfmetrics.getRegisteredFontNames():
                        pdfmetrics.registerFont(TTFont(safe_font_name, str(font_path)))  # type: ignore
                    logging.info(f"Using available CJK font: {safe_font_name} ({font_path.name})")
                    return safe_font_name
                except Exception as error:
                    logging.warning(f"Failed to register font {safe_font_name}: {error}")
                    continue

        logging.warning("No CJK fonts found in fonts/ directory.")
        print("Warning: No CJK fonts available for PDF generation.")
        print("To fix: Add CJK .ttf fonts to the 'fonts/' directory in this project.")
        print("Note: Only .ttf fonts are supported. OTF fonts will not work with reportlab.")
        print("Recommended CJK fonts:")
        print("  - Arial Unicode MS (Microsoft)")
        print("  - Source Han Sans (Adobe): https://github.com/adobe-fonts/source-han-sans")
        print("  - Apple system fonts (AppleGothic, AppleMyungjo)")
        print("Alternative: Save as .txt file for proper CJK character display.")
        return 'Times-Roman'

    except Exception as error:
        logging.warning(f"Error checking CJK fonts: {error}")
        return 'Times-Roman'


def get_docx_font(custom_font: Optional[str] = None) -> str:
    """Get an appropriate font for CJK characters for Word output."""
    try:
        fonts_dir = _fonts_dir()
        if fonts_dir.exists() and custom_font:
            custom_font_path = fonts_dir / f"{custom_font}.ttf"
            if custom_font_path.exists():
                logging.info(f"Using custom CJK font for Word: {custom_font}")
                return custom_font
            logging.warning(f"Custom font file not found: {custom_font_path}")
            print(f"Warning: Custom font '{custom_font}.ttf' not found in fonts/ directory. Using default font selection.")

        preferred_word_fonts = [
            'Arial Unicode MS',
            'AppleGothic',
            'AppleMyungjo',
            'Arial',
            'Calibri',
        ]

        selected_font = preferred_word_fonts[0]
        logging.info(f"Using CJK font for Word: {selected_font}")
        return selected_font

    except Exception as error:
        logging.warning(f"Error checking fonts for Word document: {error}")
        return 'Times New Roman'
