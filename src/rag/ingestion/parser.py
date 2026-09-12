import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document


class PDFParser:
    """
    Production-grade PDF parsing and OCR text cleaning pipeline.
    
    Resolves:
    1. Running headers & standalone page number footers via dynamic vertical coordinate analysis.
    2. OCR character/quote misinterpretations.
    3. Split hyphenated words across linebreaks.
    4. Paragraph structure reconstruction from raw spatial blocks.
    """

    def __init__(self, header_y_threshold: float = 55.0, footer_y_threshold: float = 520.0):
        self.header_y_threshold = header_y_threshold
        self.footer_y_threshold = footer_y_threshold
        self.header_keywords = [
            "WISDOM OF LIFE", "SCHOPENHAUER", "ESSAYS", "DIVISION", 
            "SUBJECT", "PERSONALITY", "PROPERTY", "POSITION", 
            "ETHICS", "SUICIDE", "PHILOSOPHY"
        ]

    def is_header_block(self, y0: float, text_str: str) -> bool:
        """Detect running headers at top of page (y0 < threshold)."""
        if y0 > self.header_y_threshold:
            return False
            
        text_clean = text_str.upper().strip()
        if any(w in text_clean for w in self.header_keywords):
            return True

        lines = [l.strip() for l in text_str.split("\n") if l.strip()]
        for l in lines:
            l_norm = re.sub(r'[oO]', '0', re.sub(r'[iIl]', '1', l))
            if re.search(r"^\(?\s*\d+\s*\)?$", l_norm) or re.search(r"^\(?\s*[ivxlcdm]+\s*\)?$", l_norm, re.I):
                return True

        if len(lines) <= 2:
            first_norm = re.sub(r'[oO]', '0', re.sub(r'[iIl]', '1', lines[0]))
            last_norm = re.sub(r'[oO]', '0', re.sub(r'[iIl]', '1', lines[-1]))
            if re.match(r"^\(?\s*\d+\s*\)?$", first_norm) or re.match(r"^\(?\s*\d+\s*\)?$", last_norm):
                return True

        return False

    def is_footer_block(self, y0: float, text_str: str) -> bool:
        """Detect standalone page numbers or footers at bottom of page (y0 > threshold)."""
        if y0 < self.footer_y_threshold:
            return False
            
        text_clean = text_str.strip()
        norm = re.sub(r'[oO]', '0', re.sub(r'[iIl]', '1', text_clean))
        
        if re.match(r"^\(?\s*\d+\s*\)?$", norm) or re.match(r"^\(?\s*[ivxlcdm]+\s*\)?$", norm, re.I):
            return True
        if re.match(r"^\(?\s*[ivxlcdm01\s()]+\)?$", norm, re.I) and len(norm) <= 8:
            return True
        return False

    def clean_ocr_text(self, text: str) -> str:
        """Clean OCR quotes, hyphens, and whitespace while maintaining paragraph breaks."""
        if not text:
            return ""

        # 1. Standardize quote symbols
        text = re.sub(r'<\(|\(\(|\)\)|«|»|®', '"', text)
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

        # 2. Join hyphenated words broken across linebreaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 3. Collapse intra-paragraph line breaks into single spaces
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for p in paragraphs:
            p_clean = re.sub(r'(?<!\n)\n(?!\n)', ' ', p)
            p_clean = re.sub(r'[ \t]+', ' ', p_clean).strip()
            if p_clean:
                cleaned_paragraphs.append(p_clean)

        cleaned_text = "\n\n".join(cleaned_paragraphs)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        return cleaned_text.strip()

    def parse_page_blocks(self, page: fitz.Page) -> str:
        """Extracts and orders body blocks, detecting paragraph indentations."""
        blocks = page.get_text("blocks")
        body_blocks = []

        for b in blocks:
            x0, y0, x1, y1, text_str, block_no, block_type = b
            if block_type != 0 or not text_str.strip():
                continue
            if self.is_header_block(y0, text_str) or self.is_footer_block(y0, text_str):
                continue
            body_blocks.append(b)

        if not body_blocks:
            return ""

        # Baseline left margin indentation calculation
        min_x0 = min(b[0] for b in body_blocks)
        lines_with_breaks = []
        for b in body_blocks:
            x0 = b[0]
            text_str = b[4].strip()
            if x0 > min_x0 + 5:
                lines_with_breaks.append("\n\n" + text_str)
            else:
                lines_with_breaks.append("\n" + text_str)

        raw_page_text = "".join(lines_with_breaks).strip()
        return self.clean_ocr_text(raw_page_text)
