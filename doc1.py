import re
import json
import fitz  # PyMuPDF
from langchain_core.documents import Document

# 1. Correct Offset:
# PDF Index 19 (1-indexed page 20) is the start of the book body.
# PDF Index 26 (1-indexed page 27) is where "Chapter I. Division of the Subject" strictly begins.
PAGE_OFFSET = 24

TOC_MAP = [
    # WISDOM OF LIFE
    {"start": 1, "end": 2, "section": "The Wisdom of Life", "chapter": "Introduction"},
    {"start": 3, "end": 11, "section": "The Wisdom of Life", "chapter": "I. Division of the Subject"},
    {"start": 12, "end": 35, "section": "The Wisdom of Life", "chapter": "II. Personality, or What a Man Is"},
    {"start": 36, "end": 43, "section": "The Wisdom of Life", "chapter": "III. Property, or What a Man Has"},
    {"start": 44, "end": 100, "section": "The Wisdom of Life", "chapter": "IV. Position, or a Man's Place in the Estimation of Others"},

    # ESSAYS
    {"start": 101, "end": 128, "section": "Essays", "chapter": "Sketch of a History of the Doctrine of the Ideal and Real"},
    {"start": 129, "end": 236, "section": "Essays", "chapter": "Fragments of the History of Philosophy"},
    {"start": 237, "end": 254, "section": "Essays", "chapter": "On Philosophy and Its Method"},
    {"start": 255, "end": 262, "section": "Essays", "chapter": "Some Reflections on the Antithesis of Thing-In-Itself and Phenomenon"},
    {"start": 263, "end": 265, "section": "Essays", "chapter": "Some Words on Pantheism"},
    {"start": 266, "end": 303, "section": "Essays", "chapter": "On Ethics"},
    {"start": 304, "end": 317, "section": "Essays", "chapter": "On the Doctrine of the Indestructibility of Our True Nature by Death"},
    {"start": 318, "end": 322, "section": "Essays", "chapter": "On Suicide"},
    {"start": 323, "end": 999, "section": "Essays", "chapter": "Contributions to the Doctrine of the Affirmation and Negation of the Will-To-Live"},
]

def get_chapter_info(book_page: int):
    """Matches printed book page number to section and chapter metadata."""
    for item in TOC_MAP:
        if item["start"] <= book_page <= item["end"]:
            return item["section"], item["chapter"]
    return "Front / Back Matter", "General"

def is_header_block(y0: float, text_str: str) -> bool:
    """Dynamically detects running headers at top of page (y0 < 55pt)."""
    if y0 > 55:
        return False
    text_upper = text_str.upper().strip()
    lines = [l.strip() for l in text_str.split("\n") if l.strip()]
    
    # Matches running headers with page numbers or book title keywords
    if any(re.search(r"^\d+$", l) or re.search(r"^[ivxlcdm]+$", l, re.I) for l in lines):
        keywords = [
            "WISDOM OF LIFE", "SCHOPENHAUER", "ESSAYS", "DIVISION", 
            "SUBJECT", "PERSONALITY", "PROPERTY", "POSITION", 
            "ETHICS", "SUICIDE", "PHILOSOPHY"
        ]
        if any(w in text_upper for w in keywords):
            return True
        if len(lines) <= 2 and (lines[0].isdigit() or lines[-1].isdigit() or re.match(r"^[ivxlcdm]+$", lines[0], re.I)):
            return True
    return False

def is_footer_block(y0: float, text_str: str) -> bool:
    """Dynamically detects standalone page numbers at bottom of page (y0 > 520pt)."""
    if y0 < 520:
        return False
    text_clean = text_str.strip()
    if re.match(r"^\(?\s*\d+\s*\)?$", text_clean) or re.match(r"^\(?\s*[ivxlcdm]+\s*\)?$", text_clean, re.I):
        return True
    return False

def fix_ocr_formatting(text: str) -> str:
    """
    Data Cleaning Pipeline:
    1. Fixes OCR quote symbol misinterpretations (<(, ®, ((, «, » -> quotes).
    2. Rejoins hyphenated words split across lines (e.g. "suf-\nfered" -> "suffered").
    3. Collapses intra-paragraph line breaks into spaces while preserving paragraph breaks.
    4. Cleans residual running header noise and normalizes whitespace.
    """
    if not text:
        return ""

    # 1. Standardize OCR quote symbols without replacing valid words (no 'w ' bug!)
    text = re.sub(r'<\(|\(\(|«', '"', text)
    text = re.sub(r'®|»', '"', text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # 2. Join hyphenated words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # 3. Collapse intra-paragraph line breaks into single spaces while preserving double-newline paragraphs
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        # Replace single newlines inside paragraph with space
        p_clean = re.sub(r'(?<!\n)\n(?!\n)', ' ', p)
        # Collapse multiple spaces
        p_clean = re.sub(r'[ \t]+', ' ', p_clean).strip()
        if p_clean:
            cleaned_paragraphs.append(p_clean)

    cleaned_text = "\n\n".join(cleaned_paragraphs)

    # 4. Remove lingering running header noise if present
    cleaned_text = re.sub(r"(?i)THE WISDOM OF LIFE|SCHOPENHAUER'S ESSAYS", "", cleaned_text)
    
    # 5. Collapse multi-line breaks to standard double newline
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    return cleaned_text.strip()

def extract_rag_documents(pdf_path: str):
    doc = fitz.open(pdf_path)
    documents = []

    for pdf_idx, page in enumerate(doc):
        pdf_page_num = pdf_idx + 1
        book_page_num = pdf_page_num - PAGE_OFFSET

        # Skip Cover & Intro pages before printed page 1
        if book_page_num < 1:
            continue

        # Extract text blocks dynamically without hard coordinate cropping
        blocks = page.get_text("blocks")
        body_parts = []

        for b in blocks:
            x0, y0, x1, y1, text_str, block_no, block_type = b
            if block_type != 0 or not text_str.strip():
                continue

            # Skip dynamic top running headers and bottom footers
            if is_header_block(y0, text_str) or is_footer_block(y0, text_str):
                continue

            body_parts.append(text_str.strip())

        raw_page_text = "\n\n".join(body_parts)
        cleaned_text = fix_ocr_formatting(raw_page_text)

        # Skip Table of Contents pages & empty/corrupt pages
        if "CONTENTS" in cleaned_text or len(cleaned_text) < 40:
            continue

        section, chapter = get_chapter_info(book_page_num)

        # Build clean structured payload
        structured_content = f"[{section} -> {chapter}]\n{cleaned_text}"

        metadata = {
            "source": pdf_path,
            "pdf_page": pdf_page_num,
            "book_page": book_page_num,
            "section": section,
            "chapter": chapter,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split())
        }

        documents.append(Document(page_content=structured_content, metadata=metadata))

    doc.close()
    return documents

if __name__ == "__main__":
    pdf_file = "docs/wisdomoflife01scho.pdf"
    docs = extract_rag_documents(pdf_file)
    print(f"Successfully loaded {len(docs)} clean document pages.")

    docs_to_dict = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    # Save to a formatted JSON file
    output_filename = "rag_documents_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(docs_to_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(docs)} clean documents to {output_filename}")


