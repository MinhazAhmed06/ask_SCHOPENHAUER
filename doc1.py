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
    text_clean = text_str.upper().strip()
    
    # Normalize common OCR digit misreads (e.g. '3o6' -> '306', 'i6o' -> '160', '3io' -> '310')
    normalized = re.sub(r'[oO]', '0', text_clean)
    normalized = re.sub(r'[iIl]', '1', normalized)
    lines = [l.strip() for l in text_str.split("\n") if l.strip()]
    
    # Matches running headers with page numbers or book title keywords
    keywords = [
        "WISDOM OF LIFE", "SCHOPENHAUER", "ESSAYS", "DIVISION", 
        "SUBJECT", "PERSONALITY", "PROPERTY", "POSITION", 
        "ETHICS", "SUICIDE", "PHILOSOPHY"
    ]
    
    if any(w in text_clean for w in keywords):
        return True

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

def is_footer_block(y0: float, text_str: str) -> bool:
    """Dynamically detects standalone page numbers/footers at bottom of page (y0 > 520pt)."""
    if y0 < 520:
        return False
    text_clean = text_str.strip()
    # Normalize OCR digit misreads (e.g. '(5i)' -> '(51)', '(IOI)' -> '(101)')
    norm = re.sub(r'[oO]', '0', text_clean)
    norm = re.sub(r'[iIl]', '1', norm)
    
    if re.match(r"^\(?\s*\d+\s*\)?$", norm) or re.match(r"^\(?\s*[ivxlcdm]+\s*\)?$", norm, re.I):
        return True
    if re.match(r"^\(?\s*[ivxlcdm01\s()]+\)?$", norm, re.I) and len(norm) <= 8:
        return True
    return False

def fix_ocr_formatting(text: str) -> str:
    """
    Data Cleaning Pipeline:
    1. Fixes OCR quote symbol misinterpretations (<(, ®, ((, «, » -> quotes).
    2. Rejoins hyphenated words split across lines (e.g. "suf-\nfered" -> "suffered").
    3. Collapses intra-paragraph line breaks into spaces while preserving paragraph breaks.
    4. Normalizes whitespace without destroying valid body text.
    """
    if not text:
        return ""

    # 1. Standardize OCR quote symbols without replacing valid words
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

    # 4. Collapse multi-line breaks to standard double newline
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
        body_blocks = []

        for b in blocks:
            x0, y0, x1, y1, text_str, block_no, block_type = b
            if block_type != 0 or not text_str.strip():
                continue

            # Skip dynamic top running headers and bottom footers
            if is_header_block(y0, text_str) or is_footer_block(y0, text_str):
                continue

            body_blocks.append(b)

        if not body_blocks:
            continue

        # Detect left margin baseline x0 for paragraph indentation detection
        x0_coords = [b[0] for b in body_blocks]
        min_x0 = min(x0_coords) if x0_coords else 0

        # Build text stream: indented blocks (x0 > min_x0 + 5) start a new paragraph (\n\n)
        lines_with_breaks = []
        for b in body_blocks:
            x0 = b[0]
            text_str = b[4].strip()
            if x0 > min_x0 + 5:
                lines_with_breaks.append("\n\n" + text_str)
            else:
                lines_with_breaks.append("\n" + text_str)

        raw_page_text = "".join(lines_with_breaks).strip()
        cleaned_text = fix_ocr_formatting(raw_page_text)

        # Skip Table of Contents pages & empty/corrupt pages
        if "CONTENTS" in cleaned_text or len(cleaned_text) < 40:
            continue

        section, chapter = get_chapter_info(book_page_num)

        # Build clean structured payload (100% pure text without redundant header prefix)
        structured_content = cleaned_text

        metadata = {
            "source": pdf_path,
            "pdf_page": pdf_page_num,
            "book_page": book_page_num,
            "section": section,
            "chapter": chapter,
            "written_year": 1851,
            "author_age": 63,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split())
        }

        documents.append(Document(page_content=structured_content, metadata=metadata))

    doc.close()
    return documents

def aggregate_documents_by_chapter(documents):
    """
    Groups page-level Document objects by (section, chapter) into clean chapter-level objects.
    Rejoins text streams seamlessly across page breaks (handling cross-page hyphenated words).
    """
    chapter_groups = []
    current_key = None
    current_docs = []

    for doc in documents:
        key = (doc.metadata["section"], doc.metadata["chapter"])
        if key != current_key:
            if current_docs:
                chapter_groups.append(current_docs)
            current_key = key
            current_docs = [doc]
        else:
            current_docs.append(doc)

    if current_docs:
        chapter_groups.append(current_docs)

    chapter_objects = []

    for doc_group in chapter_groups:
        source_path = doc_group[0].metadata["source"]
        section = doc_group[0].metadata["section"]
        chapter = doc_group[0].metadata["chapter"]
        
        start_book_page = doc_group[0].metadata["book_page"]
        end_book_page = doc_group[-1].metadata["book_page"]
        start_pdf_page = doc_group[0].metadata["pdf_page"]
        end_pdf_page = doc_group[-1].metadata["pdf_page"]
        total_pages = len(doc_group)

        # Concatenate text across pages with cross-page hyphenation rejoining
        combined_text = ""
        for doc in doc_group:
            page_text = doc.page_content.strip()
            if not page_text:
                continue

            if not combined_text:
                combined_text = page_text
            else:
                # Check if combined_text ends with a hyphenated word across page boundary (e.g. "subjec-")
                if re.search(r'\w+-\s*$', combined_text) and re.match(r'^\w+', page_text):
                    combined_text = re.sub(r'(\w+)-\s*$', r'\1', combined_text)
                    first_space = page_text.find(' ')
                    if first_space != -1:
                        word_part = page_text[:first_space]
                        rest = page_text[first_space:]
                        combined_text += word_part + rest
                    else:
                        combined_text += page_text
                # Check if previous page text ends with paragraph break
                elif combined_text.endswith('\n\n') or page_text.startswith('\n\n'):
                    combined_text = combined_text.rstrip() + "\n\n" + page_text.lstrip()
                else:
                    # Single space joining across page break
                    combined_text += " " + page_text

        # Clean any remaining double spaces or excess newlines
        cleaned_chapter_text = re.sub(r'[ \t]+', ' ', combined_text).strip()
        cleaned_chapter_text = re.sub(r'\n{3,}', '\n\n', cleaned_chapter_text)

        book_pages_str = f"{start_book_page}-{end_book_page}" if start_book_page != end_book_page else f"{start_book_page}"
        pdf_pages_str = f"{start_pdf_page}-{end_pdf_page}" if start_pdf_page != end_pdf_page else f"{start_pdf_page}"

        chapter_obj = {
            "chapter_content": cleaned_chapter_text,
            "metadata": {
                "source": source_path,
                "section": section,
                "chapter": chapter,
                "book_pages": book_pages_str,
                "pdf_pages": pdf_pages_str,
                "total_pages": total_pages,
                "written_year": 1851,
                "author_age": 63,
                "char_count": len(cleaned_chapter_text),
                "word_count": len(cleaned_chapter_text.split())
            }
        }
        chapter_objects.append(chapter_obj)

    return chapter_objects

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

    # 1. Save page-level documents
    page_output_filename = "rag_documents_output.json"
    with open(page_output_filename, "w", encoding="utf-8") as f:
        json.dump(docs_to_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(docs)} clean page documents to {page_output_filename}")

    # 2. Save chapter-level aggregated documents
    chapters_data = aggregate_documents_by_chapter(docs)
    chapter_output_filename = "rag_chapters_output.json"
    with open(chapter_output_filename, "w", encoding="utf-8") as f:
        json.dump(chapters_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chapters_data)} section & chapter-wise documents to {chapter_output_filename}")



