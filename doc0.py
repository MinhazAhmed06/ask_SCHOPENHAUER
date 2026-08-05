import fitz  # PyMuPDF
from langchain_core.documents import Document

# 1. TOC Mapping using printed book page numbers
TOC_MAP = [
    {"start": 1, "end": 2, "section": "The Wisdom of Life", "chapter": "Introduction"},
    {"start": 3, "end": 11, "section": "The Wisdom of Life", "chapter": "I. Division of the Subject"},
    {"start": 12, "end": 35, "section": "The Wisdom of Life", "chapter": "II. Personality, or What a Man Is"},
    {"start": 36, "end": 43, "section": "The Wisdom of Life", "chapter": "III. Property, or What a Man Has"},
    {"start": 44, "end": 100, "section": "The Wisdom of Life", "chapter": "IV. Position, or a Man's Place in the Estimation of Others"},
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
    """Matches a printed book page number to its section and chapter."""
    for item in TOC_MAP:
        if item["start"] <= book_page <= item["end"]:
            return item["section"], item["chapter"]
    return "Front / Back Matter", "General"

def is_page_empty(page: fitz.Page, min_char_count: int = 20) -> bool:
    """Checks if a page has negligible text and no images."""
    text = page.get_text("text").strip()
    images = page.get_images()
    
    # If text is extremely short and there are no images, consider it empty
    if len(text) < min_char_count and len(images) == 0:
        return True
    return False

def pdf_loader_with_clean_metadata(pdf_path: str, page_offset: int = 0, top_crop: float = 45, bottom_crop: float = 40):
    """
    Loads PDF, skips empty pages & pre-content cover pages, and maps TOC metadata.
    
    :param pdf_path: Path to the PDF file.
    :param page_offset: Number of PDF pages before printed Page 1.
    """
    doc = fitz.open(pdf_path)
    documents = []

    for pdf_idx, page in enumerate(doc):
        # 1. Skip completely blank or empty pages
        if is_page_empty(page):
            continue

        # Calculate printed book page number
        pdf_page_num = pdf_idx + 1
        book_page_num = pdf_page_num - page_offset

        # 2. Handle pages before the main content (e.g., covers, intro images)
        if book_page_num < 1:
            # You can choose to skip cover/intro entirely, or tag them as 'Front Matter'
            section, chapter = "Front Matter", "Preface & TOC"
        else:
            section, chapter = get_chapter_info(book_page_num)

        # 3. Crop top/bottom header & footer margins
        rect = page.rect
        crop_box = fitz.Rect(0, top_crop, rect.width, rect.height - bottom_crop)
        raw_text = page.get_text("text", clip=crop_box).strip()

        # Re-check text after cropping running headers
        if len(raw_text) < 10:
            continue

        # Prepend chapter hierarchy to text content for better embedding quality
        structured_text = f"[{section} -> {chapter}]\n{raw_text}"

        metadata = {
            "source": pdf_path,
            "pdf_page": pdf_page_num,
            "book_page": max(book_page_num, 0),
            "section": section,
            "chapter": chapter
        }

        documents.append(Document(page_content=structured_text, metadata=metadata))

    return documents

import json

# Run your pipeline
docs = pdf_loader_with_clean_metadata("docs/wisdomoflife01scho.pdf", page_offset=10)

# Convert Document objects to standard Python dictionaries
docs_to_dict = [
    {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in docs
]

# Save to a formatted JSON file
with open("rag_documents_output.json", "w", encoding="utf-8") as f:
    json.dump(docs_to_dict, f, indent=2, ensure_ascii=False)

print(f"Saved {len(docs)} documents to rag_documents_output.json")