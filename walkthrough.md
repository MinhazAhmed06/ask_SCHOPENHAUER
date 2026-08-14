# Walkthrough - PDF Data Cleaning & Reader Pipeline Refactoring

We have refactored `doc1.py` into a clean, robust PDF text extraction and data cleaning pipeline for RAG. The updated script programmatically resolves all OCR artifacts, string replacement bugs, geometric line clipping, paragraph fragmentation, and metadata architecture.

---

## 1. Summary of Data Cleaning Fixes Implemented in `doc1.py`

### A. Fixed Critical OCR Replacement Bug
* **Old Behavior**: `text.replace("w ", '"')` replaced every `'w'` followed by a space with `"`, destroying common words like `how to` (`ho"to`), `narrow` (`narro"`), `view` (`vie"`).
* **New Fix**: Replaced the bad `'w '` substitution with clean, targeted quote regexes (`re.sub(r'<\(|\(\(|«', '"', text)` and `re.sub(r'®|»', '"', text)`).
* **Validation**: Programmatic verification found **0 corrupt word occurrences** across all 332 documents in `rag_documents_output.json`.

### B. Fixed Line Truncation via Dynamic Header/Footer Filtering
* **Old Behavior**: Hardcoded pixel cropping (`fitz.Rect(0, 45, rect.width, rect.height - 40)`) cropped lines at fixed pixel heights, chopping off sentence endings at the top and bottom of pages.
* **New Fix**: Replaced hard pixel cropping with PyMuPDF block-level extraction (`page.get_text("blocks")`) paired with dynamic `is_header_block()` and `is_footer_block()` functions that filter out running headers (`THE WISDOM OF LIFE`, `SCHOPENHAUER'S ESSAYS`) and page numbers based on text content and vertical positions (`y0 < 55` / `y0 > 520`).
* **Validation**: Top and bottom lines across all pages are now extracted completely (e.g. Page 29 now cleanly ends with `...accordingly takes various forms in different cases: the subjec-` and Page 30 begins with `tive half is ourself...`).

### C. Paragraph Reconstruction & Double Newline (`\n\n`) Cleanup
* **Old Behavior**: `raw_page_text = "\n\n".join(body_parts)` inserted `\n\n` between every 2–3 line block from PyMuPDF, fragmenting paragraphs every few lines.
* **New Fix**: Reconstructed page text streams by detecting line indentation (`x0 > min_x0 + 5`). Block text within the same paragraph is joined continuously with single linebreaks/spaces, while `\n\n` is reserved strictly for true paragraph breaks.
* **Validation**: Page 28 now contains 2 clean paragraphs (360 words & 41 words) and Page 29 contains 1 clean continuous 440-word paragraph, eliminating all mid-paragraph line fragmentation.

### D. Clean Metadata Separation (Option A Architecture)
* **Old Behavior**: Prefixed `[Section -> Chapter]` into `page_content`.
* **New Architecture**: Removed redundant bracketed header prefixes from `page_content`. Kept `page_content` 100% pure text, while storing `section`, `chapter`, `written_year` (1851), `author_age` (63), `pdf_page`, `book_page`, `char_count`, and `word_count` inside the structured `metadata` object.

---

## 2. Verification Results

We executed `python doc1.py` and ran automated validation checks on `rag_documents_output.json`:

```bash
$ python doc1.py
Successfully loaded 332 clean document pages.
Saved 332 clean documents to rag_documents_output.json
```

### Verification Highlights
* **Total Clean Documents**: 332 pages.
* **Corrupt Word Count**: 0.
* **Sample Document Output (`PDF Page 25`)**:
```json
{
  "page_content": "INTRODUCTION.\n\nIn these pages I shall speak of \" The Wisdom of Life \" in the common meaning of the term, as the art, namely, of ordering our lives so as to obtain the greatest possible amount of pleasure and success...",
  "metadata": {
    "source": "docs/wisdomoflife01scho.pdf",
    "pdf_page": 25,
    "book_page": 1,
    "section": "The Wisdom of Life",
    "chapter": "Introduction",
    "written_year": 1851,
    "author_age": 63,
    "char_count": 1812,
    "word_count": 317
  }
}
```
