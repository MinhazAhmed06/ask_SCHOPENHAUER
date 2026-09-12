import json
import re
from pathlib import Path
from typing import Sequence
import fitz  # PyMuPDF
from langchain_core.documents import Document

from .parser import PDFParser
from .metadata_extractor import MetadataExtractor
from .chunker import SmartChunker
from ..config import settings


class IngestionPipeline:
    """Orchestrates end-to-end extraction, normalization, aggregation, and chunking."""

    def __init__(
        self,
        parser: PDFParser | None = None,
        extractor: MetadataExtractor | None = None,
        chunker: SmartChunker | None = None,
    ):
        self.parser = parser or PDFParser()
        self.extractor = extractor or MetadataExtractor()
        self.chunker = chunker or SmartChunker()

    def load_pdf_pages(self, pdf_path: str | Path) -> list[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc = fitz.open(pdf_path)
        documents: list[Document] = []

        for pdf_idx, page in enumerate(doc):
            pdf_page_num = pdf_idx + 1
            book_page_num = self.extractor.get_book_page(pdf_page_num)

            # Skip front covers & prefaces before page 1
            if book_page_num < 1:
                continue

            cleaned_text = self.parser.parse_page_blocks(page)

            # Skip Table of Contents and empty/corrupt pages
            if "CONTENTS" in cleaned_text or len(cleaned_text) < 40:
                continue

            metadata = self.extractor.build_metadata(
                source=str(pdf_path),
                pdf_page=pdf_page_num,
                content=cleaned_text,
                extra={"doc_id": f"page_{pdf_page_num}"},
            )

            documents.append(Document(page_content=cleaned_text, metadata=metadata))

        doc.close()
        return documents

    def strip_chapter_title_heading(self, text: str, section: str, chapter: str) -> str:
        """Strip redundant running chapter headers at the start of chapter body."""
        cleaned = text
        cleaned = re.sub(r'^(?:THE\s+WISDOM\s+OF\s+LIFE|ESSAYS)\.?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^CHAPTER\s+[IVXLCDM\d]+[\.:]?\s*', '', cleaned, flags=re.IGNORECASE)
        
        clean_chap_name = re.sub(r'^[IVXLCDM\d]+\.\s*', '', chapter).strip()
        if clean_chap_name:
            words = re.split(r'[\s\-]+', clean_chap_name)
            word_patterns = []
            for w in words:
                if not w:
                    continue
                if w.upper() == 'SUICIDE':
                    word_patterns.append(r'SU\s*I\s*CIDE')
                else:
                    word_patterns.append(re.escape(w))
            pattern = r'^\s*' + r'[\s\n\-\'\"]+'.join(word_patterns) + r'[\.\*\s\n:]*'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r'^SECTION\s+[IVXLCDM\d]+[\.:\s]*[A-Z\s]+[\.:]?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Section\s+[IVXLCDM\d]+[\.:\s]*ON\s+THE\s+SAME[\.:]?\s*', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def aggregate_chapters(self, documents: Sequence[Document]) -> list[Document]:
        """Groups page documents by chapter and rejoins broken paragraphs/hyphens."""
        chapter_groups: list[list[Document]] = []
        current_key = None
        current_docs: list[Document] = []

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

        chapter_documents: list[Document] = []

        for idx, doc_group in enumerate(chapter_groups):
            source_path = doc_group[0].metadata["source"]
            section = doc_group[0].metadata["section"]
            chapter = doc_group[0].metadata["chapter"]
            start_book_page = doc_group[0].metadata["book_page"]
            end_book_page = doc_group[-1].metadata["book_page"]
            start_pdf_page = doc_group[0].metadata["pdf_page"]
            end_pdf_page = doc_group[-1].metadata["pdf_page"]

            combined_text = ""
            for doc in doc_group:
                page_text = doc.page_content.strip()
                if not page_text:
                    continue
                if not combined_text:
                    combined_text = page_text
                else:
                    if re.search(r'\w+-\s*$', combined_text) and re.match(r'^\w+', page_text):
                        combined_text = re.sub(r'(\w+)-\s*$', r'\1', combined_text)
                        first_space = page_text.find(' ')
                        if first_space != -1:
                            combined_text += page_text[:first_space] + page_text[first_space:]
                        else:
                            combined_text += page_text
                    elif combined_text.endswith('\n\n') or page_text.startswith('\n\n'):
                        combined_text = combined_text.rstrip() + "\n\n" + page_text.lstrip()
                    else:
                        combined_text += " " + page_text

            cleaned_text = re.sub(r'[ \t]+', ' ', combined_text).strip()
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
            cleaned_text = self.strip_chapter_title_heading(cleaned_text, section, chapter)

            book_pages = f"{start_book_page}-{end_book_page}" if start_book_page != end_book_page else f"{start_book_page}"
            pdf_pages = f"{start_pdf_page}-{end_pdf_page}" if start_pdf_page != end_pdf_page else f"{start_pdf_page}"

            meta = {
                "doc_id": f"chap_{idx+1}",
                "source": source_path,
                "section": section,
                "chapter": chapter,
                "book_pages": book_pages,
                "pdf_pages": pdf_pages,
                "total_pages": len(doc_group),
                "author": "Arthur Schopenhauer",
                "written_year": 1851,
                "author_age": 63,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
            }
            chapter_documents.append(Document(page_content=cleaned_text, metadata=meta))

        return chapter_documents

    def run(self, pdf_path: str | Path | None = None) -> dict[str, list[Document]]:
        target_pdf = Path(pdf_path or settings.DEFAULT_PDF_PATH)
        pages = self.load_pdf_pages(target_pdf)
        chapters = self.aggregate_chapters(pages)
        chunks = self.chunker.split_documents(chapters)

        # Persist JSON files to DATA_DIR
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(settings.DATA_DIR / "rag_pages.json", "w", encoding="utf-8") as f:
            json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in pages], f, indent=2, ensure_ascii=False)
            
        with open(settings.DATA_DIR / "rag_chapters.json", "w", encoding="utf-8") as f:
            json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in chapters], f, indent=2, ensure_ascii=False)
            
        with open(settings.DATA_DIR / "rag_chunks.json", "w", encoding="utf-8") as f:
            json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in chunks], f, indent=2, ensure_ascii=False)

        return {"pages": pages, "chapters": chapters, "chunks": chunks}
