import pytest
from langchain_core.documents import Document
from rag.ingestion.parser import PDFParser
from rag.ingestion.metadata_extractor import MetadataExtractor
from rag.ingestion.chunker import SmartChunker


def test_pdf_parser_ocr_cleaning():
    parser = PDFParser()
    sample_text = "In these pages I shall speak of <( The Wisdom of Life )) and the art of or-\ndering our lives."
    cleaned = parser.clean_ocr_text(sample_text)
    assert '<(' not in cleaned
    assert '))' not in cleaned
    assert '" The Wisdom of Life "' in cleaned
    assert 'ordering' in cleaned  # Hyphen rejoining


def test_metadata_extractor():
    extractor = MetadataExtractor()
    sec, chap = extractor.get_chapter_info(book_page=15)
    assert sec == "The Wisdom of Life"
    assert "Personality" in chap

    meta = extractor.build_metadata(
        source="docs/wisdomoflife01scho.pdf",
        pdf_page=39,
        content="Sample content",
    )
    assert meta["book_page"] == 15
    assert meta["author"] == "Arthur Schopenhauer"
    assert meta["written_year"] == 1851


def test_smart_chunker():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=20)
    doc = Document(
        page_content="This is the first sentence. " * 10,
        metadata={"doc_id": "doc_test", "chapter": "Test Chapter"},
    )
    chunks = chunker.split_documents([doc])
    assert len(chunks) > 1
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["parent_id"] == "doc_test"
    assert chunks[0].metadata["token_count"] > 0
