class MetadataExtractor:
    """Enriches extracted text blocks with structural book metadata."""

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

    @classmethod
    def get_book_page(cls, pdf_page: int) -> int:
        return pdf_page - cls.PAGE_OFFSET

    @classmethod
    def get_chapter_info(cls, book_page: int) -> tuple[str, str]:
        for item in cls.TOC_MAP:
            if item["start"] <= book_page <= item["end"]:
                return item["section"], item["chapter"]
        return "Front / Back Matter", "General"

    @classmethod
    def build_metadata(
        cls,
        source: str,
        pdf_page: int,
        content: str,
        extra: dict | None = None
    ) -> dict:
        book_page = cls.get_book_page(pdf_page)
        section, chapter = cls.get_chapter_info(book_page)
        
        metadata = {
            "source": str(source),
            "pdf_page": pdf_page,
            "book_page": book_page,
            "section": section,
            "chapter": chapter,
            "author": "Arthur Schopenhauer",
            "written_year": 1851,
            "author_age": 63,
            "char_count": len(content),
            "word_count": len(content.split()),
        }
        if extra:
            metadata.update(extra)
        return metadata
