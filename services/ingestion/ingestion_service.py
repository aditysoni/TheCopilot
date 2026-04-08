from sqlalchemy.orm import Session

from packages.db.models import Source, DocumentChunk
from packages.utils.logger import get_logger
from services.ingestion.chunking import TextChunker
from services.ingestion.embedding_service import EmbeddingService
from services.ingestion.url_extractor import UrlExtractor
from services.ingestion.file_extractor import FileExtractor
from services.ingestion.document_parser import DocumentParser

logger = get_logger(__name__)


class IngestionService:
    def __init__(self) -> None:
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.url_extractor = UrlExtractor()
        self.file_extractor = FileExtractor()
        self.document_parser = DocumentParser()

    def ingest_text_source(
        self,
        db: Session,
        *,
        title: str,
        text: str,
        source_type: str,
        source_url: str | None = None,
        author: str | None = None,
        trust_level: str = "official",
        metadata: dict | None = None,
    ) -> tuple[str, int]:
        logger.info("Ingesting source: %s", title)

        text = (text or "").strip()
        if not text:
            raise ValueError("No text extracted for ingestion")

        source = Source(
            source_type=source_type,
            title=title,
            source_url=source_url,
            author=author,
            trust_level=trust_level,
            raw_text=text,
            metadata_json=metadata or {},
        )
        db.add(source)
        db.flush()

        sections = self.document_parser.parse_plain_text(title=title, text=text)
        chunk_records = self.chunker.chunk_sections(sections)

        for idx, item in enumerate(chunk_records):
            chunk_text = item["chunk_text"]
            embedding = self.embedding_service.embed_text(chunk_text)

            chunk = DocumentChunk(
                source_id=source.id,
                chunk_index=idx,
                section_title=item.get("section_title"),
                chunk_text=chunk_text,
                token_count=len(chunk_text.split()),
                embedding_json=embedding,
                embedding=embedding,
            )
            db.add(chunk)

        db.commit()
        db.refresh(source)

        logger.info("Ingested source %s with %s chunks", source.id, len(chunk_records))
        return source.id, len(chunk_records)

    def ingest_url(
        self,
        db: Session,
        *,
        source_url: str,
        title: str | None = None,
        author: str | None = None,
        trust_level: str = "official",
        metadata: dict | None = None,
    ) -> tuple[str, int]:
        extracted_title, extracted_text = self.url_extractor.fetch_and_extract(source_url)

        final_title = title or extracted_title

        return self.ingest_text_source(
            db=db,
            title=final_title,
            text=extracted_text,
            source_type="url",
            source_url=source_url,
            author=author,
            trust_level=trust_level,
            metadata=metadata,
        )

    def ingest_jsonl(
        self,
        db: Session,
        *,
        jsonl_text: str,
        author: str | None = None,
        trust_level: str = "official",
    ) -> tuple[int, int]:
        import json

        lines = [l.strip() for l in jsonl_text.splitlines() if l.strip()]
        ingested = 0
        failed = 0

        for line in lines:
            try:
                record = json.loads(line)
                title = (
                    record.get("title")
                    or record.get("name")
                    or record.get("id", "Untitled")
                )
                text = (
                    record.get("text")
                    or record.get("content")
                    or record.get("body")
                    or record.get("answer")
                    or ""
                )
                if record.get("question") and record.get("answer"):
                    text = f"Q: {record['question']}\nA: {record['answer']}"
                if not text:
                    failed += 1
                    continue
                self.ingest_text_source(
                    db=db,
                    title=str(title),
                    text=str(text),
                    source_type="doc",
                    author=author,
                    trust_level=trust_level,
                    metadata={"source": "jsonl"},
                )
                ingested += 1
            except Exception as exc:
                logger.warning("Failed to ingest JSONL record: %s", exc)
                failed += 1

        return ingested, failed

    def ingest_file(
        self,
        db: Session,
        *,
        file_path: str,
        original_filename: str,
        title: str | None = None,
        author: str | None = None,
        trust_level: str = "official",
        metadata: dict | None = None,
    ) -> tuple[str, int, int]:
        extracted_title, extracted_text = self.file_extractor.extract(file_path)

        final_title = title or extracted_title
        source_id, chunk_count = self.ingest_text_source(
            db=db,
            title=final_title,
            text=extracted_text,
            source_type="doc",
            source_url=None,
            author=author,
            trust_level=trust_level,
            metadata={
                **(metadata or {}),
                "original_filename": original_filename,
                "ingestion_type": "file",
            },
        )

        return source_id, chunk_count, len(extracted_text)