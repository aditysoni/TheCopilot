from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from apps.api.app.core.config import settings
from packages.db.database import get_db
from packages.types.schemas import (
    IngestResponse,
    IngestTextRequest,
    IngestUrlRequest,
    IngestFileResponse,
    IngestJsonlResponse,
)
from services.ingestion.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest, db: Session = Depends(get_db)):
    try:
        source_id, chunk_count = ingestion_service.ingest_text_source(
            db=db,
            title=request.title,
            text=request.text,
            source_type=request.source_type,
            source_url=request.source_url,
            author=request.author,
            trust_level=request.trust_level,
            metadata=request.metadata,
        )
        return IngestResponse(
            source_id=source_id,
            chunk_count=chunk_count,
            message="Text ingested successfully",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest/url", response_model=IngestResponse)
def ingest_url(request: IngestUrlRequest, db: Session = Depends(get_db)):
    try:
        source_id, chunk_count = ingestion_service.ingest_url(
            db=db,
            source_url=request.source_url,
            title=request.title,
            author=request.author,
            trust_level=request.trust_level,
            metadata=request.metadata,
        )
        return IngestResponse(
            source_id=source_id,
            chunk_count=chunk_count,
            message="URL ingested successfully",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestFileResponse)
def ingest_file(
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    author: str | None = Form(default=None),
    trust_level: str = Form(default="official"),
):
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        ext = Path(file.filename).suffix.lower()
        allowed = {".txt", ".md", ".pdf", ".docx"}
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(allowed))}",
            )

        saved_name = f"{uuid.uuid4()}{ext}"
        saved_path = upload_dir / saved_name

        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        source_id, chunk_count, extracted_chars = ingestion_service.ingest_file(
            db=db,
            file_path=str(saved_path),
            original_filename=file.filename,
            title=title,
            author=author,
            trust_level=trust_level,
        )

        return IngestFileResponse(
            source_id=source_id,
            filename=file.filename,
            chunk_count=chunk_count,
            extracted_chars=extracted_chars,
            message="File ingested successfully",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest/jsonl", response_model=IngestJsonlResponse)
def ingest_jsonl(
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    author: str | None = Form(default=None),
    trust_level: str = Form(default="official"),
):
    try:
        if not file.filename or not file.filename.endswith(".jsonl"):
            raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

        content = file.file.read().decode("utf-8")
        total = len([l for l in content.splitlines() if l.strip()])

        ingested, failed = ingestion_service.ingest_jsonl(
            db=db,
            jsonl_text=content,
            author=author,
            trust_level=trust_level,
        )

        return IngestJsonlResponse(
            filename=file.filename,
            total_records=total,
            ingested=ingested,
            failed=failed,
            message=f"Ingested {ingested}/{total} records",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc