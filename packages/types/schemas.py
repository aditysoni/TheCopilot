from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    source_id: str
    source_type: Literal["doc", "url", "linkedin", "analytics", "manual"]
    title: str
    url: Optional[str] = None
    snippet: str
    score: Optional[float] = None


class AnalyticsResult(BaseModel):
    metric: str
    period: str
    value: Any
    notes: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None
    sources: List[SourceItem] = []
    analytics: List[AnalyticsResult] = []
    meta: Dict[str, Any] = {}


class IngestTextRequest(BaseModel):
    title: str
    text: str = Field(..., min_length=1)
    source_type: Literal["doc", "manual", "linkedin"] = "manual"
    source_url: Optional[str] = None
    author: Optional[str] = None
    trust_level: str = "official"
    metadata: Optional[Dict[str, Any]] = None


class IngestUrlRequest(BaseModel):
    source_url: str
    title: Optional[str] = None
    author: Optional[str] = None
    trust_level: str = "official"
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    source_id: str
    chunk_count: int
    message: str


class ConversationItem(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources_used_json: Optional[Any] = None
    created_at: str


class ConversationDetailResponse(BaseModel):
    conversation: ConversationItem
    messages: List[MessageItem]

class IngestFileResponse(BaseModel):
    source_id: str
    filename: str
    chunk_count: int
    extracted_chars: int
    message: str


class IngestJsonlResponse(BaseModel):
    filename: str
    total_records: int
    ingested: int
    failed: int
    message: str


class SourceListItem(BaseModel):
    id: str
    source_type: str
    title: str
    source_url: Optional[str] = None
    author: Optional[str] = None
    trust_level: str
    created_at: str
    chunk_count: int


class SourceDetailResponse(BaseModel):
    id: str
    source_type: str
    title: str
    source_url: Optional[str] = None
    author: Optional[str] = None
    trust_level: str
    raw_text: str
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: str
    chunk_count: int


class ChunkItem(BaseModel):
    id: str
    source_id: str
    chunk_index: int
    section_title: Optional[str] = None
    chunk_text: str
    token_count: Optional[int] = None
    created_at: str


class SourceChunksResponse(BaseModel):
    source_id: str
    title: str
    chunks: List[ChunkItem]


class DeleteResponse(BaseModel):
    message: str


class ReindexResponse(BaseModel):
    source_id: str
    chunk_count: int
    message: str