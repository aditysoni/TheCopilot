from math import sqrt
from typing import List
from sqlalchemy.orm import Session

from packages.db.models import DocumentChunk
from packages.types.schemas import SourceItem
from packages.utils.logger import get_logger
from services.ingestion.embedding_service import EmbeddingService

logger = get_logger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def source_weight(source_type: str) -> float:
    weights = {
        "manual": 1.15,
        "doc": 1.15,
        "url": 1.0,
        "linkedin": 0.95,
        "analytics": 1.2,
    }
    return weights.get(source_type, 1.0)


class RetrievalService:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()

    def search(
        self,
        db: Session,
        query: str,
        limit: int = 5,
        min_score: float = 0.2,
    ) -> List[SourceItem]:
        logger.info("Retrieval search called for query: %s", query)

        query_embedding = self.embedding_service.embed_query(query)
        chunks = db.query(DocumentChunk).all()

        scored = []
        for chunk in chunks:
            chunk_embedding = chunk.embedding_json or []
            base_score = cosine_similarity(query_embedding, chunk_embedding)
            weighted_score = min(base_score * source_weight(chunk.source.source_type), 1.0)

            scored.append(
                {
                    "score": weighted_score,
                    "base_score": base_score,
                    "chunk": chunk,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)

        deduped: List[SourceItem] = []
        seen_source_ids = set()

        for item in scored:
            chunk = item["chunk"]
            score = item["score"]

            if score < min_score:
                continue

            if chunk.source.id in seen_source_ids:
                continue

            seen_source_ids.add(chunk.source.id)

            deduped.append(
                SourceItem(
                    source_id=chunk.source.id,
                    source_type=chunk.source.source_type,
                    title=chunk.source.title,
                    url=chunk.source.source_url,
                    snippet=chunk.chunk_text[:400],
                    score=round(score, 4),
                )
            )

            if len(deduped) >= limit:
                break

        return deduped