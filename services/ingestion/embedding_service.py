from typing import List
import requests
import google.generativeai as genai

from apps.api.app.core.config import settings
from packages.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.has_gemini = bool(settings.GEMINI_API_KEY)
        if self.has_gemini:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        self.ollama_embed_url = f"{settings.LLAMA_BASE_URL.rstrip('/')}/api/embeddings"

    def _ollama_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                self.ollama_embed_url,
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as exc:
            logger.error("Ollama embedding failed: %s", exc)
            return []

    def embed_text(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        if self.has_gemini:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]
            except Exception as exc:
                logger.exception("Gemini embedding failed, falling back to Ollama: %s", exc)

        return self._ollama_embedding(text)

    def embed_query(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        if self.has_gemini:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except Exception as exc:
                logger.exception("Gemini query embedding failed, falling back to Ollama: %s", exc)

        return self._ollama_embedding(text)
