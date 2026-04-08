from typing import List

import requests

from apps.api.app.core.config import settings
from services.llm.base import BaseLLMProvider
from packages.types.schemas import SourceItem, AnalyticsResult
from packages.utils.logger import get_logger

logger = get_logger(__name__)


class LlamaProvider(BaseLLMProvider):
    provider_name = "llama"

    def __init__(self) -> None:
        self.model_name = settings.LLAMA_MODEL
        self.base_url = settings.LLAMA_BASE_URL.rstrip("/")

    def _build_prompt(
        self,
        user_message: str,
        sources: List[SourceItem],
        analytics: List[AnalyticsResult],
    ) -> str:
        numbered_sources = "\n\n".join(
            [
                (
                    f"[{i+1}]\n"
                    f"Title: {s.title}\n"
                    f"Type: {s.source_type}\n"
                    f"URL: {s.url or 'N/A'}\n"
                    f"Snippet: {s.snippet}"
                )
                for i, s in enumerate(sources)
            ]
        )

        numbered_analytics = "\n".join(
            [
                f"[A{i+1}] {a.metric} ({a.period}): {a.value}. {a.notes or ''}"
                for i, a in enumerate(analytics)
            ]
        )

        return f"""
You are Rubix AI, a grounded knowledge assistant for Rubix.

Follow these rules strictly:
- Answer ONLY from the provided evidence.
- Do NOT invent facts.
- Do NOT use outside knowledge.
- If the evidence is weak, incomplete, or unrelated, say exactly:
  "I don’t have enough grounded information to answer that yet."
- Keep the answer concise and factual.
- If sources conflict, say so clearly.

Citation rules:
- Every factual claim must cite at least one source.
- Use citations like [1], [2], [1][2], [A1], [A2].
- Use only the citation numbers provided below.
- Do not invent citations.
- If the answer is "I don’t have enough grounded information to answer that yet.", do not add citations.

User question:
{user_message}

Retrieved sources:
{numbered_sources or 'No retrieved sources'}

Analytics data:
{numbered_analytics or 'No analytics data'}

Now write the final answer.
""".strip()

    def generate(
        self,
        user_message: str,
        sources: List[SourceItem],
        analytics: List[AnalyticsResult],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        prompt = self._build_prompt(user_message, sources, analytics)

        logger.info("Calling Llama model: %s", self.model_name)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        text = data.get("response", "").strip()

        if text:
            return text

        return "I don’t have enough grounded information to answer that yet."