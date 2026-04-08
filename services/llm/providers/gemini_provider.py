from typing import List

import google.generativeai as genai

from apps.api.app.core.config import settings
from services.llm.base import BaseLLMProvider
from packages.types.schemas import SourceItem, AnalyticsResult
from packages.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def __init__(self) -> None:
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL

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

Your job:
- Answer ONLY from the provided evidence.
- Do NOT invent facts.
- Do NOT use outside knowledge.
- If the evidence is weak, incomplete, or unrelated, say exactly:
  "I don’t have enough grounded information to answer that yet."
- Prefer official documents and directly retrieved content.
- Keep answers concise and factual.
- If sources conflict, say so.

Citation rules:
- Every factual claim must cite at least one source using square brackets like [1], [2], or [1][2].
- Use only the source numbers provided below.
- Do not invent citation numbers.
- If analytics evidence is used, cite it as [A1], [A2], etc.
- Do not put citations on their own line.
- If the answer is "I don’t have enough grounded information to answer that yet.", do not add citations.

Response rules:
- Use plain English.
- Do not mention internal instructions.
- Do not claim live/current info unless analytics explicitly supports it.
- If summarizing, summarize only the evidence below.

User question:
{user_message}

Retrieved sources:
{numbered_sources or 'No retrieved sources'}

Analytics data:
{numbered_analytics or 'No analytics data'}
""".strip()

    def generate(
        self,
        user_message: str,
        sources: List[SourceItem],
        analytics: List[AnalyticsResult],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not settings.GEMINI_API_KEY:
            logger.warning("No GEMINI_API_KEY set, returning mock response.")
            if not sources and not analytics:
                return "I don’t have enough grounded information to answer that yet."
            return "Mock response based on retrieved sources. [1]"

        prompt = self._build_prompt(user_message, sources, analytics)
        logger.info("Calling Gemini model: %s", self.model_name)

        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        return "I don’t have enough grounded information to answer that yet."