from apps.api.app.core.config import settings
from services.llm.providers.gemini_provider import GeminiProvider
from services.llm.providers.llama_provider import LlamaProvider


def get_llm_provider():
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        return GeminiProvider()

    if provider == "llama":
        return LlamaProvider()

    raise ValueError(f"Unsupported LLM provider: {provider}")