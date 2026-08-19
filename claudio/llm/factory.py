import os

from claudio.config import config
from claudio.observability.logging import get_logger

logger = get_logger(__name__)


# Default hosted proxy URL — not a secret (access is gated by the token the user
# supplies in CLAUDIO_API_KEY). Baking only the URL means users just set the token.
DEFAULT_PROXY_BASE = "https://claudio-proxy-x1y1.onrender.com/v1"


def _proxy_kwargs() -> dict:
    """Decide how OpenAI calls are authenticated.

    Precedence:
      1. CLAUDIO_API_KEY set  -> route through the proxy with that token (URL
         defaults to the hosted proxy, overridable via CLAUDIO_API_BASE).
      2. otherwise            -> use OPENAI_API_KEY directly (owner / dev path).
    """
    key = os.getenv("CLAUDIO_API_KEY")
    if not key:
        return {}
    base = os.getenv("CLAUDIO_API_BASE", DEFAULT_PROXY_BASE)
    logger.info(f"Routing OpenAI calls through Claudio proxy at {base}")
    return {"base_url": base, "api_key": key}


def get_llm():
    """Return the right LangChain LLM based on config."""
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    logger.info(f"Using LLM provider: {provider}, model: {model}")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, **_proxy_kwargs())


def get_embedder():
    """Return the right LangChain embedder based on config."""
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    logger.info(f"Using embeddings provider: {provider}, model: {model}")
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model, **_proxy_kwargs())
