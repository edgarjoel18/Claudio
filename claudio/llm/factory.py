import os

from claudio.config import config
from claudio.observability.logging import get_logger

logger = get_logger(__name__)


# Default hosted proxy — used when a proxy access token is provided but no
# custom base URL. Not a secret (the URL is public); the token gates access.
DEFAULT_PROXY_BASE = "https://claudio-proxy-x1y1.onrender.com/v1"


def _proxy_kwargs() -> dict:
    """If a Claudio proxy token is provided, route OpenAI calls through the proxy
    so users need no API key of their own (calls are billed to the proxy owner).
    The user only needs to set CLAUDIO_API_KEY; the URL defaults to the hosted
    proxy but can be overridden with CLAUDIO_API_BASE. Falls back to the standard
    OPENAI_API_KEY from the environment when no proxy token is set."""
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
