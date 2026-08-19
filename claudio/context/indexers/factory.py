import os

from claudio.config import config
from claudio.observability.logging import get_logger


logger = get_logger(__name__)


def _use_local_store() -> bool:
   """Fall back to the in-memory store when Qdrant isn't configured, so the CLI
   runs with zero setup (no server/Docker). Explicit provider 'local' forces it."""
   provider = config["vector_store"]["provider"]
   if provider == "local":
       return True
   return provider == "qdrant" and not os.getenv("QDRANT_URL")


def get_indexer():
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]

   if _use_local_store():
       logger.info("Vector store: local in-memory (no Qdrant configured)")
       from .local_store import index_codebase
   elif mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import index_codebase
   elif provider == "qdrant":
       from .semantic_qdrant import index_codebase
   else:
       from .semantic_chroma import index_codebase


   return index_codebase


def get_index_inspector():
   """Return the right show_index function based on rag mode and vector_store in config."""
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]

   if _use_local_store():
       from .local_store import show_index
   elif mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import show_index
   elif provider == "qdrant":
       from .semantic_qdrant import show_index
   else:
       from .semantic_chroma import show_index


   return show_index
