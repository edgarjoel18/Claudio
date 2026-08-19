import os

from claudio.config import config
from claudio.observability.logging import get_logger

logger = get_logger(__name__)


def _use_local_store() -> bool:
   provider = config["vector_store"]["provider"]
   if provider == "local":
       return True
   return provider == "qdrant" and not os.getenv("QDRANT_URL")


def get_retriever():
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]

   if _use_local_store():
       from .local_store import retrieve
   elif mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import retrieve
   elif provider == "qdrant":
       from .semantic_qdrant import retrieve
   else:
       from .semantic_chroma import retrieve

   return retrieve