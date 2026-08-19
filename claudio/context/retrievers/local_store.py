"""
Retriever for the in-memory dense store (see indexers/local_store.py).

Reads the same process-wide index built at startup and returns the top-k chunks
by cosine similarity, in the same dict shape the agent's search tool expects.
"""
import numpy as np

from claudio.context.indexers.local_store import get_store
from claudio.llm.factory import get_embedder
from claudio.observability.logging import get_logger

logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    store = get_store()
    vectors = store["vectors"]
    chunks = store["chunks"]
    if vectors is None or not chunks:
        logger.warning("Local index is empty — nothing to retrieve")
        return []

    qv = np.asarray(get_embedder().embed_query(query), dtype=np.float32)
    norm = np.linalg.norm(qv)
    qv = qv / (norm if norm else 1.0)

    sims = vectors @ qv
    top = np.argsort(-sims)[:k]

    results = []
    for i in top:
        c = chunks[int(i)]
        results.append({
            "content": c.content,
            "source": c.source,
            "name": c.name,
            "type": c.type,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "distance": float(sims[int(i)]),
        })
    logger.info(f"Retrieved {len(results)} chunks (local, dense) for query: {query}")
    return results
