"""
In-memory dense vector store for zero-setup runs.

Used automatically when no Qdrant server is configured (no QDRANT_URL), so the
CLI works on a fresh machine with no Docker, no Qdrant, and no fastembed/
onnxruntime. Embeddings come from the configured embedder (which may be routed
through the hosted proxy), and similarity is plain cosine over numpy.

The index is a process-wide singleton: index_codebase() builds it once and the
local retriever reads the same instance, since the CLI runs in a single process.
"""
import numpy as np

from claudio.context.indexers.code_parser import parse_file, get_source_files
from claudio.llm.factory import get_embedder
from claudio.observability.logging import get_logger

logger = get_logger(__name__)

# Process-wide index: parallel lists of chunks and their (normalized) vectors.
_CHUNKS: list = []
_VECTORS: np.ndarray | None = None


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def index_codebase(repo_path: str):
    """Parse and embed the repo into an in-memory index. Idempotent per process."""
    global _CHUNKS, _VECTORS
    if _VECTORS is not None:
        logger.info(f"Using existing in-memory index ({len(_CHUNKS)} chunks)")
        return get_store()

    chunks = []
    for filepath in get_source_files(repo_path):
        try:
            chunks.extend(parse_file(filepath))
        except (SyntaxError, ValueError) as e:
            logger.error(f"Skipping {filepath}: {e}")
            continue
    if not chunks:
        raise ValueError(f"No indexable source found in {repo_path}")

    embedder = get_embedder()
    vectors = np.asarray(embedder.embed_documents([c.content for c in chunks]), dtype=np.float32)

    _CHUNKS = chunks
    _VECTORS = _normalize(vectors)
    logger.info(f"Local in-memory index built: {len(chunks)} chunks")
    return get_store()


def get_store() -> dict:
    """Return the current in-memory index (chunks + normalized vectors)."""
    return {"chunks": _CHUNKS, "vectors": _VECTORS}


def show_index(store: dict) -> None:
    from rich.console import Console

    console = Console()
    chunks = store["chunks"]
    console.print(f"\n[bold]Local Index — {len(chunks)} chunks[/bold]\n")
    for i, c in enumerate(chunks):
        console.print(f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
        console.print(f"  File  : {c.source}")
        console.print(f"  Name  : {c.name} ({c.type})")
        console.print(f"  Lines : {c.start_line} - {c.end_line}")
        console.print(f"  Code  :\n[dim]{c.content[:300]}[/dim]\n")
