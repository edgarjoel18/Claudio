"""
Self-contained, read-only RAG over the Claudio codebase for the public web demo.

Design notes:
- Reuses Claudio's real tree-sitter chunker (`code_parser`) so the demo indexes
  code exactly the way the CLI does.
- Keeps the vector store in-memory (numpy cosine) instead of pulling the full
  Qdrant/FastEmbed/onnxruntime stack, so it deploys cleanly on free hosting.
- Talks to the hosted LLM proxy (never a raw OpenAI key), so the demo is
  key-free for visitors and billed to the proxy owner.
- Read-only: it retrieves + answers. It never runs commands or writes files —
  safe to expose on a public URL.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

# Make the `claudio` package importable (repo root is the parent of web/).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claudio.context.indexers.code_parser import parse_file, get_source_files  # noqa: E402

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# Which directory to index for the demo — the Claudio package itself by default.
INDEX_DIR = os.getenv("INDEX_DIR", str(REPO_ROOT / "claudio"))
EMBED_BATCH = 100
# Top retrieval score above which we treat the question as codebase-related
# (grounds the answer + shows sources). Below it, we answer conversationally.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.28"))

SYSTEM_PROMPT = (
    "You are Claudio, a friendly and knowledgeable code assistant. You have access "
    "to the Claudio project's source code, provided to you as optional context.\n\n"
    "- If the question is about the Claudio codebase, ground your answer in that "
    "context and cite specific file paths, function/class names, and line numbers.\n"
    "- If the question is general, conversational, or unrelated to the code, just "
    "answer helpfully and naturally — do not force the code context in, and do not "
    "refuse.\n"
    "Be concise and clear."
)


@dataclass
class Source:
    path: str        # repo-relative path
    name: str
    type: str
    start_line: int
    end_line: int
    score: float
    content: str


def get_client() -> OpenAI:
    """OpenAI-compatible client pointed at the hosted proxy (never a raw key)."""
    base = os.getenv("CLAUDIO_API_BASE")
    key = os.getenv("CLAUDIO_API_KEY")
    if not base or not key:
        raise RuntimeError(
            "CLAUDIO_API_BASE and CLAUDIO_API_KEY must be set (proxy URL + access token)."
        )
    return OpenAI(base_url=base, api_key=key)


def _embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(d.embedding for d in resp.data)
    return np.asarray(out, dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class CodebaseIndex:
    """In-memory dense index over the parsed code chunks."""

    def __init__(self, chunks: list, vectors: np.ndarray):
        self._chunks = chunks
        self._vectors = _normalize(vectors)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, client: OpenAI, query: str, k: int = 5) -> list[Source]:
        qv = _normalize(_embed(client, [query]))[0]
        sims = self._vectors @ qv
        top = np.argsort(-sims)[:k]
        results = []
        for i in top:
            c = self._chunks[int(i)]
            try:
                rel = str(Path(c.source).resolve().relative_to(REPO_ROOT))
            except ValueError:
                rel = c.source
            results.append(
                Source(
                    path=rel,
                    name=c.name,
                    type=c.type,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    score=float(sims[int(i)]),
                    content=c.content,
                )
            )
        return results


def build_index(client: OpenAI) -> CodebaseIndex:
    """Parse and embed the target codebase once. Reuses Claudio's tree-sitter chunker."""
    chunks = []
    for filepath in get_source_files(INDEX_DIR):
        try:
            chunks.extend(parse_file(filepath))
        except (SyntaxError, ValueError):
            continue  # skip files the parser can't handle
    if not chunks:
        raise RuntimeError(f"No indexable source found under {INDEX_DIR}")
    vectors = _embed(client, [c.content for c in chunks])
    return CodebaseIndex(chunks, vectors)


def answer(client: OpenAI, index: CodebaseIndex, question: str, k: int = 5) -> tuple[str, list[Source]]:
    """Answer any question. Code is always retrieved and offered as optional
    context, so codebase questions are grounded and cited; general questions are
    answered conversationally. Sources are only surfaced when the retrieved code
    is actually relevant (by score), so chit-chat doesn't show random files."""
    sources = index.search(client, question, k=k)
    context = "\n\n".join(
        f"### {s.path} — {s.name} ({s.type}, lines {s.start_line}-{s.end_line})\n{s.content}"
        for s in sources
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    f"Possibly-relevant code from the Claudio project "
                    f"(use only if it helps answer the question):\n{context}"
                ),
            },
        ],
    )
    top_score = sources[0].score if sources else 0.0
    shown = sources if top_score >= RELEVANCE_THRESHOLD else []
    return resp.choices[0].message.content, shown
