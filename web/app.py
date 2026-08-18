"""
Claudio — public web demo.

A read-only chat UI that answers questions about the Claudio codebase using
retrieval-augmented generation. Visitors need no API key and no install; the
app talks to a hosted LLM proxy (billed to the project owner).

Run locally:   streamlit run web/app.py
Deploy:        Streamlit Community Cloud → main file path `web/app.py`
"""
import os

import streamlit as st

# Bridge Streamlit secrets → env vars that rag.py reads (works locally too).
for _key in (
    "CLAUDIO_API_BASE", "CLAUDIO_API_KEY", "EMBED_MODEL", "CHAT_MODEL",
    "INDEX_DIR", "RELEVANCE_THRESHOLD",
):
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = str(st.secrets[_key])

import rag  # noqa: E402  (imported after secrets are wired into the environment)

st.set_page_config(page_title="Claudio — code assistant demo", page_icon="🤖", layout="centered")

st.title("🤖 Claudio")
st.caption(
    "A RAG-powered code assistant. Ask it **anything** — it chats normally, and for "
    "questions about **the Claudio app itself, its architecture, and its codebase** it "
    "retrieves the relevant source and answers with citations. You can also **paste your "
    "own code** in the sidebar and ask questions about that. Read-only demo; the full "
    "agentic CLI (with tools, memory, and planning) lives in the repo."
)

# Sidebar: let a visitor paste their own snippet and ask about it.
with st.sidebar:
    st.header("Ask about your own code")
    pasted_code = st.text_area(
        "Paste code here (optional)",
        height=220,
        placeholder="def greet(name):\n    return f'hi {name}'",
    )
    if pasted_code and pasted_code.strip():
        st.caption("Questions will be answered about this pasted code.")

EXAMPLES = [
    "How does the semantic cache work?",
    "How is short-term memory implemented?",
    "What does the tree-sitter code parser do?",
    "How are LLM providers selected?",
]


@st.cache_resource(show_spinner="Indexing the Claudio codebase…")
def _load():
    """Build the client + index once per app instance (cached across reruns)."""
    client = rag.get_client()
    index = rag.build_index(client)
    return client, index


# Fail clearly if the proxy isn't configured, instead of a cryptic stack trace.
try:
    client, index = _load()
except Exception as e:  # noqa: BLE001
    st.error(
        "Demo is not configured. The owner needs to set `CLAUDIO_API_BASE` and "
        f"`CLAUDIO_API_KEY` in the app secrets.\n\nDetails: {e}"
    )
    st.stop()

st.success(f"Indexed {index.size} code chunks — ask away.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Example prompts (only before the first question).
if not st.session_state.messages:
    st.write("**Try one:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- `{s.path}` — **{s.name}** ({s.type}, lines {s.start_line}–{s.end_line})"
                    )

prompt = st.chat_input("Ask about the Claudio codebase…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            try:
                reply, sources = rag.answer(client, index, prompt, pasted_code=pasted_code)
            except Exception as e:  # noqa: BLE001
                reply, sources = f"Sorry — something went wrong: {e}", []
        st.markdown(reply)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"- `{s.path}` — **{s.name}** ({s.type}, lines {s.start_line}–{s.end_line})"
                    )

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
