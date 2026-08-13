# Claudio — web demo

A read-only Streamlit chat UI that answers questions about the Claudio codebase
using retrieval-augmented generation. Visitors need **no API key and no install**
— it talks to the hosted [LLM proxy](../backend) (billed to the project owner).

## Safety

This demo is **read-only** on purpose. It reuses Claudio's tree-sitter chunker to
index the source, then does embed → retrieve → answer. It does **not** load the
agent's terminal / filesystem / MCP tools, so it cannot run commands or write
files. The full agentic tool-use lives in the CLI, where it runs on the user's
own machine.

## Run locally

```bash
cd web
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   then edit .streamlit/secrets.toml with your proxy URL + access token
streamlit run app.py
```

Opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → pick this repo.
3. Set **Main file path** to `web/app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   CLAUDIO_API_BASE = "https://claudio-proxy-x1y1.onrender.com/v1"
   CLAUDIO_API_KEY  = "your-proxy-access-token"
   ```
5. Deploy. You get a public `https://<name>.streamlit.app` link to share.

Streamlit Cloud auto-installs from `web/requirements.txt`.

## Configuration

| Secret / env | Default | Purpose |
| --- | --- | --- |
| `CLAUDIO_API_BASE` | — (required) | Proxy base URL, ending in `/v1` |
| `CLAUDIO_API_KEY` | — (required) | Proxy access token (not an OpenAI key) |
| `CHAT_MODEL` | `gpt-4o-mini` | Answer model |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `INDEX_DIR` | `../claudio` | Directory to index for the demo |
