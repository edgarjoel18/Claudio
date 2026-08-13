# Claudio LLM proxy

An OpenAI-compatible gateway that holds your **real** OpenAI key server-side, so
downloaded copies of the Claudio CLI never ship a secret. Clients authenticate
with a revocable **access token**; the proxy enforces a rate limit and a hard
monthly spend cap, then forwards to OpenAI with the real key.

## Why this exists

You want a recruiter to install the CLI and use it **without their own API key**,
with **you paying**. Shipping your key inside the download would expose it to
anyone. Instead the key lives here, on a server you control.

> **The access token is not your real protection.** Treat every issued token as
> public. Your real safety net is (1) the `MONTHLY_USD_CAP` here and (2) a hard
> **usage limit set in your OpenAI billing dashboard**. Set both.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat
- `POST /v1/embeddings` — OpenAI-compatible embeddings
- `GET /health` — status + current month's spend vs. cap

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env with your real key + a token
export $(grep -v '^#' .env | xargs)   # load env vars
uvicorn main:app --reload
```

Smoke-test it:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer demo-token-1" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"say hi"}]}'
```

## Deploy to Render (free tier)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, select the repo (it reads `backend/render.yaml`).
3. In the service's **Environment** tab set the secrets:
   - `OPENAI_API_KEY` = your real key
   - `PROXY_ACCESS_TOKENS` = one or more tokens, comma-separated
     (generate: `python -c "import secrets; print(secrets.token_urlsafe(24))"`)
4. Deploy. Your proxy URL will be like `https://claudio-proxy.onrender.com`.
5. Confirm: `curl https://claudio-proxy.onrender.com/health`.

Then set a **hard monthly limit** at
<https://platform.openai.com/settings/organization/limits> as the ultimate backstop.

## Config reference

| Env var | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — (required) | Your real key; server-side only |
| `PROXY_ACCESS_TOKENS` | — | Comma-separated client tokens; rotate to revoke |
| `MONTHLY_USD_CAP` | `20` | Proxy stops forwarding past this monthly spend |
| `RATE_LIMIT_PER_MIN` | `20` | Per-token requests/minute |
| `ALLOWED_MODELS` | `gpt-4o,gpt-4o-mini,text-embedding-3-small` | Model allowlist |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream base URL |
| `PRICES_JSON` | built-in table | Override pricing for spend estimates |
| `USAGE_FILE` | `/tmp/claudio_proxy_usage.json` | Where the monthly total is persisted |

> Note: Render's free tier has ephemeral disk and sleeps when idle, so the
> proxy-side spend counter can reset on restart. The OpenAI account-level hard
> limit is what guarantees you never overspend.
