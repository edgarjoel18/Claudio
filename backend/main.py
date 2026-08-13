"""
Claudio LLM proxy — an OpenAI-compatible gateway that holds the *real* OpenAI
key server-side so downloaded copies of the Claudio CLI never ship a secret.

A client (the Claudio CLI, or a web demo) points its OpenAI base_url at this
service and authenticates with a short-lived ACCESS TOKEN. The proxy:

  1. validates the access token,
  2. enforces a per-token request rate limit,
  3. enforces a hard monthly USD spend cap (best-effort accounting from the
     `usage` field OpenAI returns), and
  4. forwards the call to OpenAI using the real key.

The access token is NOT your protection against a leak — the spend cap and the
account-level hard limit you set in the OpenAI billing dashboard are. Treat any
issued token as public and rotate it freely.
"""
from __future__ import annotations

import hmac
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration (all from environment — never hardcode the key)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]                      # required
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Comma-separated tokens you hand out. Rotate/revoke by editing this env var.
ACCESS_TOKENS = {
    t.strip() for t in os.getenv("PROXY_ACCESS_TOKENS", "").split(",") if t.strip()
}

# Hard monthly spend cap (USD). The proxy stops forwarding once exceeded.
MONTHLY_USD_CAP = float(os.getenv("MONTHLY_USD_CAP", "20"))

# Per-token request rate limit.
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

# Only these models may be requested (prevents switching to a pricey model).
ALLOWED_MODELS = {
    m.strip()
    for m in os.getenv(
        "ALLOWED_MODELS",
        "gpt-4o,gpt-4o-mini,text-embedding-3-small",
    ).split(",")
    if m.strip()
}

# Where to persist the running spend total (best-effort; disk may be ephemeral).
USAGE_FILE = Path(os.getenv("USAGE_FILE", "/tmp/claudio_proxy_usage.json"))

# Approximate USD per 1M tokens. Override via PRICES_JSON if OpenAI pricing moves.
# These drive the *soft* cap; the authoritative backstop is your OpenAI billing limit.
DEFAULT_PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}
PRICES = json.loads(os.getenv("PRICES_JSON", "")) if os.getenv("PRICES_JSON") else DEFAULT_PRICES

REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

app = FastAPI(title="Claudio LLM proxy")
_http: httpx.AsyncClient | None = None

# Per-token sliding window of request timestamps for rate limiting.
_req_times: dict[str, deque[float]] = defaultdict(deque)


# ---------------------------------------------------------------------------
# Spend tracking (persisted, monthly bucket)
# ---------------------------------------------------------------------------
def _current_month() -> str:
    return time.strftime("%Y-%m")


def _load_usage() -> dict:
    try:
        data = json.loads(USAGE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    if data.get("month") != _current_month():
        data = {"month": _current_month(), "usd": 0.0}  # new month resets the bucket
    return data


def _save_usage(data: dict) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(data))
    except OSError:
        pass  # best-effort; the OpenAI account cap is the real backstop


def _cost_for(model: str, usage: dict) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return (prompt * price["input"] + completion * price["output"]) / 1_000_000


def _record_cost(model: str, usage: dict) -> None:
    data = _load_usage()
    data["usd"] = round(data.get("usd", 0.0) + _cost_for(model, usage), 6)
    _save_usage(data)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def _check_auth(authorization: str | None) -> str:
    if not ACCESS_TOKENS:
        raise HTTPException(500, "Proxy misconfigured: no PROXY_ACCESS_TOKENS set")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    # constant-time comparison against each allowed token
    if not any(hmac.compare_digest(token, t) for t in ACCESS_TOKENS):
        raise HTTPException(401, "Invalid access token")
    return token


def _check_rate_limit(token: str) -> None:
    now = time.time()
    window = _req_times[token]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded — slow down")
    window.append(now)


def _check_budget() -> None:
    if _load_usage().get("usd", 0.0) >= MONTHLY_USD_CAP:
        raise HTTPException(
            402, f"Monthly demo budget of ${MONTHLY_USD_CAP:.2f} reached — try again next month"
        )


def _check_model(model: str | None) -> None:
    if model and ALLOWED_MODELS and model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Model '{model}' is not permitted on this demo proxy")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global _http
    _http = httpx.AsyncClient(base_url=OPENAI_BASE_URL, timeout=REQUEST_TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _http is not None:
        await _http.aclose()


@app.get("/health")
async def health() -> dict:
    usage = _load_usage()
    return {
        "status": "ok",
        "month": usage.get("month"),
        "spent_usd": usage.get("usd", 0.0),
        "cap_usd": MONTHLY_USD_CAP,
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------
async def _forward(path: str, body: dict) -> JSONResponse:
    assert _http is not None
    resp = await _http.post(
        path,
        json=body,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    )
    if resp.status_code >= 400:
        # surface OpenAI's error to the client without leaking the key
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    data = resp.json()
    model = data.get("model") or body.get("model")
    if isinstance(data.get("usage"), dict) and model:
        _record_cost(model, data["usage"])
    return JSONResponse(content=data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    token = _check_auth(authorization)
    _check_rate_limit(token)
    _check_budget()
    body = await request.json()
    _check_model(body.get("model"))
    body.pop("stream", None)  # force non-streaming so usage accounting is reliable
    return await _forward("/chat/completions", body)


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str | None = Header(None)):
    token = _check_auth(authorization)
    _check_rate_limit(token)
    _check_budget()
    body = await request.json()
    _check_model(body.get("model"))
    return await _forward("/embeddings", body)
