"""
SCM Assistant — FastAPI Backend
================================
Embeddings  : Ollama Cloud API  (gemma4:31b-cloud via /api/chat — prompt-based)
LLM         : Ollama Cloud API  (gemma4:31b-cloud via /api/chat)
Vector Store: Qdrant Cloud      (free tier)
Data        : CSV + PDF uploaded via /ingest endpoints
"""

from __future__ import annotations

import os, io, json, time, logging, hashlib, re
from typing import Optional

import pandas as pd
import requests
import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────── ENV ───────────────────
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
_OLLAMA_BASE    = OLLAMA_HOST if OLLAMA_HOST else "https://ollama.com"
OLLAMA_CHAT_URL = f"{_OLLAMA_BASE}/api/chat"

CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gemma4:31b-cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemma4:31b-cloud")

QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "scm_docs"

# Fixed at 256 — the embed prompt asks for exactly 256 floats.
# DO NOT auto-detect at startup (causes race condition on Render cold boot).
EMBED_DIM = 256

# ─────────────────── AUTH ───────────────────
def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        h["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return h

# ─────────────────── QDRANT CLIENT ───────────────────
_qclient: Optional[QdrantClient] = None

def get_qdrant() -> QdrantClient:
    global _qclient
    if _qclient is None:
        if QDRANT_URL and QDRANT_API_KEY:
            _qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _qclient = QdrantClient(":memory:")
    return _qclient


def _ensure_collection():
    """
    Create collection if missing.
    If it exists with wrong dim — delete and recreate.
    Called lazily on first ingest/chat, NOT at startup.
    """
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        stored_dim = info.config.params.vectors.size
        if stored_dim == EMBED_DIM:
            log.info("Collection '%s' OK (dim=%d)", COLLECTION_NAME, EMBED_DIM)
            return
        log.warning(
            "Collection '%s' dim=%d != EMBED_DIM=%d — recreating.",
            COLLECTION_NAME, stored_dim, EMBED_DIM
        )
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    log.info("Created collection '%s' dim=%d", COLLECTION_NAME, EMBED_DIM)


# ─────────────────── EMBED VIA CHAT ───────────────────
_EMBED_SYSTEM = (
    f"You are an embedding engine. "
    f"When given text, respond with ONLY a JSON array of exactly {EMBED_DIM} "
    f"floating-point numbers between -1.0 and 1.0 representing the semantic meaning. "
    f"No explanation, no markdown — ONLY the JSON array."
)

import time as _time

def _embed_one(text: str, retries: int = 5) -> list[float]:
    """
    Call Ollama Cloud chat API to get embeddings.
    Retries up to `retries` times with exponential backoff on:
      - 429 (rate limit)
      - 503 / 502 (server overload)
      - Timeout
      - Bad JSON / empty vector
    """
    body = {
        "model":  EMBED_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _EMBED_SYSTEM},
            {"role": "user",   "content": f"Embed this text:\n{text[:2000]}"},
        ],
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 4096,
        },
    }

    last_err = None
    for attempt in range(retries):
        wait = 2 ** attempt          # 1s, 2s, 4s, 8s, 16s
        try:
            r = requests.post(OLLAMA_CHAT_URL, json=body, headers=_headers(), timeout=120)

            if r.status_code == 401:
                raise RuntimeError("Ollama 401 Unauthorized — check OLLAMA_API_KEY")
            if r.status_code == 404:
                raise RuntimeError(f"Model '{EMBED_MODEL}' not found on Ollama Cloud")
            if r.status_code in (429, 502, 503):
                log.warning("Ollama HTTP %d on attempt %d — retrying in %ds", r.status_code, attempt+1, wait)
                _time.sleep(wait)
                last_err = f"HTTP {r.status_code}"
                continue

            r.raise_for_status()
            raw = r.json().get("message", {}).get("content", "[]")

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\[.*?\]', raw, re.DOTALL)
                parsed = json.loads(m.group()) if m else []

            if isinstance(parsed, dict):
                parsed = (
                    parsed.get("embedding")
                    or parsed.get("values")
                    or parsed.get("vector")
                    or list(parsed.values())[0]
                )

            vec = [float(v) for v in parsed]

            if len(vec) == 0:
                log.warning("Empty vector on attempt %d — retrying in %ds", attempt+1, wait)
                _time.sleep(wait)
                last_err = "empty vector"
                continue

            # Pad or trim to exactly EMBED_DIM
            if len(vec) < EMBED_DIM:
                vec.extend([0.0] * (EMBED_DIM - len(vec)))
            elif len(vec) > EMBED_DIM:
                vec = vec[:EMBED_DIM]

            # L2-normalise
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]

            return vec   # success

        except RuntimeError:
            raise   # 401/404 — don't retry, always fatal
        except requests.Timeout:
            log.warning("Ollama timeout on attempt %d — retrying in %ds", attempt+1, wait)
            _time.sleep(wait)
            last_err = "timeout"
        except Exception as e:
            log.warning("Ollama error on attempt %d: %s — retrying in %ds", attempt+1, e, wait)
            _time.sleep(wait)
            last_err = str(e)

    raise RuntimeError(f"_embed_one failed after {retries} attempts: {last_err}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [_embed_one(t) for t in texts]


def embed_query(text: str) -> list[float]:
    return _embed_one(text)


# ─────────────────── CHUNKING ───────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def csv_to_chunks(df: pd.DataFrame, chunk_rows: int = 20) -> list[dict]:
    chunks = []
    for start in range(0, len(df), chunk_rows):
        batch = df.iloc[start : start + chunk_rows]
        lines = []
        for _, row in batch.iterrows():
            lines.append(
                f"PO {row['PO_ID']} | {row['Supplier_Name']} ({row['Supplier_ID']}) | "
                f"Region:{row['Region']} | Tier:{row['Contract_Tier']} | "
                f"OTD:{row['OTD_Rate_Pct']}% | Defect:{row['Defect_Rate_Pct']}% | "
                f"Compliance:{row['Compliance_Score']} | Risk:{row['Risk_Level']} | "
                f"Disruption:{row['Active_Disruptions']} | "
                f"Sustainability:{row['Sustainability_Score']} | "
                f"PO_Value:${row['PO_Value_USD']} | Quarter:{row['PO_Quarter']}"
            )
        chunks.append({
            "text":   "\n".join(lines),
            "source": "csv",
            "rows":   f"{start}-{start+len(batch)-1}",
        })
    return chunks


# ─────────────────── INGEST ───────────────────
def upsert_chunks(chunks: list[dict], source_tag: str) -> int:
    client  = get_qdrant()
    total   = 0
    skipped = 0
    for i, chunk in enumerate(chunks):
        try:
            vec = _embed_one(chunk["text"])   # has built-in retry
        except RuntimeError as e:
            if "401" in str(e) or "not found" in str(e):
                raise   # fatal — bubble up immediately
            log.error("Skipping chunk %d/%d after retries: %s", i+1, len(chunks), e)
            skipped += 1
            continue    # skip this chunk, keep going

        uid = int(hashlib.md5(f"{source_tag}{i}".encode()).hexdigest(), 16) % (10**12)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=uid,
                vector=vec,
                payload={**chunk, "source_tag": source_tag},
            )],
        )
        total += 1
        log.info("Embedded chunk %d/%d  [%s] (skipped=%d)", i+1, len(chunks), source_tag, skipped)

    if skipped:
        log.warning("Ingest complete — %d chunks embedded, %d skipped due to errors", total, skipped)
    return total


# ─────────────────── RETRIEVAL ───────────────────
def retrieve(query: str, top_k: int = 6) -> list[str]:
    q_vec = embed_query(query)
    hits  = get_qdrant().search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [h.payload.get("text", "") for h in hits]


# ─────────────────── LLM CHAT ───────────────────
SYSTEM_PROMPT = """You are SCM Assistant, an expert supply chain analyst for BQBYTE Technologies.
Answer questions using ONLY the context provided. Be precise, cite supplier names/IDs.
If the answer isn't in the context, say so clearly. Do not hallucinate numbers."""

def llm_chat(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    body = {
        "model":  CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 2048},
    }
    r = requests.post(OLLAMA_CHAT_URL, json=body, headers=_headers(), timeout=120)
    if r.status_code == 401:
        raise RuntimeError("Ollama 401 — check OLLAMA_API_KEY")
    if r.status_code == 404:
        raise RuntimeError(f"Chat model '{CHAT_MODEL}' not found")
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "No response.")


# ─────────────────── FASTAPI ───────────────────
app = FastAPI(title="SCM Assistant API", version="1.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

ingest_status = {
    "csv":  {"status": "idle", "chunks": 0, "message": ""},
    "pdf":  {"status": "idle", "chunks": 0, "message": ""},
    "total_chunks": 0,
}


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "embed_model": EMBED_MODEL,
        "chat_model":  CHAT_MODEL,
        "ollama_url":  OLLAMA_CHAT_URL,
        "embed_dim":   EMBED_DIM,
    }


@app.get("/status")
def status():
    try:
        count = get_qdrant().get_collection(COLLECTION_NAME).points_count
    except Exception:
        count = -1
    return {**ingest_status, "qdrant_points": count}


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...), chunk_rows: int = 20):
    ingest_status["csv"]["status"]  = "processing"
    ingest_status["csv"]["message"] = "Reading CSV…"
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        ingest_status["csv"]["message"] = f"Loaded {len(df)} rows. Chunking…"
        chunks = csv_to_chunks(df, chunk_rows=chunk_rows)
        ingest_status["csv"]["message"] = f"{len(chunks)} chunks. Embedding…"
        _ensure_collection()           # create/fix collection right before upsert
        n = upsert_chunks(chunks, "csv")
        ingest_status["csv"] = {"status": "done", "chunks": n, "message": f"✓ {n} chunks"}
        ingest_status["total_chunks"] = ingest_status["csv"]["chunks"] + ingest_status["pdf"]["chunks"]
        return {"ok": True, "chunks_upserted": n, "rows": len(df)}
    except Exception as e:
        ingest_status["csv"] = {"status": "error", "chunks": 0, "message": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), chunk_size: int = 400, overlap: int = 80):
    ingest_status["pdf"]["status"]  = "processing"
    ingest_status["pdf"]["message"] = "Reading PDF…"
    try:
        raw = await file.read()
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        doc.close()
        chunks = [{"text": c, "source": "pdf"} for c in chunk_text(text, chunk_size, overlap)]
        ingest_status["pdf"]["message"] = f"{len(chunks)} chunks. Embedding…"
        _ensure_collection()           # create/fix collection right before upsert
        n = upsert_chunks(chunks, "pdf")
        ingest_status["pdf"] = {"status": "done", "chunks": n, "message": f"✓ {n} chunks"}
        ingest_status["total_chunks"] = ingest_status["csv"]["chunks"] + ingest_status["pdf"]["chunks"]
        return {"ok": True, "chunks_upserted": n}
    except Exception as e:
        ingest_status["pdf"] = {"status": "error", "chunks": 0, "message": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    question: str
    top_k: int = 6


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    t0     = time.time()
    chunks = retrieve(req.question, top_k=req.top_k)
    answer = llm_chat(req.question, chunks)
    return {"answer": answer, "sources_used": len(chunks), "latency_s": round(time.time()-t0, 2)}


@app.post("/collection/reset")
def reset_collection():
    """POST instead of DELETE — works on all proxies including Render."""
    client = get_qdrant()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass   # already gone — that's fine
    _ensure_collection()
    ingest_status["csv"]  = {"status": "idle", "chunks": 0, "message": ""}
    ingest_status["pdf"]  = {"status": "idle", "chunks": 0, "message": ""}
    ingest_status["total_chunks"] = 0
    return {"ok": True, "message": f"Collection reset with dim={EMBED_DIM}"}
