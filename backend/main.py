"""
SCM Assistant — FastAPI Backend
================================
Embeddings  : Ollama Cloud API  (nomic-embed-text)
LLM         : Ollama Cloud API  (llama3 / gemma)
Vector Store: Qdrant Cloud      (free tier — no RAM used locally)
Data        : CSV + PDF uploaded via /ingest endpoints
"""

from __future__ import annotations

import os, io, json, time, logging, hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import fitz                          # PyMuPDF — lightweight PDF text extract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────── ENV ───────────────────
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_EMBED_URL  = "https://ollama.com/api/embed"
OLLAMA_CHAT_URL   = "https://ollama.com/api/chat"
EMBED_MODEL       = os.getenv("EMBED_MODEL",  "nomic-embed-text")
CHAT_MODEL        = os.getenv("CHAT_MODEL",   "llama3.2")

QDRANT_URL        = os.getenv("QDRANT_URL", "")      # e.g. https://xxx.qdrant.io:6333
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME   = "scm_docs"
EMBED_DIM         = 768                               # nomic-embed-text default

# ─────────────────── CLIENTS ───────────────────
def _qdrant() -> QdrantClient:
    if QDRANT_URL and QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # fallback: in-memory (won't persist between restarts — OK for demo)
    return QdrantClient(":memory:")

_qclient: Optional[QdrantClient] = None

def get_qdrant() -> QdrantClient:
    global _qclient
    if _qclient is None:
        _qclient = _qdrant()
        _ensure_collection(_qclient)
    return _qclient

def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection: %s", COLLECTION_NAME)

# ─────────────────── EMBED ───────────────────
_EMBED_HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}),
}

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call Ollama Cloud /api/embed — batched."""
    if not texts:
        return []
    payload = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(OLLAMA_EMBED_URL, json=payload, headers=_EMBED_HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"embeddings": [[...]]}
    return data.get("embeddings", [])

# ─────────────────── CHUNKING ───────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def csv_to_chunks(df: pd.DataFrame, chunk_rows: int = 20) -> list[dict]:
    """Convert each group of rows into a text chunk with metadata."""
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
            "text": "\n".join(lines),
            "source": "csv",
            "rows": f"{start}-{start+len(batch)-1}",
        })
    return chunks

# ─────────────────── INGEST ───────────────────
def upsert_chunks(chunks: list[dict], source_tag: str, client: QdrantClient) -> int:
    BATCH = 32
    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts  = [c["text"] for c in batch]
        vecs   = embed_texts(texts)
        points = []
        for j, (chunk, vec) in enumerate(zip(batch, vecs)):
            uid = int(hashlib.md5(f"{source_tag}{i+j}".encode()).hexdigest(), 16) % (10**12)
            points.append(PointStruct(
                id=uid,
                vector=vec,
                payload={**chunk, "source_tag": source_tag},
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
    return total

# ─────────────────── RETRIEVAL ───────────────────
def retrieve(query: str, top_k: int = 6) -> list[str]:
    client = get_qdrant()
    q_vec  = embed_texts([query])[0]
    hits   = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [h.payload.get("text", "") for h in hits]

# ─────────────────── LLM CHAT ───────────────────
_CHAT_HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}),
}

SYSTEM_PROMPT = """You are SCM Assistant, an expert supply chain analyst for BQBYTE Technologies.
Answer questions using ONLY the context provided. Be precise, cite supplier names/IDs.
If the answer isn't in the context, say so clearly. Do not hallucinate numbers."""

def llm_chat(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    payload = {"model": CHAT_MODEL, "messages": messages, "stream": False}
    r = requests.post(OLLAMA_CHAT_URL, json=payload, headers=_CHAT_HEADERS, timeout=120)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "No response.")

# ─────────────────── FASTAPI APP ───────────────────
app = FastAPI(title="SCM Assistant API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── State tracker (in-memory, resets on restart) ──
ingest_status = {
    "csv":  {"status": "idle", "chunks": 0, "message": ""},
    "pdf":  {"status": "idle", "chunks": 0, "message": ""},
    "total_chunks": 0,
}


@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL}


@app.get("/status")
def status():
    try:
        client = get_qdrant()
        info   = client.get_collection(COLLECTION_NAME)
        count  = info.points_count
    except Exception as e:
        count = -1
    return {**ingest_status, "qdrant_points": count}


@app.post("/ingest/csv")
async def ingest_csv(
    file: UploadFile = File(...),
    chunk_rows: int = 20,
):
    """Upload supplier_performance_data.csv and embed it."""
    ingest_status["csv"]["status"] = "processing"
    ingest_status["csv"]["message"] = "Reading CSV…"
    try:
        raw = await file.read()
        df  = pd.read_csv(io.BytesIO(raw))
        ingest_status["csv"]["message"] = f"Loaded {len(df)} rows. Chunking…"

        chunks = csv_to_chunks(df, chunk_rows=chunk_rows)
        ingest_status["csv"]["message"] = f"{len(chunks)} chunks created. Embedding…"

        client = get_qdrant()
        n = upsert_chunks(chunks, source_tag="csv", client=client)

        ingest_status["csv"]  = {"status": "done", "chunks": n, "message": f"✓ {n} chunks embedded"}
        ingest_status["total_chunks"] = (
            ingest_status["csv"]["chunks"] + ingest_status["pdf"]["chunks"]
        )
        return {"ok": True, "chunks_upserted": n, "rows": len(df), "chunk_rows": chunk_rows}

    except Exception as e:
        ingest_status["csv"]["status"] = "error"
        ingest_status["csv"]["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 400,
    overlap: int = 80,
):
    """Upload SupplyChain_Governance_Policy PDF and embed it."""
    ingest_status["pdf"]["status"] = "processing"
    ingest_status["pdf"]["message"] = "Reading PDF…"
    try:
        raw = await file.read()
        doc = fitz.open(stream=raw, filetype="pdf")
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()

        ingest_status["pdf"]["message"] = f"Extracted {len(full_text)} chars. Chunking…"
        raw_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        chunks = [{"text": c, "source": "pdf"} for c in raw_chunks]

        ingest_status["pdf"]["message"] = f"{len(chunks)} chunks created. Embedding…"
        client = get_qdrant()
        n = upsert_chunks(chunks, source_tag="pdf", client=client)

        ingest_status["pdf"]  = {"status": "done", "chunks": n, "message": f"✓ {n} chunks embedded"}
        ingest_status["total_chunks"] = (
            ingest_status["csv"]["chunks"] + ingest_status["pdf"]["chunks"]
        )
        return {"ok": True, "chunks_upserted": n, "chunk_size": chunk_size, "overlap": overlap}

    except Exception as e:
        ingest_status["pdf"]["status"] = "error"
        ingest_status["pdf"]["message"] = str(e)
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
    return {
        "answer":       answer,
        "sources_used": len(chunks),
        "latency_s":    round(time.time() - t0, 2),
    }


@app.delete("/collection/reset")
def reset_collection():
    """Delete and recreate the Qdrant collection."""
    global _qclient
    client = get_qdrant()
    client.delete_collection(COLLECTION_NAME)
    _ensure_collection(client)
    ingest_status["csv"]  = {"status": "idle", "chunks": 0, "message": ""}
    ingest_status["pdf"]  = {"status": "idle", "chunks": 0, "message": ""}
    ingest_status["total_chunks"] = 0
    return {"ok": True, "message": "Collection reset"}
