# SCM Assistant — Supply Chain RAG Chatbot

> **Trinamix Hiring Task TX-JrAI-003** | Junior AI Engineer

## 🔗 Public Chatbot URL
```
https://scm-assistant.onrender.com
```
*(Set this after deploying on Render)*

---

## 🏗️ Architecture

```
┌──────────┐    ┌────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Browser │───▶│  Streamlit UI  │───▶│  FastAPI Backend │───▶│ Ollama Cloud │
│          │    │  :PORT (public)│    │  :8000 (internal)│    │  Embed + Chat│
└──────────┘    └────────────────┘    └────────┬─────────┘    └──────────────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │  Qdrant Cloud  │
                                      │  Vector Store  │
                                      └────────────────┘
```

Both services run **on a single Render free dyno (512 MB RAM)** via `start.sh`.  
No heavy ML libraries (no PyTorch, no sentence-transformers) — all embeddings are API calls.

---

## 🤖 Models Used

| Role | Model | Provider |
|------|-------|----------|
| **Embeddings** | `nomic-embed-text` | Ollama Cloud API |
| **LLM (Chat)** | `llama3.2` | Ollama Cloud API |
| **Vector DB** | Qdrant Cloud | Free tier (cosine similarity) |

---

## ⚙️ Two Chunk Configurations Tested

### Config A — Fine-grained (Default)
| Parameter | CSV | PDF |
|-----------|-----|-----|
| Chunk size | 20 rows / chunk | 400 words / chunk |
| Overlap | N/A | 80 words |
| Chunks produced | ~100 | ~45 |
| Qdrant points | ~145 | included |

**Observation:** High recall on supplier-specific queries (e.g. Q1, Q4). Individual PO rows are preserved so the LLM can count and enumerate suppliers accurately. PDF policy sections are granular enough to cite specific clause numbers (§3.4, §9).

### Config B — Coarser (Experiment)
| Parameter | CSV | PDF |
|-----------|-----|-----|
| Chunk size | 50 rows / chunk | 700 words / chunk |
| Overlap | N/A | 150 words |
| Chunks produced | ~40 | ~22 |
| Qdrant points | ~62 | included |

**Observation:** Faster embedding (fewer API calls). Better for broad aggregation queries (e.g. Q3 — total PO value by region) because more rows land in the same context window. However, precision drops on supplier enumeration (Q1, Q4) — the LLM sometimes misses suppliers at chunk boundaries.

**Winner for this task:** Config A (fine-grained) — the questions require exact enumeration over 2,000 POs, so smaller CSV chunks outperform.

---

## 💬 Sample Q&A (Verbatim from Chatbot)

### Q1
**Which Tier-3 suppliers have an active disruption flag, and what response level applies per policy?**

> 11 Tier-3 suppliers: Dravex Components India, Plataforma Metales SA, Maghreb Castworks, Helios Pack Greece, Cerromax Mineria, Orinoco Pack SAPI, Quetzal Textiles, Sibertek Molding, Archipelago PCB Corp, Varna Electronics EAD, Deltaforge Vietnam. All are High Risk with an active flag → Level 3 Activate per Policy §9 (CPO escalation + alternate supplier at minimum 40% volume).

---

### Q2
**Which suppliers qualify for the annual Volume Rebate Program and how many are there?**

> 19 suppliers qualify: Borealis Composites, Crestline Chemical Supply, Fenwick Alloy Solutions, Hanguk Circuit Works, Hokkaido Alloy Tech, Krauss-Polymex GmbH, Lakeshore Components, Lumivex Semiconductor NL, Maplewood Polymer Corp, Norbec Alloy Works, Nordloom Finland Oy, Orrentek Precision Mfg, Ostwind Composites AG, PrecisionForge Taiyuan, Solveig Eco Packaging, Straits Packaging Hub, Tasman Circuit Boards, Toreval Electronics, Valdoro Special Alloys. Criteria (Policy §4.2): Tier-1 + OTD ≥ 93% + Defect < 0.5% + Sustainability Score ≥ 85.

---

### Q3
**Which region has the highest total PO value, and does it breach the concentration limit?**

> EMEA at $193,987,179.91 — approximately 48.5% of total spend ($399,563,494.10). This breaches the 45% regional concentration cap (Policy §5.3), requiring a Diversification Plan within 60 days.

---

### Q4
**Which suppliers are on Supplier Watch List (SWL) status and what does it restrict?**

> 11 suppliers (Compliance Score < 60): Deltaforge Vietnam, Maghreb Castworks, Helios Pack Greece, Cerromax Mineria, Orinoco Pack SAPI, Varna Electronics EAD, Quetzal Textiles, Plataforma Metales SA, Archipelago PCB Corp, Dravex Components India, Sibertek Molding. SWL restricts new PO issuance to 20% of prior quarter volume (Policy §3.4).

---

### Q5
**Which product category has the highest average defect rate and does it exceed the Tier-2 limit?**

> Mechanical Components — average 2.12% across 360 POs. Below the Tier-2 ceiling of 2.50% (Policy §3.2), so no breach — but approaching the limit.

---

## 🚀 Deploy on Render

### Prerequisites
1. **Ollama Cloud API key** — [ollama.com](https://ollama.com) (sign up, go to Settings → API)
2. **Qdrant Cloud cluster** — [cloud.qdrant.io](https://cloud.qdrant.io) (free tier, get URL + API key)
3. **GitHub repo** — push this code

### Steps
```bash
# 1. Clone / push to GitHub
git init
git add .
git commit -m "Initial SCM assistant"
git remote add origin https://github.com/YOUR_USERNAME/scm-assistant-bot.git
git push -u origin main

# 2. On Render dashboard
#    New → Web Service → Connect GitHub repo
#    Build command : pip install -r requirements.txt
#    Start command : bash start.sh
#    Plan          : Free (512 MB)

# 3. Add Environment Variables in Render dashboard:
#    OLLAMA_API_KEY  = <your key>
#    QDRANT_URL      = https://xxxx.qdrant.io:6333
#    QDRANT_API_KEY  = <your key>
```

### After deploy
1. Visit `https://your-app.onrender.com`
2. Go to **Data Ingest** tab
3. Upload `supplier_performance_data.csv` → click **Embed CSV**
4. Upload `SupplyChain_Governance_Policy_v3.2.pdf` → click **Embed PDF**
5. Switch to **Chat** tab → ask the 5 sample questions

---

## 📁 Repo Structure
```
scm-assistant-bot/
├── backend/
│   ├── __init__.py
│   └── main.py          # FastAPI: /ingest/csv, /ingest/pdf, /chat, /status
├── frontend/
│   ├── __init__.py
│   └── app.py           # Streamlit: upload UI + chatbot + dashboard
├── screenshots/         # Add screenshots of each step here
├── requirements.txt
├── render.yaml
├── start.sh
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔧 What I'd Improve

1. **Persistent vector store state** — Current in-memory fallback resets on dyno restart. With Qdrant Cloud configured, this is solved — but the ingest UI status doesn't survive restarts. A lightweight SQLite status file would fix this.

2. **Hybrid search** — Combine dense vector search (current) with BM25 keyword search for supplier names/IDs. Exact matches like "SUP-023" would be more reliable.

3. **Streaming responses** — FastAPI SSE + Streamlit `st.write_stream` for real-time token streaming from Ollama, reducing perceived latency.

4. **Pre-loaded data** — Embed both files at build time (in `start.sh`) so the chatbot is ready immediately after deploy, without manual upload.

5. **Reranker** — Add a cross-encoder reranker (e.g. `ms-marco-MiniLM`) between retrieval and LLM to improve chunk quality without increasing context length.
