"""
SCM Assistant — Streamlit Frontend
====================================
Tabs:
  1. 📤 Data Ingest  — upload CSV + PDF, show dataflow, chunk config
  2. 💬 Chat         — RAG chatbot with sample questions
  3. 📊 Dashboard    — quick stats from the status endpoint
"""

import time
import requests
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd

# ─────────────────── LOCALSTORAGE BRIDGE ───────────────────
import json as _json

# localStorage key
_LS_KEY = "scm_chat_history"

def _save_to_ls(messages: list):
    """Write chat history to browser localStorage via injected JS."""
    payload = _json.dumps(messages, ensure_ascii=True)
    # Use single quotes around the JSON string so double-quotes inside don't break JS
    js = f"try{{localStorage.setItem('{_LS_KEY}', JSON.stringify({payload}));}}catch(e){{}}"
    _components.html(f"<script>{js}</script>", height=0)

def _clear_ls():
    """Remove chat history from localStorage."""
    _components.html(
        f"<script>try{{localStorage.removeItem('{_LS_KEY}');}}catch(e){{}}</script>",
        height=0,
    )

def _inject_ls_loader():
    """
    On first render: read localStorage and push it into ?_hist= query param,
    triggering a Streamlit reload that populates session_state.
    Guard: skip if ?_hist already present (already loaded this session).
    """
    _components.html(f"""
    <script>
    (function(){{
      if (new URLSearchParams(window.location.search).get('_hist')) return;
      var raw = null;
      try {{ raw = localStorage.getItem('{_LS_KEY}'); }} catch(e) {{}}
      if (!raw) return;
      var url = new URL(window.location.href);
      url.searchParams.set('_hist', encodeURIComponent(raw));
      window.location.replace(url.toString());
    }})();
    </script>
    """, height=0)



# ─────────────────── CONFIG ───────────────────
API_BASE = "http://localhost:8000"   # FastAPI on same dyno

st.set_page_config(
    page_title="SCM Assistant",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── CUSTOM CSS ───────────────────
st.markdown("""
<style>
  /* ── Palette: deep navy + electric teal + warm amber ── */
  :root {
    --navy:   #0D1B2A;
    --teal:   #00C9B1;
    --amber:  #F5A623;
    --slate:  #1E2D3D;
    --light:  #E8EDF2;
    --muted:  #7A8FA6;
  }

  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    color: var(--light) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--slate) !important;
    border-right: 1px solid #243447;
  }

  /* Headers */
  h1 { color: var(--teal) !important; letter-spacing: -0.5px; }
  h2, h3 { color: var(--light) !important; }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--slate);
    border-radius: 10px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-weight: 500;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: var(--teal) !important;
    color: var(--navy) !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--teal) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* Danger button */
  .danger-btn > button {
    background: #C0392B !important;
    color: white !important;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--slate);
    border: 1px solid #243447;
    border-radius: 10px;
    padding: 12px;
  }
  [data-testid="metric-container"] label { color: var(--muted) !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-size: 1.8rem !important;
  }

  /* Chat bubbles */
  .user-bubble {
    background: #1E3A5F;
    border-left: 3px solid var(--teal);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    color: var(--light);
  }
  .bot-bubble {
    background: var(--slate);
    border-left: 3px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    color: var(--light);
  }
  .meta-tag {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Flow step cards */
  .flow-step {
    background: var(--slate);
    border: 1px solid #243447;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    position: relative;
  }
  .flow-step .step-num {
    font-size: 11px;
    font-weight: 700;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .flow-step .step-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--light);
  }
  .flow-step .step-desc {
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
  }
  .status-done  { border-left: 4px solid var(--teal) !important; }
  .status-proc  { border-left: 4px solid var(--amber) !important; }
  .status-idle  { border-left: 4px solid #243447 !important; }
  .status-error { border-left: 4px solid #E74C3C !important; }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--slate) !important;
    color: var(--light) !important;
    border-radius: 8px !important;
  }

  /* Input */
  .stTextInput input, .stTextArea textarea {
    background: var(--slate) !important;
    color: var(--light) !important;
    border: 1px solid #243447 !important;
    border-radius: 8px !important;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: var(--slate) !important;
    border: 2px dashed #243447 !important;
    border-radius: 10px !important;
  }

  /* Divider */
  hr { border-color: #243447 !important; }

  /* Select / slider */
  .stSelectbox div, .stSlider { color: var(--light) !important; }

  /* Progress */
  .stProgress > div > div { background: var(--teal) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: #243447; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────── HELPERS ───────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path: str, timeout: int = 300, **kwargs):
    # /chat needs a long timeout — Ollama Cloud gemma4:31b can take 2-4 min
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.Timeout:
        return None, f"Request timed out after {timeout}s — Ollama Cloud may be slow, try again."
    except Exception as e:
        err = str(e)
        try:
            detail = e.response.json().get("detail", "")
            if detail:
                err = detail
        except Exception:
            pass
        return None, err

def status_badge(s: str) -> str:
    icons = {"done": "✅", "processing": "⏳", "error": "❌", "idle": "⬜"}
    return icons.get(s, "⬜")

def flow_step(num: str, title: str, desc: str, status: str = "idle"):
    cls = f"status-{status[:4]}"
    st.markdown(f"""
    <div class="flow-step {cls}">
      <div class="step-num">Step {num}</div>
      <div class="step-title">{status_badge(status)} {title}</div>
      <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────── SIDEBAR ───────────────────
with st.sidebar:
    st.markdown("## 🔗 SCM Assistant")
    st.markdown("*Supply Chain RAG Chatbot*")
    st.markdown("---")

    status_data, err = api_get("/status")
    if status_data:
        csv_s = status_data["csv"]["status"]
        pdf_s = status_data["pdf"]["status"]
        st.markdown(f"**CSV** {status_badge(csv_s)} `{csv_s}` — {status_data['csv']['chunks']} chunks")
        st.markdown(f"**PDF** {status_badge(pdf_s)} `{pdf_s}` — {status_data['pdf']['chunks']} chunks")
        st.markdown(f"**Qdrant points:** `{status_data.get('qdrant_points', '?')}`")
    else:
        st.warning("API not reachable yet…")

    st.markdown("---")
    st.markdown("**Models**")
    health, _ = api_get("/health")
    if health:
        st.code(f"Embed : {health['embed_model']}\nChat  : {health['chat_model']}", language="")
    st.markdown("---")
    with st.expander("⚠️ Reset Collection"):
        if st.button("🗑️ Reset All Vectors", key="reset"):
            r, e = api_post("/collection/reset")
            if r:
                st.success("Collection reset")
            else:
                st.error(e)


# ─────────────────── MAIN TABS ───────────────────
tab1, tab2, tab3 = st.tabs(["📤  Data Ingest & Dataflow", "💬  Chat", "📊  Dashboard"])


# ══════════════════════════════════════════════════
# TAB 1 — DATA INGEST
# ══════════════════════════════════════════════════
with tab1:
    st.markdown("## Data Ingest Pipeline")
    st.markdown("Upload your data files, configure chunking, and embed into Qdrant. Watch each stage update in real-time.")

    # ── Dataflow Diagram ──
    st.markdown("### Pipeline Architecture")
    stat, _ = api_get("/status")

    csv_status = stat["csv"]["status"] if stat else "idle"
    pdf_status = stat["pdf"]["status"] if stat else "idle"

    col_flow = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1, 0.15, 1])

    def flow_box(col, icon, title, subtitle, highlight=False):
        bg = "#00C9B1" if highlight else "#1E2D3D"
        fg = "#0D1B2A" if highlight else "#E8EDF2"
        sub_color = "#0D1B2A" if highlight else "#7A8FA6"
        col.markdown(f"""
        <div style="background:{bg};border-radius:10px;padding:16px 12px;text-align:center;
                    border:1px solid #243447;min-height:90px;">
          <div style="font-size:22px">{icon}</div>
          <div style="font-weight:700;color:{fg};font-size:13px;margin-top:4px">{title}</div>
          <div style="font-size:11px;color:{sub_color};margin-top:2px">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    def arrow(col):
        col.markdown("<div style='text-align:center;padding-top:32px;font-size:20px;color:#7A8FA6'>→</div>",
                     unsafe_allow_html=True)

    flow_box(col_flow[0], "📁", "Raw Files", "CSV + PDF")
    arrow(col_flow[1])
    flow_box(col_flow[2], "✂️", "Chunker", "Text splitting")
    arrow(col_flow[3])
    flow_box(col_flow[4], "🤖", "Ollama Embed", "nomic-embed-text",
             highlight=(csv_status=="done" or pdf_status=="done"))
    arrow(col_flow[5])
    flow_box(col_flow[6], "🗄️", "Qdrant Cloud", "Vector store",
             highlight=(csv_status=="done" or pdf_status=="done"))
    arrow(col_flow[7])
    flow_box(col_flow[8], "💬", "RAG Chat", "LLM + retrieval",
             highlight=(csv_status=="done" or pdf_status=="done"))

    st.markdown("---")

    # ── Two-column upload ──
    left, right = st.columns(2)

    # ─── CSV upload ───
    with left:
        st.markdown("### 📊 Supplier Performance CSV")

        with st.expander("⚙️ Chunk Configuration", expanded=True):
            chunk_rows = st.slider("Rows per chunk", 5, 50, 20, 5,
                help="Each chunk = N rows of PO data converted to text")
            est_chunks = st.empty()

        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

        if csv_file:
            try:
                df_preview = pd.read_csv(csv_file)
                csv_file.seek(0)
                n_rows  = len(df_preview)
                n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
                est_chunks.info(f"📦 ~{n_chunks} chunks from {n_rows} rows")

                st.dataframe(df_preview.head(5), use_container_width=True,
                             hide_index=True)

                if st.button("🚀 Embed CSV", key="embed_csv"):
                    with st.spinner("Embedding CSV into Qdrant…"):
                        result, err = api_post(
                            f"/ingest/csv?chunk_rows={chunk_rows}",
                            files={"file": (csv_file.name, csv_file.getvalue(), "text/csv")},
                        )
                    if result:
                        st.success(f"✅ {result['chunks_upserted']} chunks embedded from {result['rows']} rows")
                        st.rerun()
                    else:
                        st.error(f"Error: {err}")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        # Status card
        if stat:
            s = stat["csv"]
            cls = "status-" + s["status"][:4]
            st.markdown(f"""
            <div class="flow-step {cls}">
              <div class="step-num">CSV Status</div>
              <div class="step-title">{status_badge(s['status'])} {s['status'].title()}</div>
              <div class="step-desc">{s['message'] or 'Not started'}</div>
            </div>
            """, unsafe_allow_html=True)

    # ─── PDF upload ───
    with right:
        st.markdown("### 📄 Governance Policy PDF")

        with st.expander("⚙️ Chunk Configuration", expanded=True):
            chunk_size = st.slider("Words per chunk", 100, 800, 400, 50,
                help="Larger = more context per chunk but fewer results retrieved")
            overlap    = st.slider("Overlap (words)", 0, 200, 80, 20,
                help="Overlap between consecutive chunks for context continuity")

        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

        if pdf_file:
            st.info(f"📄 {pdf_file.name} — {pdf_file.size // 1024} KB")

            if st.button("🚀 Embed PDF", key="embed_pdf"):
                with st.spinner("Extracting text and embedding…"):
                    result, err = api_post(
                        f"/ingest/pdf?chunk_size={chunk_size}&overlap={overlap}",
                        files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
                    )
                if result:
                    st.success(f"✅ {result['chunks_upserted']} chunks embedded")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")

        # Status card
        if stat:
            s = stat["pdf"]
            cls = "status-" + s["status"][:4]
            st.markdown(f"""
            <div class="flow-step {cls}">
              <div class="step-num">PDF Status</div>
              <div class="step-title">{status_badge(s['status'])} {s['status'].title()}</div>
              <div class="step-desc">{s['message'] or 'Not started'}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    # ── Step-by-step pipeline trace ──
    st.markdown("### Pipeline Steps")
    c1, c2 = st.columns(2)
    with c1:
        flow_step("1", "File Upload", "CSV or PDF uploaded via browser", "done" if (csv_status=="done" or pdf_status=="done") else "idle")
        flow_step("2", "Text Extraction", "CSV → row-text | PDF → PyMuPDF text", "done" if (csv_status=="done" or pdf_status=="done") else "idle")
        flow_step("3", "Chunking", f"CSV: {chunk_rows} rows/chunk | PDF: {chunk_size}w overlap {overlap}", "done" if (csv_status=="done" or pdf_status=="done") else "idle")
    with c2:
        flow_step("4", "Ollama Cloud Embed", f"nomic-embed-text → 768-dim vectors", "done" if (csv_status=="done" or pdf_status=="done") else "idle")
        flow_step("5", "Qdrant Upsert", "Vectors + metadata stored in cloud collection", "done" if (csv_status=="done" or pdf_status=="done") else "idle")
        flow_step("6", "RAG Ready", "Chatbot can now answer questions", "done" if (csv_status=="done" or pdf_status=="done") else "idle")


# ══════════════════════════════════════════════════
# TAB 2 — CHATBOT
# ══════════════════════════════════════════════════
with tab2:
    st.markdown("## 💬 Supply Chain Chatbot")

    # ── Load history from localStorage on first load ──
    _inject_ls_loader()   # injects JS; on reload populates ?_hist= query param

    if "messages" not in st.session_state:
        # Try to restore from query param (set by the JS bridge above)
        raw_hist = st.query_params.get("_hist", "")
        if raw_hist:
            try:
                import urllib.parse as _up
                st.session_state.messages = _json.loads(_up.unquote(raw_hist))
                # Clean the URL so reloads don't re-inject
                st.query_params.clear()
            except Exception:
                st.session_state.messages = []
        else:
            st.session_state.messages = []

    # Sample questions
    st.markdown("### 📋 Sample Questions")
    sample_qs = [
        "Which Tier-3 suppliers have an active disruption flag, and what response level applies per policy?",
        "Which suppliers qualify for the annual Volume Rebate Program and how many are there?",
        "Which region has the highest total PO value, and does it breach the concentration limit?",
        "Which suppliers are on Supplier Watch List (SWL) status and what does it restrict?",
        "Which product category has the highest average defect rate and does it exceed the Tier-2 limit?",
    ]

    cols = st.columns(3)
    for i, q in enumerate(sample_qs):
        if cols[i % 3].button(f"Q{i+1}", key=f"sq_{i}", use_container_width=True,
                              help=q):
            st.session_state.pending_q = q

    st.markdown("---")

    # Input FIRST — so the key is registered before we read session state below
    question = st.text_input(
        "Ask a question about your supply chain…",
        value=st.session_state.pop("pending_q", ""),
        key="chat_input",
        placeholder="e.g. Which suppliers have active disruption flags?",
    )

    # Chat history — rendered after input so it always shows latest messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 <strong>You</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 <strong>SCM Assistant</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)
            if msg.get("meta"):
                st.markdown(f'<div class="meta-tag">{msg["meta"]}</div>', unsafe_allow_html=True)

    col_send, col_clear = st.columns([4, 1])
    send_btn  = col_send.button("Send ➤", key="send_btn")
    clear_btn = col_clear.button("Clear", key="clear_btn")

    if clear_btn:
        st.session_state.messages = []
        _clear_ls()
        st.rerun()

    if send_btn and question.strip():
        # Store question immediately, clear input via flag
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state["_last_q"] = question

        # Show spinner and call API — do NOT rerun until after storing the answer
        with st.spinner("⏳ Querying Ollama Cloud… (may take 30-90s for gemma4:31b)"):
            result, err = api_post("/chat", timeout=360, json={"question": question, "top_k": 6})

        if result:
            meta = f"⏱ {result['latency_s']}s · {result['sources_used']} chunks retrieved"
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "meta": meta,
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {err}",
            })
        # Persist to browser localStorage before rerun
        _save_to_ls(st.session_state.messages)
        # Only rerun AFTER answer is stored — this re-renders chat history
        st.rerun()


# ══════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 System Dashboard")
    st.button("🔄 Refresh", key="refresh_dash")

    stat, err = api_get("/status")
    health, _ = api_get("/health")

    if stat:
        st.markdown("### Ingest Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CSV Chunks", stat["csv"]["chunks"])
        m2.metric("PDF Chunks", stat["pdf"]["chunks"])
        m3.metric("Total Vectors", stat.get("qdrant_points", "?"))
        m4.metric("Collection", "scm_docs")

        st.markdown("### File Status")
        rows = [
            {"File": "supplier_performance_data.csv", "Status": stat["csv"]["status"],
             "Chunks": stat["csv"]["chunks"], "Message": stat["csv"]["message"]},
            {"File": "SupplyChain_Governance_Policy.pdf", "Status": stat["pdf"]["status"],
             "Chunks": stat["pdf"]["chunks"], "Message": stat["pdf"]["message"]},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if health:
        st.markdown("### Model Configuration")
        c1, c2 = st.columns(2)
        c1.info(f"**Embedding Model:** {health['embed_model']}")
        c2.info(f"**Chat Model:** {health['chat_model']}")

    st.markdown("### About This Architecture")
    st.markdown("""
    | Component | Service | Notes |
    |-----------|---------|-------|
    | **Embeddings** | Ollama Cloud API | `nomic-embed-text` 768-dim |
    | **LLM** | Ollama Cloud API | `llama3.2` / `gemma` |
    | **Vector Store** | Qdrant Cloud | Free tier, cosine similarity |
    | **Backend** | FastAPI | `/ingest/csv`, `/ingest/pdf`, `/chat` |
    | **Frontend** | Streamlit | Upload UI + chatbot |
    | **Deployment** | Render | Single web service, 512 MB RAM |
    """)
