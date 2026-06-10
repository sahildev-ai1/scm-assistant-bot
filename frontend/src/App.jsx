import { useState, useRef, useEffect } from "react";

const API_BASE = "https://scm-assistant-bot-1ex0.onrender.com";

const SAMPLE_QUESTIONS = [
  "Which Tier-3 suppliers have an active disruption flag?",
  "Which suppliers qualify for the annual Volume Rebate Program?",
  "Which region has the highest total PO value?",
  "Which suppliers are on Supplier Watch List (SWL) status?",
  "Which product category has the highest average defect rate?",
];

function TypingDots() {
  return (
    <div className="typing-dots">
      <span /><span /><span />
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "bot"}`}>
      <div className="bubble">
        <div className="sender">{isUser ? "You" : "SCM Assistant"}</div>
        <div className="text">{msg.content}</div>
        {msg.meta && <div className="meta">{msg.meta}</div>}
      </div>
    </div>
  );
}

function ChatTab() {
  const [messages, setMessages] = useState(() => {
    try { return JSON.parse(localStorage.getItem("scm_chat") || "[]"); } catch { return []; }
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    localStorage.setItem("scm_chat", JSON.stringify(messages));
  }, [messages]);

  const send = async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;
    setInput("");
    const userMsg = { role: "user", content: q };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: 6 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Server error");
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer,
        meta: `⏱ ${data.latency_s}s · ${data.sources_used} chunks retrieved`,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: `⚠️ Error: ${e.message}` }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-tab">
      <div className="samples">
        <p className="samples-label">Sample questions</p>
        <div className="samples-grid">
          {SAMPLE_QUESTIONS.map((q, i) => (
            <button key={i} className="sample-btn" onClick={() => send(q)}>
              {q}
            </button>
          ))}
        </div>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <div className="empty-icon">🔗</div>
            <p>Ask anything about your supply chain data</p>
          </div>
        )}
        {messages.map((m, i) => <ChatMessage key={i} msg={m} />)}
        {loading && (
          <div className="message bot">
            <div className="bubble"><TypingDots /></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
          placeholder="Ask about suppliers, disruptions, compliance…"
          disabled={loading}
        />
        <button onClick={() => send()} disabled={loading || !input.trim()} className="send-btn">
          {loading ? "…" : "Send"}
        </button>
        <button
          onClick={() => { setMessages([]); localStorage.removeItem("scm_chat"); }}
          className="clear-btn"
        >Clear</button>
      </div>
    </div>
  );
}

function IngestTab() {
  const [csvFile, setCsvFile] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [csvStatus, setCsvStatus] = useState(null);
  const [pdfStatus, setPdfStatus] = useState(null);
  const [csvLoading, setCsvLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [chunkRows, setChunkRows] = useState(20);
  const [chunkSize, setChunkSize] = useState(400);
  const [overlap, setOverlap] = useState(80);

  const uploadCsv = async () => {
    if (!csvFile) return;
    setCsvLoading(true);
    setCsvStatus({ type: "info", msg: "Uploading and embedding…" });
    try {
      const fd = new FormData();
      fd.append("file", csvFile);
      const res = await fetch(`${API_BASE}/ingest/csv?chunk_rows=${chunkRows}`, {
        method: "POST", body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setCsvStatus({ type: "success", msg: `✅ ${data.chunks_upserted} chunks from ${data.rows} rows` });
    } catch (e) {
      setCsvStatus({ type: "error", msg: `⚠️ ${e.message}` });
    }
    setCsvLoading(false);
  };

  const uploadPdf = async () => {
    if (!pdfFile) return;
    setPdfLoading(true);
    setPdfStatus({ type: "info", msg: "Extracting text and embedding…" });
    try {
      const fd = new FormData();
      fd.append("file", pdfFile);
      const res = await fetch(`${API_BASE}/ingest/pdf?chunk_size=${chunkSize}&overlap=${overlap}`, {
        method: "POST", body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setPdfStatus({ type: "success", msg: `✅ ${data.chunks_upserted} chunks embedded` });
    } catch (e) {
      setPdfStatus({ type: "error", msg: `⚠️ ${e.message}` });
    }
    setPdfLoading(false);
  };

  return (
    <div className="ingest-tab">
      <div className="ingest-grid">
        {/* CSV */}
        <div className="ingest-card">
          <h3>📊 Supplier Performance CSV</h3>
          <div className="field">
            <label>Rows per chunk: <strong>{chunkRows}</strong></label>
            <input type="range" min={5} max={100} step={5} value={chunkRows}
              onChange={e => setChunkRows(+e.target.value)} />
          </div>
          <div className="drop-zone" onClick={() => document.getElementById("csv-input").click()}>
            {csvFile ? <span>📄 {csvFile.name} ({(csvFile.size/1024).toFixed(1)} KB)</span>
              : <span>Click to upload CSV</span>}
          </div>
          <input id="csv-input" type="file" accept=".csv" hidden
            onChange={e => setCsvFile(e.target.files[0])} />
          <button onClick={uploadCsv} disabled={!csvFile || csvLoading} className="upload-btn">
            {csvLoading ? "Embedding…" : "Embed CSV"}
          </button>
          {csvStatus && <div className={`status-msg ${csvStatus.type}`}>{csvStatus.msg}</div>}
        </div>

        {/* PDF */}
        <div className="ingest-card">
          <h3>📄 Governance Policy PDF</h3>
          <div className="field">
            <label>Words per chunk: <strong>{chunkSize}</strong></label>
            <input type="range" min={100} max={800} step={50} value={chunkSize}
              onChange={e => setChunkSize(+e.target.value)} />
          </div>
          <div className="field">
            <label>Overlap: <strong>{overlap}</strong></label>
            <input type="range" min={0} max={200} step={20} value={overlap}
              onChange={e => setOverlap(+e.target.value)} />
          </div>
          <div className="drop-zone" onClick={() => document.getElementById("pdf-input").click()}>
            {pdfFile ? <span>📄 {pdfFile.name} ({(pdfFile.size/1024).toFixed(1)} KB)</span>
              : <span>Click to upload PDF</span>}
          </div>
          <input id="pdf-input" type="file" accept=".pdf" hidden
            onChange={e => setPdfFile(e.target.files[0])} />
          <button onClick={uploadPdf} disabled={!pdfFile || pdfLoading} className="upload-btn">
            {pdfLoading ? "Embedding…" : "Embed PDF"}
          </button>
          {pdfStatus && <div className={`status-msg ${pdfStatus.type}`}>{pdfStatus.msg}</div>}
        </div>
      </div>

      <div className="pipeline">
        <h3>Pipeline</h3>
        <div className="pipeline-steps">
          {["File Upload","Text Extraction","Chunking","Ollama Embed","Qdrant Upsert","RAG Ready"].map((s,i) => (
            <div key={i} className="pipe-step">
              <span className="pipe-num">{i+1}</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DashTab() {
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const [h, s] = await Promise.all([
        fetch(`${API_BASE}/health`).then(r => r.json()),
        fetch(`${API_BASE}/status`).then(r => r.json()),
      ]);
      setHealth(h); setStatus(s);
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="dash-tab">
      <div className="dash-header">
        <h2>System Dashboard</h2>
        <button onClick={load} disabled={loading} className="refresh-btn">
          {loading ? "Loading…" : "🔄 Refresh"}
        </button>
      </div>

      {error && <div className="status-msg error">⚠️ {error}</div>}

      {status && (
        <div className="metrics">
          <div className="metric"><div className="metric-val">{status.csv.chunks}</div><div className="metric-label">CSV Chunks</div></div>
          <div className="metric"><div className="metric-val">{status.pdf.chunks}</div><div className="metric-label">PDF Chunks</div></div>
          <div className="metric"><div className="metric-val">{status.qdrant_points ?? "?"}</div><div className="metric-label">Vectors in Qdrant</div></div>
          <div className="metric"><div className="metric-val">{status.total_chunks}</div><div className="metric-label">Total Chunks</div></div>
        </div>
      )}

      {health && (
        <div className="model-info">
          <h3>Model Config</h3>
          <div className="model-grid">
            <div className="model-item"><span>Embed Model</span><strong>{health.embed_model}</strong></div>
            <div className="model-item"><span>Chat Model</span><strong>{health.chat_model}</strong></div>
            <div className="model-item"><span>Embed Dim</span><strong>{health.embed_dim}</strong></div>
            <div className="model-item"><span>Backend</span><strong>{health.status}</strong></div>
          </div>
        </div>
      )}

      <div className="arch-table">
        <h3>Architecture</h3>
        <table>
          <thead><tr><th>Component</th><th>Service</th></tr></thead>
          <tbody>
            <tr><td>Embeddings</td><td>Ollama Cloud (gemma4:31b-cloud)</td></tr>
            <tr><td>LLM</td><td>Ollama Cloud (gemma4:31b-cloud)</td></tr>
            <tr><td>Vector Store</td><td>Qdrant Cloud (cosine, dim=256)</td></tr>
            <tr><td>Backend</td><td>FastAPI on Render</td></tr>
            <tr><td>Frontend</td><td>React + Vite on Render</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("chat");

  return (
    <div className="app">
      <header>
        <div className="logo">🔗 SCM Assistant</div>
        <p className="tagline">Supply Chain RAG — BQBYTE Technologies</p>
      </header>

      <nav>
        {[["chat","💬 Chat"],["ingest","📤 Data Ingest"],["dash","📊 Dashboard"]].map(([id,label]) => (
          <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "chat" && <ChatTab />}
        {tab === "ingest" && <IngestTab />}
        {tab === "dash" && <DashTab />}
      </main>
    </div>
  );
}
