# SCM Assistant — Frontend

React + Vite chatbot UI connected to the FastAPI backend at:
`https://scm-assistant-bot-1ex0.onrender.com`

---

## Run Locally

```bash
npm install
npm run dev
# Open http://localhost:5173
```

---

## Deploy on Render (Static Site)

1. Push this folder to a **GitHub repo** (can be a new repo, just push the contents of this folder).

2. Go to [https://render.com](https://render.com) → **New → Static Site**

3. Connect your GitHub repo.

4. Set these values:
   | Field | Value |
   |-------|-------|
   | **Build Command** | `npm install && npm run build` |
   | **Publish Directory** | `dist` |

5. Click **Create Static Site** — Render will build and deploy automatically.

6. After ~2 minutes your site will be live at a `*.onrender.com` URL.

> ✅ No environment variables needed — the backend URL is hardcoded in `src/App.jsx`.

---

## Features

- **💬 Chat** — Ask questions about your supply chain data, with sample questions
- **📤 Data Ingest** — Upload CSV and PDF files directly to the backend with configurable chunking
- **📊 Dashboard** — Live health/status metrics from the backend

---

## Backend API Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | RAG chat query |
| `/ingest/csv` | POST | Upload & embed CSV |
| `/ingest/pdf` | POST | Upload & embed PDF |
| `/health` | GET | Model config |
| `/status` | GET | Ingest + Qdrant stats |
