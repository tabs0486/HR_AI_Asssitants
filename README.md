# 📄 HR Policy Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that lets employees upload an
HR policy PDF and ask natural-language questions about it.

**Stack**
- **Streamlit** – web UI
- **PyMuPDF (fitz)** – PDF text extraction
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) – text embeddings
- **FAISS** – vector similarity search
- **Groq API** (`openai/gpt-oss-20b`) – LLM answer generation

---

## How it works

1. User uploads an HR policy PDF.
2. PyMuPDF extracts the raw text.
3. Text is split into overlapping chunks.
4. Each chunk is embedded with Sentence-Transformers and stored in a FAISS
   in-memory index.
5. When the user asks a question, the question is embedded and FAISS
   retrieves the most relevant chunks.
6. Those chunks + the question are sent to Groq's `openai/gpt-oss-20b` model,
   which answers using only that context.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files/folders excluded from Git |
| `README.md` | This file |

---

## Getting a Groq API key

1. Go to https://console.groq.com
2. Sign up / log in (free tier available).
3. Go to **API Keys** → **Create API Key**.
4. Copy the key — you'll paste it into the app's sidebar, or store it as a
   Streamlit secret (see deployment steps below).

---

## Deploy it — 100% through the browser (no Colab, no VS Code, no terminal)

You only need a **GitHub account** and a **Streamlit Community Cloud
account** (https://share.streamlit.io). Everything below is done by
clicking around on websites.

### Step 1 — Create a GitHub repository (via the GitHub website)

1. Go to https://github.com and log in.
2. Click the **+** icon (top-right) → **New repository**.
3. Give it a name, e.g. `hr-policy-assistant`.
4. Set it to **Public** (Streamlit Community Cloud's free tier needs a
   public repo, or a private one it has access to).
5. Check **"Add a README file"** (optional, we'll replace it) and click
   **Create repository**.

### Step 2 — Upload the files (no terminal, no Git commands)

1. Inside your new repository, click **Add file** → **Upload files**.
2. Drag and drop (or browse and select) these 4 files:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
3. Scroll down and click **Commit changes**.

That's it — your code is now on GitHub, entirely through the web UI.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and log in with your GitHub account.
2. Click **New app**.
3. Choose:
   - **Repository:** `your-username/hr-policy-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** (before deploying) to add your secret:
   - In the **Secrets** box, add:
     ```
     GROQ_API_KEY = "your_groq_api_key_here"
     ```
   - This lets the app auto-fill the key so users don't have to paste it
     each time (they can still override it manually in the sidebar).
5. Click **Deploy**.

Streamlit Cloud will install everything from `requirements.txt` and launch
the app. The first build can take a few minutes since
`sentence-transformers` pulls in PyTorch.

### Step 4 — Use the app

1. Open the deployed app URL (Streamlit gives you a link like
   `https://your-app-name.streamlit.app`).
2. Upload an HR policy PDF in the sidebar.
3. Wait for the "Indexed N chunks..." success message.
4. Ask questions in the chat box at the bottom.

### Updating the app later (still no terminal)

1. Go to your GitHub repo in the browser.
2. Open the file you want to change (e.g. `app.py`).
3. Click the pencil ✏️ **Edit** icon.
4. Make your changes and click **Commit changes**.
5. Streamlit Community Cloud auto-detects the push and redeploys the app
   within a minute or two.

---

## Notes & tips

- **Free Groq tier limits:** if you hit rate limits, wait a bit or reduce
  `TOP_K` / `max_tokens` in `app.py`.
- **Scanned PDFs:** if a PDF is a scanned image (no selectable text),
  PyMuPDF can't extract text from it — you'd need OCR (not included here).
- **Privacy:** the uploaded PDF and its chunks live only in the app's
  memory for that session; nothing is persisted to disk or a database.
- **Never commit your API key** directly into `app.py` or any file pushed
  to GitHub — always use Streamlit's **Secrets** manager instead.
