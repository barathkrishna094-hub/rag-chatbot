# 🤖 RAG Chatbot — Chat With Your Own PDFs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF
and ask questions about it. Instead of hallucinating answers, the model
retrieves the most relevant parts of your document and uses them as context
before answering.

## How It Works

1. **Upload** a PDF through the Streamlit UI.
2. **Chunking** — the PDF's text is split into overlapping chunks (so
   context isn't lost at chunk boundaries).
3. **Embedding** — each chunk is converted into a vector using
   `sentence-transformers` (runs locally, free, no API needed).
4. **Indexing** — vectors are stored in a `FAISS` index for fast similarity
   search.
5. **Retrieval** — when you ask a question, it's embedded too, and FAISS
   finds the top-k most similar chunks from the PDF.
6. **Generation** — those chunks + your question are sent to an LLM
   (via [Groq](https://groq.com)'s free API) which answers using only that
   context.

```
PDF → chunks → embeddings → FAISS index
                                 │
User question → embedding → search index → top-k chunks
                                 │
                    chunks + question → LLM → Answer
```

## Tech Stack

| Component        | Tool                                   |
|-------------------|-----------------------------------------|
| UI                | Streamlit                              |
| PDF parsing       | PyPDF2                                 |
| Embeddings        | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search     | FAISS                                  |
| LLM               | Groq API (`llama-3.1-8b-instant`)      |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys) — it's
free and takes under a minute. Then either:

- Copy `.env.example` to `.env` and paste your key in, **or**
- Just paste the key directly into the sidebar when the app runs.

### 5. Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Paste your Groq API key in the sidebar.
2. Upload a PDF (research paper, notes, resume, textbook chapter, etc.).
3. Wait a few seconds for it to index.
4. Ask questions in the chat box. Expand "View retrieved context" under
   any answer to see exactly which chunks the model used.

## Possible Extensions

Good ways to build on this for a stronger portfolio piece:
- Support multiple file formats (`.docx`, `.txt`, web pages).
- Persist the FAISS index to disk so you don't re-embed on every restart.
- Swap Groq for a fully local LLM (e.g. via Ollama) for an offline version.
- Add citation markers so answers point back to the exact chunk/page used.
- Deploy it for free on [Streamlit Community Cloud](https://streamlit.io/cloud)
  and link the live demo in this README.

## License

MIT — free to use and modify.
