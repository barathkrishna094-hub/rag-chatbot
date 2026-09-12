"""
RAG Chatbot — Chat with your own PDF documents.

How it works:
1. User uploads a PDF.
2. The text is split into chunks and converted into embeddings (numeric vectors)
   using a free local model (sentence-transformers).
3. Embeddings are stored in a FAISS vector index for fast similarity search.
4. When the user asks a question, we embed the question, retrieve the most
   relevant chunks from the PDF, and pass them + the question to an LLM
   (Groq's free API) which answers using ONLY that context.

This is called Retrieval-Augmented Generation (RAG) because the model's
answer is "augmented" with retrieved facts instead of relying purely on
what it memorized during training.
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, free, runs locally
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks so context isn't cut off
TOP_K = 3              # number of chunks to retrieve per question
LLM_MODEL = "llama-3.1-8b-instant"  # free & fast on Groq

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat with your PDF (RAG)")


# ---------- Helper functions ----------

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and cache it across reruns."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks: list[str], embed_model):
    """Embed all chunks and build a FAISS index for similarity search."""
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve_relevant_chunks(question: str, chunks, index, embed_model, k=TOP_K):
    """Find the k chunks most similar to the question."""
    question_embedding = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]


def ask_llm(question: str, context_chunks: list[str], api_key: str) -> str:
    """Send the question + retrieved context to the Groq LLM and get an answer."""
    client = Groq(api_key=api_key)

    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- Streamlit UI ----------

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com/keys",
    )
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.markdown("---")
    st.caption(
        "Built with Streamlit, sentence-transformers, FAISS, and Groq. "
        "Everything except the final answer generation runs locally and free."
    )

# Session state to avoid recomputing embeddings on every rerun
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.file_name = None

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text.strip():
                st.error("Couldn't extract any text from this PDF (it may be a scanned image).")
            else:
                chunks = chunk_text(text)
                embed_model = load_embedding_model()
                index, _ = build_faiss_index(chunks, embed_model)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.file_name = uploaded_file.name
        st.success(f"Indexed '{uploaded_file.name}' into {len(st.session_state.chunks)} chunks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask something about your PDF...")

if question:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif st.session_state.index is None:
        st.error("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embed_model = load_embedding_model()
                relevant_chunks = retrieve_relevant_chunks(
                    question, st.session_state.chunks, st.session_state.index, embed_model
                )
                answer = ask_llm(question, relevant_chunks, api_key)
                st.markdown(answer)

                with st.expander("View retrieved context"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Chunk {i}:** {chunk}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
