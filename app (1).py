import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="HR Policy Assistant", page_icon="📄", layout="wide")

# ---------------- Config ----------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 100
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ---------------- PDF processing ----------------
def extract_text_from_pdf(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_index(query, embedder, index, chunks, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def ask_groq(client, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are an HR Policy Assistant. Answer the user's question strictly "
        "using the provided HR policy context below. If the answer is not "
        "present in the context, clearly say you could not find that "
        "information in the uploaded policy document. Be concise, accurate, "
        "and professional."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


# ---------------- Session state ----------------
for key, default in [
    ("chunks", None),
    ("index", None),
    ("chat_history", []),
    ("file_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_secret_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=get_secret_key(),
        help="Get a free key at console.groq.com. If deployed with a secret "
        "named GROQ_API_KEY, it will be prefilled automatically.",
    )
    st.divider()
    uploaded_file = st.file_uploader("Upload HR Policy PDF", type=["pdf"])
    if st.button("🗑️ Clear session"):
        st.session_state.chunks = None
        st.session_state.index = None
        st.session_state.chat_history = []
        st.session_state.file_name = None
        st.rerun()

st.title("📄 HR Policy Assistant")
st.caption("Upload your company's HR policy PDF and ask questions about it.")

embedder = load_embedder()

# ---------------- Indexing ----------------
if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
    with st.spinner("Reading and indexing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        if not raw_text.strip():
            st.error(
                "Could not extract text from this PDF. It may be a "
                "scanned/image-only document."
            )
        else:
            chunks = chunk_text(raw_text)
            index = build_faiss_index(chunks, embedder)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'")

# ---------------- Chat ----------------
if st.session_state.chunks is None:
    st.info("👈 Upload an HR policy PDF from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the HR policy...")
    if question:
        if not api_key_input:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        relevant_chunks = search_index(
                            question, embedder, st.session_state.index, st.session_state.chunks
                        )
                        client = get_groq_client(api_key_input)
                        answer = ask_groq(client, question, relevant_chunks)
                        st.markdown(answer)
                        with st.expander("📚 Source excerpts used"):
                            for i, c in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Excerpt {i}:** {c[:400]}...")
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                        st.error(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
