import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load preprocessed chunks and FAISS index
@st.cache_resource
def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    return chunks, index

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1  # CPU only
    )
    return embed_model, qa_model

# Load everything
chunks, index = load_data()
embed_model, qa_model = load_models()

# UI
st.title("📘 Ask My PDF (Fast Version)")
st.write("Ask questions based on the PDF I preloaded.")

# User input
question = st.text_input("Enter your question:")

if question:
    # Embed the question
    query_vec = embed_model.encode([question])

    # Search for top 2 relevant chunks
    _, I = index.search(query_vec, k=2)

    # Build context from those chunks
    context = ". ".join([chunks[i] for i in I[0]])

    # Truncate context to ~800 characters for speed
    context = context[:800]

    # Build a Flan-style prompt
    prompt = f"Answer the question based only on this context:\n\n{context}\n\nQuestion: {question}"

    # Generate answer
    with st.spinner("Thinking..."):
        result = qa_model(prompt, max_new_tokens=80)[0]["generated_text"]

    # Show answer
    st.markdown("### 📎 Answer:")
    st.write(result.strip())

