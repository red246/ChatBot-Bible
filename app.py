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
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        device=-1  # Run on CPU
    )
    return embed_model, qa_model

# Load everything
chunks, index = load_data()
embed_model, qa_model = load_models()

# UI
st.title("📘 Ask My PDF")
st.write("Ask questions based on the uploaded PDF content.")

# User input
question = st.text_input("Enter your question:")

if question:
    # Embed the question
    query_vec = embed_model.encode([question])

    # Search for top 2 relevant chunks
    _, I = index.search(query_vec, k=2)

    # Build context from those chunks
    context = ". ".join([chunks[i] for i in I[0]])

    # Truncate context to avoid exceeding model's input limit
    context = context[:1200]  # ~1200 characters is safe for 1024-token models

    # Build the prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # Generate answer
    with st.spinner("Thinking..."):
        output = qa_model(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        answer = output[0]["generated_text"].split("Answer:")[-1].strip()

    # Show answer
    st.markdown("### 📎 Answer:")
    st.write(answer)
