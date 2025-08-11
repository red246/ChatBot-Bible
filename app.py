import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

@st.cache_resource
def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    return chunks, index

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        device=-1  # Force CPU
    )
    return embed_model, qa_model

chunks, index = load_data()
embed_model, qa_model = load_models()

st.title("📘 Ask My PDF")
st.write("Ask any question based on the PDF content.")

question = st.text_input("Enter your question:")

if question:
    query_vec = embed_model.encode([question])
    _, I = index.search(query_vec, k=3)
    context = ". ".join([chunks[i] for i in I[0]])

    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    with st.spinner("Thinking..."):
        result = qa_model(prompt, max_new_tokens=200)
        answer = result[0]["generated_text"].split("Answer:")[-1].strip()

    st.markdown("### Answer:")
    st.write(answer)
