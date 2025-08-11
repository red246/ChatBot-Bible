import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load data
@st.cache_resource
def load_data():
    try:
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        index = faiss.read_index("faiss.index")
        return chunks, index
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

chunks, index = load_data()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load free model from Hugging Face
qa_model = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device="cpu")

st.title("📄 Ask My PDF")
st.write("Ask questions about a specific PDF document")

question = st.text_input("Your question")

if question:
    query_vec = embed_model.encode([question])
    D, I = index.search(query_vec, k=3)
    context = ". ".join([chunks[i] for i in I[0]])

    prompt = f"[INST] Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer: [/INST]"

    with st.spinner("Thinking..."):
        response = qa_model(prompt, do_sample=True, max_new_tokens=256)[0]["generated_text"]
        answer = response.split("[/INST]")[-1].strip()

    st.markdown("### Answer:")
    st.write(answer)
