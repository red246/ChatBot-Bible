import streamlit as st
import faiss
import pickle
import hashlib
import openai
from sentence_transformers import SentenceTransformer

# In-memory cache (reset each session)
answer_cache = {}

# OpenRouter API Setup
openai.api_key = st.secrets["openrouter"]["api_key"]
openai.api_base = "https://openrouter.ai/api/v1"
model_name = "microsoft/phi-3-mini-128k-instruct"

# Load data
@st.cache_resource
def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    return chunks, index

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

chunks, index = load_data()
embed_model = load_embedder()

# UI
st.title("📘 Ask My PDF (Powered by OpenRouter)")
st.markdown("Ask questions based on a preloaded PDF document.")

question = st.text_input("Enter your question:")

# Hashing for cache key
def get_cache_key(question: str):
    clean = question.strip().lower()  # Normalize: remove extra spaces, lowercase
    return hashlib.md5(clean.encode("utf-8")).hexdigest()

if question:
    key = get_cache_key(question)

    if key in answer_cache:
        result = answer_cache[key]
        st.info("✅ Loaded from cache.")
    else:
        # Embed & search
        query_vec = embed_model.encode([question])
        _, I = index.search(query_vec, k=2)
        context = ". ".join([chunks[i] for i in I[0]])
        context = context[:1000]

        # Prompt
        prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
"""

        # Call OpenRouter
        with st.spinner("Thinking..."):
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    temperature=0.2,
                )
                result = response.choices[0].message.content.strip()
                answer_cache[key] = result
            except Exception as e:
                result = f"⚠️ Error: {str(e)}"

    # Show answer
    st.markdown("### 📎 Answer:")
    st.write(result)
