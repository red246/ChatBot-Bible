import streamlit as st
import faiss
import pickle
import hashlib
import openai
from sentence_transformers import SentenceTransformer

theme = st.radio("🌗 Choose Theme", ["Light", "Dark"], horizontal=True)

def set_theme_css(mode):
    if mode == "Light":
        css = """
        body {
            background-color: #f9f9f9;
            color: #222;
        }
        .stMarkdown {
            background-color: #ffffff;
            border-left: 4px solid #3498db;
            padding: 1rem;
            border-radius: 8px;
        }
        h1 {
            color: #1f77b4;
        }
        input {
            background-color: #fff;
            color: #000;
        }
        """
    else:  # Dark mode
        css = """
        body {
            background-color: #121212;
            color: #eee;
        }
        .stMarkdown {
            background-color: #1e1e1e;
            border-left: 4px solid #6ab0f3;
            padding: 1rem;
            border-radius: 8px;
            color: #eee;
        }
        h1 {
            color: #6ab0f3;
        }
        input {
            background-color: #2c2c2c;
            color: #fff;
        }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

set_theme_css(theme)

# In-memory cache (reset each session)
answer_cache = {}

# OpenRouter API Setup
openai.api_key = st.secrets["openrouter"]["chatbotkey"]
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
st.title("Just Ask The Bible")
st.markdown("Ask questions about the Bible and life according to the Bible")

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

local_css("""
body {
    background-color: #fefefe;
    background-image: linear-gradient(to right, #f0f2f5, #ffffff);
}

    /* Make title stand out */
    h1 {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Add padding to input box */
    .stTextInput>div>div>input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }

    /* Answer output styling */
    .stMarkdown {
        font-size: 1.1em;
        background: #eaf2f8;
        border-left: 5px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
    }
""")

