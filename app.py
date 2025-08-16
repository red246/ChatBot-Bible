import streamlit as st
import faiss
import pickle
import hashlib
import openai
from sentence_transformers import SentenceTransformer

st.markdown("""
<a href="#main-content" class="skip-link">Skip to main content</a>
<style>
body {{
    background-image: url('https://cdn12.picryl.com/photo/2016/12/31/bible-book-holy-scripture-religion-6e74a9-1024.jpg');
    background-color: #5E095E; /* Fallback color */
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    font-family: Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #FFFFFF;
    margin: 0;
    padding: 0;
}}

.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #000;
    color: #fff;
    padding: 8px 16px;
    z-index: 1000;
    text-decoration: none;
}
.skip-link:focus {
    top: 0;
}

/* ---------- Center All Headers ---------- */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* ---------- Input Styling (Accessible) ---------- */
.st-key-styledinput input {
    border: 2px solid #CBC3E3; /* Light lavender */
    border-radius: 5px;
    background-color: #FFFFFF;
    color: #000000;
    padding: 10px;
    font-size: 1rem;
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
    display: block;
}

/* Hover and focus states with accessible outline and contrast */
.st-key-styledinput input:hover {
    border: 2px solid #f02035;
    box-shadow: 0 0 8px #f02092;
}

.st-key-styledinput input:focus {
    border: 2px solid #FFD700;
    box-shadow: 0 0 10px #FFD700;
    outline: 3px solid #FFD700; /* Ensure visible keyboard focus */
}

/* ---------- Skip Link (Optional Accessibility Feature) ---------- */
.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #000;
    color: #fff;
    padding: 8px 16px;
    z-index: 1000;
    text-decoration: none;
}
.skip-link:focus {
    top: 0;
}

/* ---------- Dark Mode Support ---------- */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #121212;
        color: #FFFFFF;
    }

    .st-key-styledinput input {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 2px solid #888;
    }

    .st-key-styledinput input:focus {
        border: 2px solid #FFD700;
        box-shadow: 0 0 10px #FFD700;
        outline: 3px solid #FFD700;
    }
}

</style>
""", unsafe_allow_html=True)

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
st.markdown("# Ask My PDF")  # H2 — Section title
st.markdown("## Enter your question below")  # H3 — Instruction
st.markdown("### This bible reading is from the American Standard version ")  # H3 — Instruction

question = st.text_input(
    "Your question",
    key="styledinput",
    help="Type a question about the uploaded PDF document. Press Enter to submit."
)

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
    st.markdown("""
    <div id="main-content" role="main" aria-label="Answer Section">
      <h4>Answer:</h4>
    </div>
    """, unsafe_allow_html=True)
    st.write(result)
    

