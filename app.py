import streamlit as st
import faiss
import pickle
import hashlib
import openai
from sentence_transformers import SentenceTransformer
import re

st.markdown("""
<a href="#main-content" class="skip-link">Skip to main content</a>
<style>
st.app {{
   background-color: velvet;
    font-family: Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #FFFFFF;
    margin: 0;
    padding: 0;
}}

.skip-link {
    position: absolute;
    left: -9999px;
    top: auto;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(1px, 1px, 1px, 1px);
    white-space: nowrap;
}

.skip-link:focus {
    left: 0;
    top: 0;
    width: auto;
    height: auto;
    clip: auto;
    padding: 8px 16px;
    background: #000;
    color: #fff;
    z-index: 1000;
}

h1, h2, h3, h4, h5, h6 {
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.st-key-styledinput input {
    border: 2px solid #CBC3E3;
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

.st-key-styledinput input:hover {
    border: 2px solid #f02035;
    box-shadow: 0 0 8px #f02092;
}

.st-key-styledinput input:focus {
    border: 2px solid #FFD700;
    box-shadow: 0 0 10px #FFD700;
    outline: 3px solid #FFD700;
}

.search-strategy {
    background: rgba(0,0,100,0.7);
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# In-memory cache
answer_cache = {}

# OpenRouter API Setup
openai.api_key = st.secrets["openrouter"]["chatbotkey"]
openai.api_base = "https://openrouter.ai/api/v1"
model_name = "microsoft/phi-3-mini-128k-instruct"

# Smart query expansion for topics that may not have direct answers
CONCEPT_MAPPINGS = {
    # Age-related concepts -> biblical principles about life stages
    "age": ["wisdom", "youth", "elder", "generation", "child", "father", "mother", "teach", "learn", "experience"],
    "old": ["wisdom", "elder", "gray", "ancient", "counsel", "understanding", "mature"],
    "young": ["youth", "child", "son", "daughter", "learn", "obey", "grow", "train"],
    
    # Race/ethnicity -> biblical concepts of unity and inclusion
    "race": ["nation", "people", "tribe", "gentile", "jew", "foreigner", "stranger", "neighbor", "brother"],
    "ethnicity": ["nation", "people", "kindred", "tongue", "tribe", "israel", "gentile"],
    "color": ["nation", "people", "tribe", "all", "every", "whosoever"],
    
    # Life guidance -> core biblical themes
    "life": ["live", "walk", "path", "way", "light", "truth", "peace", "joy", "love", "hope", "faith"],
    "purpose": ["called", "chosen", "will", "plan", "work", "serve", "glorify", "kingdom"],
    "meaning": ["truth", "wisdom", "understanding", "purpose", "eternal", "treasure", "heart"],
    "guidance": ["lead", "guide", "path", "way", "counsel", "wisdom", "direction", "teach"]
}

@st.cache_resource
def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    return chunks, index

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def detect_question_type(question):
    """Identify if this is a topic that may not have direct biblical answers"""
    q_lower = question.lower()
    
    # Questions likely to have indirect answers
    indirect_indicators = [
        ("age", ["how old", "what age", "age of", "years old"]),
        ("race", ["what race", "what color", "ethnicity", "skin color"]),
        ("specific_life", ["what job", "career", "where to live", "who to marry"])
    ]
    
    for category, patterns in indirect_indicators:
        if any(pattern in q_lower for pattern in patterns):
            return category
    
    return "direct"

def create_alternative_queries(question, question_type):
    """Create alternative search queries for indirect topics"""
    q_lower = question.lower()
    alternatives = [question]  # Always include original
    
    if question_type == "age":
        # Focus on principles about life stages instead of specific ages
        if "old" in q_lower or "age" in q_lower:
            alternatives.extend([
                "wisdom of elders scripture",
                "honoring father and mother",
                "gray hair crown of glory",
                "teaching children wisdom"
            ])
    
    elif question_type == "race":
        # Focus on unity and love for all people
        alternatives.extend([
            "all nations tribes tongues",
            "love your neighbor as yourself", 
            "no jew nor greek bond nor free",
            "whosoever believes shall not perish",
            "God shows no partiality"
        ])
    
    elif question_type == "specific_life":
        # Focus on general life principles
        alternatives.extend([
            "seek first the kingdom of God",
            "trust in the Lord with all your heart",
            "walk in wisdom",
            "whatever you do do it heartily"
        ])
    
    # Always add conceptual alternatives
    for word in q_lower.split():
        if word in CONCEPT_MAPPINGS:
            concepts = CONCEPT_MAPPINGS[word][:3]  # Top 3 related concepts
            alternatives.extend(concepts)
    
    return alternatives[:6]  # Limit to 6 queries max

def smart_search(question, chunks, index, embed_model):
    """Enhanced search that tries multiple strategies"""
    question_type = detect_question_type(question)
    search_queries = create_alternative_queries(question, question_type)
    
    all_chunk_indices = set()
    
    # Search with all alternative queries
    for query in search_queries:
        query_vec = embed_model.encode([query])
        _, indices = index.search(query_vec, k=3)
        all_chunk_indices.update(indices[0])
    
    # Get unique chunks
    selected_chunks = [chunks[i] for i in list(all_chunk_indices)[:8]]
    
    return selected_chunks, question_type, len(search_queries)

def create_smart_prompt(question, chunks, question_type):
    """Create prompts that acknowledge when direct answers aren't available"""
    
    context = "\n\n".join([f"Passage {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
    
    if question_type in ["age", "race", "specific_life"]:
        instruction = f"""You are a helpful Bible teacher. The user asked a question that may not have a direct biblical answer. 

Biblical Context:
{context}

Question: {question}

Instructions:
1. If the Bible doesn't directly answer this specific question, say so honestly
2. Then explain what biblical principles DO apply to this topic
3. Focus on what Scripture teaches about related themes (love, wisdom, unity, God's character)
4. For age questions: discuss biblical principles about different life stages
5. For race questions: emphasize biblical teachings on unity, love for all people, and God's inclusive love
6. For life guidance: provide relevant biblical principles even if not addressing the exact situation
7. Be encouraging and practical while staying biblically grounded

Answer:"""
    else:
        instruction = f"""Answer the question using the biblical passages provided.

Biblical Context:
{context}

Question: {question}

Instructions:
1. Answer based on what the passages say
2. Include relevant verse references when possible
3. If passages don't fully address the question, explain what information is available

Answer:"""
    
    return instruction

chunks, index = load_data()
embed_model = load_embedder()

# UI
st.markdown("# Ask The Bible")
st.markdown("### This bible reading is from the American Standard Bible")
st.markdown("## Enter your question below")

# Example questions with click-to-fill functionality
with st.expander("📝 Click any question to try it!"):
    
    st.markdown("**Direct Biblical Questions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💝 What does Jesus say about love?", key="q1"):
            st.session_state.styledinput = "What does Jesus say about love?"
            st.rerun()
        
        if st.button("🕊️ How can I find peace in troubles?", key="q2"):
            st.session_state.styledinput = "How can I find peace in troubles?"
            st.rerun()
            
        if st.button("🙏 What does the Bible say about prayer?", key="q3"):
            st.session_state.styledinput = "What does the Bible say about prayer?"
            st.rerun()
    
    with col2:
        if st.button("💪 How to have strength in difficult times?", key="q4"):
            st.session_state.styledinput = "How to have strength in difficult times?"
            st.rerun()
            
        if st.button("😊 What brings joy according to scripture?", key="q5"):
            st.session_state.styledinput = "What brings joy according to scripture?"
            st.rerun()
            
        if st.button("🌟 How do I trust God's plan?", key="q6"):
            st.session_state.styledinput = "How do I trust God's plan?"
            st.rerun()
    
    st.markdown("**Challenging Questions (Indirect Answers):**")
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🤝 What does the Bible say about racism?", key="q7"):
            st.session_state.styledinput = "What does the Bible say about racism?"
            st.rerun()
            
        if st.button("👴 What does the Bible say about elderly people?", key="q8"):
            st.session_state.styledinput = "What does the Bible say about elderly people?"
            st.rerun()
            
        if st.button("🎯 How do I find my purpose in life?", key="q9"):
            st.session_state.styledinput = "How do I find my purpose in life?"
            st.rerun()
    
    with col4:
        if st.button("💑 What age should someone get married?", key="q10"):
            st.session_state.styledinput = "What age should someone get married?"
            st.rerun()
            
        if st.button("🌍 How should different races treat each other?", key="q11"):
            st.session_state.styledinput = "How should different races treat each other?"
            st.rerun()
            
        if st.button("💼 What does the Bible say about work?", key="q12"):
            st.session_state.styledinput = "What does the Bible say about work?"
            st.rerun()


question = st.text_input(
    "Your question",
    key="styledinput",
    help="Ask any biblical question. The app will find relevant principles even for topics not directly addressed in Scripture."
)

def get_cache_key(question: str):
    clean = question.strip().lower()
    return hashlib.md5(clean.encode("utf-8")).hexdigest()

if question:
    key = get_cache_key(question)

    if key in answer_cache:
        result, question_type, search_count = answer_cache[key]
        st.info("✅ Loaded from cache.")
    else:
        # Smart search
        relevant_chunks, question_type, search_count = smart_search(question, chunks, index, embed_model)
        
        # Create appropriate prompt
        prompt = create_smart_prompt(question, relevant_chunks, question_type)
        
        # Call OpenRouter
        with st.spinner("Searching biblical wisdom..."):
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a wise Bible teacher who helps people understand Scripture and apply biblical principles to life questions."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,  # More space for nuanced answers
                    temperature=0.2,
                )
                result = response.choices[0].message.content.strip()
                answer_cache[key] = (result, question_type, search_count)
            except Exception as e:
                result = f"⚠️ Error: {str(e)}"
                question_type = "error"
                search_count = 0

    # Display results
    st.markdown("""
    <div id="main-content" role="main" aria-label="Answer Section">
      <h4>Biblical Wisdom:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Show search strategy used
    if question_type in ["age", "race", "specific_life"]:
        strategy_text = {
            "age": "🔍 Searched for biblical principles about life stages and wisdom",
            "race": "🔍 Searched for biblical teachings on unity and love for all people", 
            "specific_life": "🔍 Searched for general biblical life principles"
        }
        st.markdown(f'<div class="search-strategy">{strategy_text.get(question_type, "")}</div>', unsafe_allow_html=True)
    
    st.write(result)
    
    # Helpful follow-up suggestions
    if question_type in ["age", "race", "specific_life"]:
        st.info("💡 **Remember**: The Bible focuses more on principles and character than specific details. Try asking about related themes like 'wisdom', 'love', 'unity', or 'God's will' for deeper insights!")
