import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 AI Movie Extractor",
    page_icon="🎥",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.stTextArea textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 15px !important;
    border: 2px solid #6366f1 !important;
    padding: 15px !important;
    font-size: 16px !important;
}

.stButton button {
    width: 100%;
    background: linear-gradient(to right, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 15px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #2563eb, #7c3aed);
}

.movie-card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(to right, #8b5cf6, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 30px;
}

.info-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# MODEL
# =========================
model = ChatMistralAI(
    model="mistral-small-2506"
)

# =========================
# PYDANTIC MODEL
# =========================
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# =========================
# PROMPT
# =========================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the paragraph.

{format_instructions}
"""
    ),
    ("human", "{paragraph}")
])

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🎬 AI Movie Information Extractor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Extract movie details instantly using LangChain + Mistral AI 🚀</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚡ Features")
    st.write("✅ AI Powered")
    st.write("✅ Structured JSON Extraction")
    st.write("✅ Modern UI")
    st.write("✅ LangChain + Mistral")
    st.write("✅ Pydantic Validation")

    st.divider()

    st.info("Paste any movie paragraph and get structured movie data instantly.")

# =========================
# INPUT
# =========================
paragraph = st.text_area(
    "🎥 Enter Movie Paragraph",
    height=250,
    placeholder="""
Example:
Inception, directed by Christopher Nolan and released in 2010, is a science fiction thriller starring Leonardo DiCaprio...
"""
)

# =========================
# BUTTON
# =========================
if st.button("🚀 Extract Movie Information"):

    if paragraph.strip() == "":
        st.warning("⚠️ Please enter a movie paragraph.")
    else:

        with st.spinner("🤖 AI is analyzing the movie..."):

            final_prompt = prompt.invoke({
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            movie_data = parser.parse(response.content)

            time.sleep(1)

        st.success("✅ Extraction Completed!")

        # =========================
        # RESULT CARD
        # =========================
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="info-box">
            <h3>🎬 Title</h3>
            <p>{movie_data.title}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
            <h3>📅 Release Year</h3>
            <p>{movie_data.release_year}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
            <h3>🎭 Genres</h3>
            <p>{", ".join(movie_data.genre)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-box">
            <h3>🎬 Director</h3>
            <p>{movie_data.director}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
            <h3>⭐ Rating</h3>
            <p>{movie_data.rating}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
            <h3>👥 Cast</h3>
            <p>{", ".join(movie_data.cast)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
        <h3>📝 Summary</h3>
        <p>{movie_data.summary}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # JSON VIEW
        # =========================
        st.subheader("📦 Raw Structured Output")
        st.json(movie_data.dict())

# =========================
# FOOTER
# =========================
st.markdown("""
<br><br>
<hr>
<center>
Made with ❤️ using Streamlit + LangChain + Mistral AI
</center>
""", unsafe_allow_html=True)