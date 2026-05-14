from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model

load_dotenv()

# Load environment variables
load_dotenv()

# Initialize model
model = init_chat_model("mistral-small-latest")

# Page Config
st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #A0A0A0;
    margin-bottom: 30px;
}

.stSelectbox label {
    font-size: 18px;
    font-weight: bold;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🤖 AI Mood Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose your AI personality and start chatting</div>', unsafe_allow_html=True)

# Mood Selection
choice = st.selectbox(
    "Choose AI Mode",
    ["😡 Angry Mode", "😢 Sad Mode", "😂 Funny Mode"]
)

# Define system prompt
if choice == "😡 Angry Mode":
    mode = "you are an angry AI agent. you respond aggressively and impatiently."

elif choice == "😢 Sad Mode":
    mode = "you are a sad AI agent. you respond with emotional and sad replies."

elif choice == "😂 Funny Mode":
    mode = "you are a funny AI agent. you respond in a funny way."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset chat if mode changes
if "last_mode" not in st.session_state:
    st.session_state.last_mode = choice

if st.session_state.last_mode != choice:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]
    st.session_state.chat_history = []
    st.session_state.last_mode = choice

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# User input
prompt = st.chat_input("Type your message...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append(("user", prompt))

    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Generate AI response
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.chat_history.append(("assistant", response.content))