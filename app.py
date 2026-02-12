# app.py
import streamlit as st
from transformers import pipeline
import time

# Page config
st.set_page_config(page_title="Laxmi AI", page_icon="🤖", layout="wide")

# Dark mode CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4B6EAF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Laxmi AI Chatbot")

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

bot = load_model()

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Typing animation
    with st.chat_message("assistant"):
        response = bot(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]
        reply = response.replace(prompt, "").strip()
        placeholder = st.empty()
        display_text = ""
        for char in reply:
            display_text += char
            placeholder.markdown(display_text)
            time.sleep(0.02)  # adjust speed
        st.session_state.messages.append({"role": "assistant", "content": reply})
