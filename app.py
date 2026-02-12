import streamlit as st
import time
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="Laxmi AI", page_icon="🤖", layout="centered")

# Custom Dark Styling
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { background-color: #4B6EAF; color: white; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laxmi AI Chatbot")

# 2. Optimized Model Loading using the Secret
@st.cache_resource
def load_model():
    try:
        # Accessing the token from the secrets you just saved
        hf_token = st.secrets["HUGGINGFACE_TOKEN"]
        
        # Passing the token to the pipeline
        return pipeline(
            "text-generation", 
            model="sshleifer/tiny-gpt2", 
            token=hf_token
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

bot = load_model()

# 3. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Chat Logic
if prompt := st.chat_input("Type your message..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if bot:
            # Generate
            output = bot(prompt, max_new_tokens=40, do_sample=True)
            reply = output[0]["generated_text"].replace(prompt, "").strip()
            
            # Typing effect
            placeholder = st.empty()
            full_response = ""
            for char in reply:
                full_response += char
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
            placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("Model not loaded. Check your Hugging Face Token.")
