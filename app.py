import streamlit as st
import time
from transformers import pipeline

# 1. Modern Page Config
st.set_page_config(page_title="Laxmi AI", page_icon="🤖", layout="centered")

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { background-color: #4B6EAF; color: white; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laxmi AI Chatbot")

# 2. Optimized Model Loading
@st.cache_resource
def load_model():
    try:
        # Using a small model for demo; swap with 'gpt2' or larger for better results
        return pipeline("text-generation", model="sshleifer/tiny-gpt2")
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

bot = load_model()

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()  # Updated from experimental_rerun()

# 4. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User Input and Interaction
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Response Logic
    with st.chat_message("assistant"):
        if bot:
            # Generate response
            raw_output = bot(prompt, max_new_tokens=30, do_sample=True)[0]["generated_text"]
            reply = raw_output.replace(prompt, "").strip()
            
            # 6. Improved Typing Effect with Cursor
            placeholder = st.empty()
            full_response = ""
            for char in reply:
                full_response += char
                placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
            placeholder.markdown(full_response)
            
            # Save assistant response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("AI is currently unavailable.")
