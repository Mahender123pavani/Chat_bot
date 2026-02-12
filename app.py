# app.py
import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Page config
st.set_page_config(page_title="Laxmi AI", page_icon="🤖", layout="wide")

# Dark mode & chat styling with scrollable chat box
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
.chat-box {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 10px;
    background-color: #121417;
}
.stChatMessage {
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 6px;
}
.stChatMessage.user {
    background-color: #3B5998;
    color: white;
}
.stChatMessage.assistant {
    background-color: #2E2E2E;
    color: #FFFFFF;
}
/* Scrollbar styling */
.chat-box::-webkit-scrollbar {
    width: 8px;
}
.chat-box::-webkit-scrollbar-thumb {
    background-color: #4B6EAF;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laxmi AI Chatbot")

# Load model with Hugging Face token
@st.cache_resource
def load_model():
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    MODEL_NAME = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

bot = load_model()

# Initialize session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Scrollable chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Typing animation for assistant with streaming
    with st.chat_message("assistant"):
        prompt = f"You are a helpful, friendly AI chatbot. Respond clearly and concisely.\nUser: {user_input}\nAssistant:"

        max_tokens = max(50, min(len(user_input)*2, 150))  # limit for smoother performance
        response = bot(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]["generated_text"]

        reply = response.replace(prompt, "").strip()

        # Emoji reactions
        if "hello" in reply.lower() or "hi" in reply.lower():
            reply += " 👋"
        elif "good" in reply.lower():
            reply += " 😊"

        # Stream text character by character
        placeholder = st.empty()
        display_text = ""
        for char in reply:
            display_text += char
            placeholder.markdown(display_text)
            time.sleep(0.01)

        st.session_state.messages.append({"role": "assistant", "content": reply})
