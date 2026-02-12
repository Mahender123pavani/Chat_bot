# app.py
import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline

# Page config
st.set_page_config(page_title="Laxmi AI", page_icon="🤖", layout="wide")

# Dark mode & chat styling
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.stTextInput>div>div>input { background-color: #1E1E1E; color: #FFFFFF; }
.stButton>button { background-color: #4B6EAF; color: white; }
.chat-box { max-height: 500px; overflow-y: auto; padding: 10px; border-radius: 10px; background-color: #121417; }
.stChatMessage { border-radius: 10px; padding: 8px; margin-bottom: 6px; }
.stChatMessage.user { background-color: #3B5998; color: white; }
.stChatMessage.assistant { background-color: #2E2E2E; color: #FFFFFF; }
.chat-box::-webkit-scrollbar { width: 8px; }
.chat-box::-webkit-scrollbar-thumb { background-color: #4B6EAF; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laxmi AI Chatbot (Falcon-7B-Instruct)")

# Load Falcon-7B-Instruct with token
@st.cache_resource
def load_model():
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN", None)
    if hf_token is None:
        st.warning("Hugging Face token missing. Only public models can be used.")
    MODEL_NAME = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

bot: TextGenerationPipeline = load_model()

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")

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

    with st.chat_message("assistant"):
        prompt = f"You are a helpful AI chatbot. Respond clearly and concisely.\nUser: {user_input}\nAssistant:"

        # Stream small chunk responses
        response = bot(
            prompt,
            max_new_tokens=150,
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

        # Typing animation
        placeholder = st.empty()
        display_text = ""
        for char in reply:
            display_text += char
            placeholder.markdown(display_text)
            time.sleep(0.01)

        st.session_state.messages.append({"role": "assistant", "content": reply})
