import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Laxmi AI", page_icon="🤖")

st.title("🤖 Laxmi AI Chatbot")

# load model
@st.cache_resource
def load():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

bot = load()

# memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# input
prompt = st.chat_input("Type message")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    response = bot(prompt, max_new_tokens=40, do_sample=False)[0]["generated_text"]

    reply = response.replace(prompt,"").strip()

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role":"assistant","content":reply})
