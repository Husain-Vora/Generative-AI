from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


model = get_model()

# Same three modes and system prompts as the original script
MODES = {
    1: ("😡 Angry mode", "You are an angry AI agent. You respond aggressively and impatiently."),
    2: ("😂 Funny mode", "You are a very funny AI agent. You respond with humor and jokes"),
    3: ("😢 Sad mode", "You are a Sad AI agent. you respond in a sad way"),
}

if "messages" not in st.session_state:
    st.session_state.messages = None
if "mode_label" not in st.session_state:
    st.session_state.mode_label = None
if "ended" not in st.session_state:
    st.session_state.ended = False

# Step 1: choose the AI mode
if st.session_state.messages is None:
    st.subheader("Choose your AI mode")
    cols = st.columns(3)
    for col, (choice, (label, system_prompt)) in zip(cols, MODES.items()):
        if col.button(label, use_container_width=True, key=f"mode_{choice}"):
            st.session_state.messages = [SystemMessage(content=system_prompt)]
            st.session_state.mode_label = label
            st.rerun()
    st.stop()

# Step 2: chat
st.caption(f"{st.session_state.mode_label} • Type 0 to exit the application")

# Render chat history (system message is not shown)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if st.session_state.ended:
    st.info("Chat ended. Refresh the page to start a new conversation.")
    st.stop()

prompt = st.chat_input("You: ")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    if prompt == "0":
        st.session_state.ended = True
        st.info("Chat ended. Refresh the page to start a new conversation.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
        st.write(response.text)

    st.session_state.messages.append(AIMessage(content=response.text))
