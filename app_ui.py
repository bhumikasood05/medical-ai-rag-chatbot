import streamlit as st
from llm import generate_answer
from retriever import retrieve_documents

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺"
)

st.title("🩺 Medical AI Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("Ask your medical question:")

if query:

    docs = retrieve_documents(query)

    context = "\n".join(docs)

    answer = generate_answer(context, query)

    st.session_state.chat.append(("You", query))
    st.session_state.chat.append(("AI", answer))

for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")