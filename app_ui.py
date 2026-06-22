import streamlit as st
from rag_langchain import load_pdf, build_vector_store, ask_rag

st.set_page_config(page_title="Medical AI Chatbot", page_icon="🩺")

st.title("🩺 LangChain Medical Chatbot")

# -----------------------
# SESSION STATE
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# -----------------------
# PDF UPLOAD
# -----------------------
uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.pdf_loaded:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text = load_pdf("temp.pdf")
    build_vector_store(text)

    st.session_state.pdf_loaded = True
    st.success("PDF loaded successfully!")

# -----------------------
# SHOW CHAT HISTORY
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# CHAT INPUT
# -----------------------
query = st.chat_input("Ask your medical question...")

if query:

    # user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            answer = ask_rag(query)

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})