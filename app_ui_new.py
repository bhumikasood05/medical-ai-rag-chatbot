import streamlit as st
import tempfile

from rag_langchain_new import (
    load_pdf,
    build_vector_store,
    ask_rag,
    clear_history
)

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺"
)

st.title("🩺 Medical AI Assistant")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


# ---------------- CLEAR CHAT ----------------
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    clear_history()
    st.rerun()


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📄 Upload Medical PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    text = load_pdf(pdf_path)

    build_vector_store(
        text,
        uploaded_file.name
    )

    st.session_state.pdf_loaded = True

    st.success(f"✅ PDF Loaded Successfully: {uploaded_file.name}")


# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------- USER INPUT ----------------
query = st.chat_input("Ask your medical question...")

if query:

    if not query.strip():
        st.warning("⚠️ Please enter a valid question.")
        st.stop()

    if not st.session_state.pdf_loaded:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    # store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Generating answer..."):
        answer = ask_rag(query)

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)