from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

db = None
retriever = None
chat_history = []


def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def build_vector_store(text, pdf_name):
    global db, retriever

    docs = [
        Document(
            page_content=text,
            metadata={"source": pdf_name}
        )
    ]

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )


def ask_rag(question):
    global chat_history

    if retriever is None:
        return "Please upload a PDF first."

    docs = retriever.invoke(question)

    if not docs:
        return "Not found in PDF"

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    history_text = "\n".join(chat_history[-6:])

    prompt = f"""
You are a STRICT medical assistant.

RULES:
- Use ONLY the provided context
- Do NOT guess anything
- If answer is not present in the context, say "Not found in PDF"
- Keep answer short and clear

CHAT HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {response.content}")

    sources = list(
        set(
            [doc.metadata.get("source", "Unknown") for doc in docs]
        )
    )

    return response.content + "\n\n📚 Source: " + ", ".join(sources)


def clear_history():
    global chat_history
    chat_history = []