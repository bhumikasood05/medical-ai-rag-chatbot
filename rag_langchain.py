from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# GROQ LLM
# -----------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# -----------------------
# GLOBAL VARIABLES
# -----------------------
db = None
retriever = None

# -----------------------
# LOAD PDF FUNCTION
# -----------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

# -----------------------
# BUILD VECTOR STORE
# -----------------------
def build_vector_store(text):

    docs = [Document(page_content=text)]

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    global db, retriever

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

# -----------------------
# MAIN RAG FUNCTION
# -----------------------
def ask_rag(question):

    if retriever is None:
        return "Please upload a PDF first."

    docs = retriever.invoke(question)

    context = "\n".join([d.page_content for d in docs])

    prompt = (
        "You are a STRICT medical assistant.\n\n"
        "RULES:\n"
        "- Use ONLY the provided context\n"
        "- Do NOT guess or assume anything\n"
        "- If answer is not in context, say 'Not found in PDF'\n"
        "- Keep answer short (2-5 lines)\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:"
    )

    response = llm.invoke(prompt)

    return response.content