# 🩺 Medical AI RAG Chatbot

> A smart **AI-powered Medical Question Answering System** using **Retrieval-Augmented Generation (RAG)** with LangChain, FAISS, and Groq LLM.

---

## ✨ Overview

This project allows users to upload **medical PDF documents** and ask questions.  
The system retrieves the most relevant information from the document and generates **accurate, context-based answers** using a powerful LLM.

⚠️ Designed for **educational purposes only** (not a medical diagnostic tool).

---

## 🚀 Demo Features

💬 Chat with your medical PDF  
📄 Upload any medical document  
🧠 Context-aware AI responses  
⚡ Fast semantic search (FAISS)  
🤖 Powered by Groq LLaMA 3.1  
🎯 Strict “answer only from context” mode  

---

## 🧠 How It Works (RAG Pipeline)

PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Search → Groq LLM → Final Answer

---

## 🛠️ Tech Stack

🐍 Python  
🎨 Streamlit  
🦜 LangChain  
📦 FAISS (Vector Database)  
🤗 HuggingFace Embeddings  
⚡ Groq LLM (LLaMA 3.1)  
📄 PyPDF  

---

## 📁 Project Structure

Medical AI Assistant/
│
├── app_ui.py              # Streamlit chatbot UI
├── rag_langchain.py      # RAG pipeline logic
├── llm.py                # Groq LLM integration
├── embed_store.py        # Embeddings & vector store
├── test_rag.py           # CLI testing script
├── data/                 # PDF dataset folder
├── temp.pdf              # Sample PDF
├── .env                  # API keys (ignored in Git)
└── README.md

---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/bhumikasood05/medical-ai-rag-chatbot.git
cd medical-ai-rag-chatbot
