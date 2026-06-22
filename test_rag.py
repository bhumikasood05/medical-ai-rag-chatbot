from rag_langchain import load_pdf, build_vector_store, ask_rag

text = load_pdf("temp.pdf")
build_vector_store(text)

print(ask_rag("What is diabetes?"))