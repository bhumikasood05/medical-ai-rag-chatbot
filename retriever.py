# retriever.py

def retrieve_documents(query):
    """
    Simple retriever (dummy RAG baseline).
    Later you can replace this with BM25 / FAISS / embeddings.
    """

    # Example small medical knowledge base (you can expand this)
    knowledge_base = [
        "Fever is often caused by viral or bacterial infection.",
        "Headache can be caused by stress, dehydration, or migraine.",
        "Common cold symptoms include sneezing, cough, and sore throat.",
        "Diabetes is a chronic condition that affects blood sugar levels.",
        "High blood pressure can increase risk of heart disease.",
        "Back pain often occurs due to poor posture or muscle strain.",
        "Vitamin D deficiency can cause fatigue and bone pain.",
        "Stomach pain may be caused by indigestion or infection."
    ]

    query = query.lower()

    # Simple keyword matching (baseline retrieval)
    results = []

    for doc in knowledge_base:
        if any(word in doc.lower() for word in query.split()):
            results.append(doc)

    # If nothing matches, return general fallback
    if not results:
        results = [
            "No exact match found in medical database.",
            "Try rephrasing your question or adding more details."
        ]

    return results