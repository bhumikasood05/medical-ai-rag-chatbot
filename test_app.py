from llm import generate_answer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------------
# 1. SAMPLE MEDICAL DOCUMENTS
# -----------------------------
documents = [
    "Diabetes symptoms include frequent urination, thirst, fatigue, and weight loss.",
    "High blood pressure may cause headache, dizziness, and nosebleeds.",
    "Asthma symptoms include shortness of breath, chest tightness, and wheezing.",
    "Common cold symptoms are sneezing, sore throat, runny nose, and cough.",
    "Fever is a temporary increase in body temperature often due to infection.",
    "Heart attack symptoms include chest pain, sweating, and shortness of breath.",
    "Migraine causes severe headache, nausea, and sensitivity to light.",
    "Anemia symptoms include fatigue, weakness, and pale skin.",
    "COVID-19 symptoms include fever, dry cough, and loss of taste or smell.",
    "Allergy symptoms include sneezing, itching, and watery eyes."
]

# -----------------------------
# 2. EMBEDDING MODEL
# -----------------------------
print("Loading embeddings model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents)

print("Embeddings created successfully!")
print("Total documents indexed:", len(documents))

# -----------------------------
# 3. BM25 SETUP
# -----------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# 4. HYBRID RETRIEVAL FUNCTION
# -----------------------------
def hybrid_retrieve(query, top_k=3):

    # ----- BM25 SCORE -----
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # ----- VECTOR SCORE -----
    query_vec = embedding_model.encode(query)

    vector_scores = []
    for emb in doc_embeddings:
        score = np.dot(query_vec, emb) / (
            np.linalg.norm(query_vec) * np.linalg.norm(emb)
        )
        vector_scores.append(score)

    vector_scores = np.array(vector_scores)

    # ----- NORMALIZE -----
    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-6)
    vector_scores = vector_scores / (np.max(vector_scores) + 1e-6)

    # ----- COMBINE -----
    final_scores = 0.5 * bm25_scores + 0.5 * vector_scores

    # ----- TOP K -----
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]

# -----------------------------
# 5. MAIN CHAT LOOP
# -----------------------------
print("\n🏥 Medical AI Assistant (Hybrid RAG)")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a medical question: ")

    if query.lower() == "exit":
        break

    # ----- RETRIEVE CONTEXT -----
    retrieved_docs = hybrid_retrieve(query)
    context = "\n".join(retrieved_docs)

    # ----- GENERATE ANSWER -----
    answer = generate_answer(context, query)

    print("\nMedical AI Assistant:\n", answer, "\n")