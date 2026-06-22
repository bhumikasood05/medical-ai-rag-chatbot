from llm import generate_answer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------------
# MEDICAL DATASET (IMPROVED)
# -----------------------------
documents = [
    "Diabetes symptoms include frequent urination, thirst, fatigue, weight loss, and weakness.",
    "High blood pressure may cause headache, dizziness, and nosebleeds.",
    "Asthma symptoms include shortness of breath, chest tightness, and wheezing.",
    "Common cold symptoms are sneezing, sore throat, runny nose, and cough.",
    "Fever is a temporary increase in body temperature often due to infection.",
    "Heart attack symptoms include chest pain, sweating, and shortness of breath.",
    "Migraine causes severe headache, nausea, and sensitivity to light.",
    "Anemia symptoms include fatigue, weakness, tiredness, low energy, and pale skin.",
    "COVID-19 symptoms include fever, dry cough, and loss of taste or smell.",
    "Allergy symptoms include sneezing, itching, and watery eyes."
]

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
print("Loading embeddings model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

print("Embeddings created successfully!")
print("Total documents indexed:", len(documents))

# -----------------------------
# BM25 SETUP
# -----------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# HYBRID RETRIEVAL (IMPROVED)
# -----------------------------
def hybrid_retrieve(query, top_k=4):

    # BM25 scoring
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Vector scoring
    query_vec = embedding_model.encode(query)
    query_vec = np.array(query_vec).astype("float32")

    vector_scores = []
    for emb in doc_embeddings:
        score = np.dot(query_vec, emb) / (
            np.linalg.norm(query_vec) * np.linalg.norm(emb)
        )
        vector_scores.append(score)

    vector_scores = np.array(vector_scores)

    # SAFE NORMALIZATION
    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-8)
    vector_scores = vector_scores / (np.max(vector_scores) + 1e-8)

    # IMPROVED WEIGHTING (MORE SEMANTIC POWER)
    final_scores = 0.2 * bm25_scores + 0.8 * vector_scores

    # TOP K SELECTION
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]

# -----------------------------
# MAIN LOOP
# -----------------------------
print("\n🏥 Medical AI Assistant (Hybrid RAG)")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask a medical question: ")

    if query.lower() == "exit":
        break

    # RETRIEVE CONTEXT
    retrieved_docs = hybrid_retrieve(query)
    context = "\n".join(retrieved_docs)

    # GENERATE ANSWER
    answer = generate_answer(context, query)

    print("\n🤖 Answer:\n", answer, "\n")