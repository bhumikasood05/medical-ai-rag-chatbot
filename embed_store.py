import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------
# Load embedding model
# -----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Load documents
# -----------------------
documents_list = []

data_path = "data"

for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            documents_list.append(f.read())

# -----------------------
# Create embeddings
# -----------------------
document_embeddings = model.encode(documents_list)
document_embeddings = np.array(document_embeddings).astype("float32")

# -----------------------
# Create FAISS index
# -----------------------
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(document_embeddings)

print("✅ Embeddings created successfully!")
print("📄 Total documents indexed:", index.ntotal)

# -----------------------
# SEARCH FUNCTION (RAG CORE)
# -----------------------
def search(query, k=2):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        results.append(documents_list[i])

    return results

# -----------------------
# TEST RUN
# -----------------------
if __name__ == "__main__":
    query = "what is fever"
    results = search(query)

    print("\n🔍 Top matching documents:\n")
    for r in results:
        print("-", r)