import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------
# Load embedding model
# -----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Load documents with metadata
# -----------------------
documents_list = []

data_path = "data"

for filename in os.listdir(data_path):

    if filename.endswith(".txt"):

        file_path = os.path.join(data_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:

            text = f.read()

            # Simple Chunking
            chunk_size = 500
            chunk_overlap = 100

            for chunk_id, start in enumerate(
                range(0, len(text), chunk_size)
            ):

                chunk = text[start:start + chunk_size]

                documents_list.append(
                    {
                        "text": chunk,
                        "source": filename,
                        "chunk_id": chunk_id
                    }
                )

# -----------------------
# Create embeddings
# -----------------------
texts = [doc["text"] for doc in documents_list]

document_embeddings = model.encode(texts)
document_embeddings = np.array(document_embeddings).astype("float32")

# -----------------------
# Cosine Similarity
# -----------------------
faiss.normalize_L2(document_embeddings)

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print("✅ Embeddings created successfully!")
print("📄 Total chunks indexed:", index.ntotal)

# -----------------------
# SEARCH FUNCTION
# -----------------------
def search(query, k=3, threshold=0.40):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if score >= threshold:

            results.append(
                {
                    "text": documents_list[idx]["text"],
                    "source": documents_list[idx]["source"],
                    "chunk_id": documents_list[idx]["chunk_id"],
                    "score": float(score)
                }
            )

    return results

# -----------------------
# TEST RUN
# -----------------------
if __name__ == "__main__":

    query = "what is fever"

    results = search(query)

    print("\n🔍 Top Matching Chunks:\n")

    for r in results:

        print(f"📄 Source: {r['source']}")
        print(f"🧩 Chunk: {r['chunk_id']}")
        print(f"⭐ Score: {r['score']:.3f}")
        print(r["text"][:200])
        print("-" * 50)