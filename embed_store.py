import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read documents from data folder
documents = []

for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Create embeddings
document_embeddings = model.encode(documents)

# Create FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(np.array(document_embeddings))

print("Embeddings created successfully!")
print("Total documents indexed:", index.ntotal)