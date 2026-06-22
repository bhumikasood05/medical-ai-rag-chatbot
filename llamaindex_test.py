from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set embedding model globally
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = SimpleDirectoryReader("data").load_data()

print("Documents Loaded:", len(documents))

index = VectorStoreIndex.from_documents(documents)

print("Vector Index Created Successfully!")