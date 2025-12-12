from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

class Embedder:
    def __init__(self):
        # Use a local transformer model
        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, chunks: list[str]) -> list[list[float]]:
        return self.model.embed_documents(chunks)
