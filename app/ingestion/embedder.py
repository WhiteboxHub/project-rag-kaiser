from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class Embedder:
    def __init__(self):
        self.model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )

    def embed(self, chunks: list[str]) -> list[list[float]]:
        return self.model.embed_documents(chunks)
