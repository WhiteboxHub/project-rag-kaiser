from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()
