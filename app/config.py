from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # AWS / S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""
    S3_PREFIX: str = ""

    # Embedding model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # FAISS index
    FAISS_INDEX_PATH: str = "data/faiss_index/index.faiss"
    FAISS_METADATA_PATH: str = "data/faiss_index/metadata.pkl"
    FAISS_TOP_K: int = 5

    # In-house LLM
    LLM_ENDPOINT: AnyHttpUrl = Field(default="http://localhost:8080/generate")
    LLM_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: int = 60

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()