from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    llm_model: str
    embed_model: str
    num_ctx: int = Field(ge=1024, le=32768)


@lru_cache
def get_settings() -> Settings:
    return Settings()