from __future__ import annotations
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@dataclass(frozen=True)
class Settings:
    supabase_url: str | None
    supabase_key: str | None
    gemini_api_key: str | None
    ollama_host: str
    model_size: str
    frame_interval: int

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def rag_enabled(self) -> bool:
        return self.supabase_enabled and bool(self.ollama_host)

def get_settings() -> Settings:
    return Settings(
        supabase_url=os.getenv("SUPABASE_URL") or None,
        supabase_key=os.getenv("SUPABASE_KEY") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model_size=os.getenv("MODEL_SIZE", "small"),
        frame_interval=int(os.getenv("FRAME_INTERVAL", "4")),
    )
