import importlib

def test_defaults_when_env_absent(monkeypatch):
    for k in ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY", "OLLAMA_HOST", "MODEL_SIZE", "FRAME_INTERVAL"]:
        monkeypatch.delenv(k, raising=False)
    import settings as s
    importlib.reload(s)
    cfg = s.get_settings()
    assert cfg.model_size == "small"
    assert cfg.frame_interval == 4
    assert cfg.ollama_host == "http://localhost:11434"
    assert cfg.rag_enabled is False

def test_rag_enabled_when_supabase_and_ollama_present(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    import settings as s
    importlib.reload(s)
    cfg = s.get_settings()
    assert cfg.rag_enabled is True
