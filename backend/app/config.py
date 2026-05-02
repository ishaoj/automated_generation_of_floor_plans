"""Application configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve paths relative to this file so they work on any machine/server
_BACKEND_DIR  = Path(__file__).resolve().parent.parent   # .../backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                       # .../vastu-ai/

_DEFAULT_GNN_PATH    = str(_PROJECT_ROOT / "ml" / "models" / "saved" / "gnn_v3" / "best_model.pt")
_DEFAULT_PIPE_CONFIG = str(_BACKEND_DIR / "configs" / "pipeline_config.yaml")


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Application
    app_name:    str  = "Vastu-AI"
    app_version: str  = "2.0.0"
    debug:       bool = False   # False in production

    # API
    api_prefix: str = "/api"

    # CORS — override via CORS_ORIGINS env var (comma-separated) on Render:
    #   CORS_ORIGINS=https://vastu-ai.netlify.app,http://localhost:3000
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @property
    def all_cors_origins(self) -> list[str]:
        """Merge env-provided origins with defaults (deduped)."""
        return list(dict.fromkeys(self.cors_origins))

    # Generation settings
    default_variations: int = 3
    max_variations:     int = 5

    # ML pipeline config (resolved from project root — works on any machine)
    pipeline_config: str = _DEFAULT_PIPE_CONFIG

    # Override via GNN_MODEL_PATH env var if needed
    gnn_model_path: str = _DEFAULT_GNN_PATH
    ml_device:      str = "auto"   # auto: cuda > mps > cpu

    class Config:
        env_file            = ".env"
        env_file_encoding   = "utf-8"


settings = Settings()
