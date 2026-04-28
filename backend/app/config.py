"""
Application configuration settings.

IMPORTANT: GNN and Pix2Pix are both MANDATORY.
There are no optional / fallback flags — every generated floor plan
goes through the full ML pipeline.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str    = "Vastu-AI"
    app_version: str = "2.0.0"
    debug: bool      = True

    # API
    api_prefix: str = "/api"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Generation settings
    default_variations: int = 3
    max_variations: int     = 5

    # ML pipeline config file — governs both GNN and Pix2Pix
    pipeline_config: str = "backend/configs/pipeline_config.yaml"

    # Convenience overrides (can also be set in pipeline_config.yaml)
    gnn_model_path:     str = "ml/models/saved/gnn_v1/best_model.pt"
    pix2pix_model_path: str = "ml/models/saved/pix2pix_v1/best_model.pt"
    ml_device:          str = "auto"     # auto: cuda > mps > cpu
    ml_image_size:      int = 256        # Pix2Pix output resolution

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
