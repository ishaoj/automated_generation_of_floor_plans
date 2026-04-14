"""
Application configuration settings
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Vastu-AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # API
    api_prefix: str = "/api"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Generation settings
    default_variations: int = 3
    max_variations: int = 5
    image_width: int = 800
    image_height: int = 800

    # ML Model settings
    use_gnn_refiner: bool = True  # Enable GNN layout refinement
    gnn_model_path: str = "ml/models/saved/gnn_v1/best_model.pt"

    use_ml_renderer: bool = False  # Enable ML visual rendering
    renderer_model_path: str = "ml/models/saved/pix2pix_renderer.pt"
    renderer_type: Literal["pillow", "ml"] = "pillow"

    # ML inference settings
    ml_device: str = "cpu"  # 'cpu' or 'cuda'
    gnn_max_adjustment: float = 0.15  # Max position adjustment by GNN (15%)
    gnn_min_room_size: float = 0.05  # Min room size after GNN refinement (5%)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
