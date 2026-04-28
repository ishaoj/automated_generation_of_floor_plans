"""
FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import generate, health, preview


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered Vastu-compliant floor plan generator",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        generate.router,
        prefix=settings.api_prefix,
        tags=["Generation"]
    )
    app.include_router(
        preview.router,
        prefix=settings.api_prefix,
        tags=["Preview"]
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
