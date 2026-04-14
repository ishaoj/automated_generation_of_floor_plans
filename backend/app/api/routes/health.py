"""
Health check endpoints
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "vastu-ai"}


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Vastu-AI API",
        "docs": "/docs",
        "version": "0.1.0"
    }
