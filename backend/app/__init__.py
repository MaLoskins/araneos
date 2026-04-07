from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routes import graph, training
from app.dependencies import verify_api_key
from app.middleware.rate_limit import RateLimitMiddleware


def create_app() -> FastAPI:
    settings = get_settings()

    # Only require API key globally when API_KEY is set
    dependencies = []
    if settings.API_KEY:
        dependencies.append(Depends(verify_api_key))

    app = FastAPI(title="Araneos API", version="1.0.0", dependencies=dependencies)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=len(settings.CORS_ORIGINS) == 1 and settings.CORS_ORIGINS[0] != "*",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RateLimitMiddleware, default_rpm=100, train_rpm=5)

    # v1 routes
    app.include_router(graph.router, prefix="/v1")
    app.include_router(training.router, prefix="/v1")

    # Backward-compatible un-prefixed routes
    app.include_router(graph.router)
    app.include_router(training.router)

    return app
