"""
NetWorld Cyberattack Forecasting AI - FastAPI Main Application Entrypoint
SIH 2026 Project Backend Foundation
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.routes.traffic import router as traffic_router
from app.routes.forecast import router as forecast_router
from app.routes.explain import router as explain_router
from app.routes.whatif import router as whatif_router
from app.routes.validation import router as validation_router
from app.routes.response import router as response_router
from app.routes.defender import router as defender_router
from app.model.loader import ModelLoader

app = FastAPI(
    title="NetWorld API",
    description="World-Model AI That Forecasts Cyberattacks - FastAPI Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration for React Vite Frontend
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    if "Access-Control-Allow-Private-Network" not in response.headers:
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# Include APIRouters (supporting /api and /api/v1)
app.include_router(traffic_router, prefix="/api/traffic")
app.include_router(traffic_router, prefix="/api/v1/traffic")

app.include_router(forecast_router, prefix="/api/forecast")
app.include_router(forecast_router, prefix="/api/v1/forecast")
app.include_router(forecast_router, prefix="/forecast")

app.include_router(explain_router, prefix="/api/explain")
app.include_router(explain_router, prefix="/api/v1/explain")

app.include_router(whatif_router, prefix="/api/whatif")
app.include_router(whatif_router, prefix="/api/what-if")
app.include_router(whatif_router, prefix="/api/v1/whatif")

app.include_router(validation_router, prefix="/api/validation")
app.include_router(validation_router, prefix="/api/v1/validation")

app.include_router(response_router, prefix="/api/response")
app.include_router(response_router, prefix="/api/v1/response")

app.include_router(defender_router, prefix="/api/defender")
app.include_router(defender_router, prefix="/api/v1/defender")


@app.get("/")
async def root():
    """
    Root endpoint returning basic NetWorld API information and status.
    """
    loader = ModelLoader()
    artifact_status = loader.check_artifacts_exist()

    return {
        "name": "NetWorld API",
        "description": "World-Model AI That Forecasts Cyberattacks",
        "version": "1.0.0",
        "status": "online",
        "mode": "offline",
        "model_artifacts": artifact_status,
        "supported_input": {
            "sequence_length": 20,
            "num_features": 36,
            "dataset": "CIC-IDS2018",
        },
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for application monitoring.
    """
    return {
        "status": "ok",
        "service": "networld-backend",
        "version": "1.0.0",
    }
