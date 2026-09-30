"""
NetWorld API Route Handlers Package
"""

from app.routes.traffic import router as traffic_router
from app.routes.forecast import router as forecast_router
from app.routes.explain import router as explain_router
from app.routes.whatif import router as whatif_router
from app.routes.validation import router as validation_router

__all__ = [
    "traffic_router",
    "forecast_router",
    "explain_router",
    "whatif_router",
    "validation_router",
]
