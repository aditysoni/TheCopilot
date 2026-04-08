from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.app.core.config import settings
from apps.api.app.api.routes.health import router as health_router
from apps.api.app.api.routes.chat import router as chat_router
from apps.api.app.api.routes.ingest import router as ingest_router
from apps.api.app.api.routes.conversations import router as conversations_router
from apps.api.app.api.routes.sources import router as sources_router
from packages.db.database import Base, engine
import packages.db.models  # noqa: F401


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
app.include_router(conversations_router, prefix="/api/v1", tags=["conversations"])
app.include_router(sources_router, prefix="/api/v1", tags=["sources"])