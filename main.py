from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.auth import router as auth_router
from api.routers.chat_session import router as chat_session_router
from api.routers.upload import router as upload_router
from api.routers.analysis import router as analysis_router
from rag_pipeline.qdrant import lifespan

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_session_router)
app.include_router(upload_router)
app.include_router(analysis_router)
