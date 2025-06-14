from fastapi.concurrency import asynccontextmanager
import psycopg2

from fastapi import FastAPI

from pydantic import BaseModel
from typing import Optional

from app.routers import document_processing, video_processing, audio_processing


class ProcessingResponse(BaseModel):
    processor: str
    model: str
    result: str
    time_to_process: float
    video_id: Optional[str] = None
    request_id: str
    error: Optional[bool] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Video processing cleanup will be added when video_processing module is implemented

    # audio processing 
    if hasattr(audio_processing, "executor") and audio_processing.executor:
        audio_processing.executor.shutdown(wait=True)


app = FastAPI(
    title="ADP Video Pipeline API",
    description="API for processing video and audio for StadPrin",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(video_processing.router)
app.include_router(audio_processing.router)
app.include_router(document_processing.router)


@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}


@app.get("/schema")
async def process_document():
    return {"message": "Video Processing Pipeline - StadPrin"}
