from fastapi.concurrency import asynccontextmanager
import psycopg2

from fastapi import FastAPI

from pydantic import BaseModel
from typing import Optional

from app.routers import document_processing, document_flow #, video_processing TODO


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


app = FastAPI(
    title="ADP Video Pipeline API",
    description="API for processing video and audio for StadPrin",
    version="1.0.0",
    lifespan=lifespan,
)
# app.include_router(video_processing.router) TODO
app.include_router(document_processing.router)
app.include_router(document_flow.router)


@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}


@app.get("/schema")
async def process_document():
    return {"message": "Video Processing Pipeline - StadPrin"}
