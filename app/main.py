import shutil
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse
import psycopg2
from fastapi import status
from fastapi import FastAPI, File, HTTPException, UploadFile

from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from routers import document_processing, audio_processing, video_processing
from routers import video_processing


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

    if hasattr(video_processing, "executor") and video_processing.executor:
        video_processing.executor.shutdown(wait=True)

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}


@app.post("/upload-video")
async def upload_video(file: UploadFile=File(...)):
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return JSONResponse(status_code=200, content={"message": "success"})


@app.get("/schema")
async def process_document():
    return {"message": "Video Processing Pipeline - StadPrin"}
