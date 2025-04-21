from fastapi import FastAPI
from pydantic import BaseModel

class BaseProcessor(BaseModel):
    processor: str
    response: str
    time_to_process: float

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}

@app.post("/audio-processor")
async def get_audio_to_text(data: BaseProcessor):
    return {"message": "Video Processing Pipeline - StadPrin"}

@app.post("/video-processor")
async def get_video_to_text(data: BaseProcessor):
    return {"message": data.model_dump()}

@app.get("/schema")
async def get_document_to_schema():
    return {"message": "Video Processing Pipeline - StadPrin"}

