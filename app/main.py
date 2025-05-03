import os
import time
import logging
from fastapi.concurrency import asynccontextmanager
import psycopg2

from fastapi import FastAPI, HTTPException, BackgroundTasks

from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from psycopg2.extras import Json
from datetime import datetime
from app.routers import video_processing


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

    if hasattr(video_processing, 'executor') and video_processing.executor:
        video_processing.executor.shutdown(wait=True)

app = FastAPI(
    title="ADP Video Pipeline API",
    description="API for processing video and audio for StadPrin",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(video_processing.router)


@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}


@app.get("/schema")
async def process_document():
    return {"message": "Video Processing Pipeline - StadPrin"}


# @app.post("/audio-processor", response_model=ProcessingResponse)
# async def process_audio(data: BaseProcessor, background_tasks: BackgroundTasks):
#     """Process audio content and return results"""
#     start_time = time.time()
#     processor_fn = SERVICES.get(data.processor)

#     if not processor_fn:
#         raise HTTPException(status_code=400, detail=f"Processor {data.processor} not found")

#     try:
#         # Process based on processor type
#         processor_type = "audio"
#         if data.processor == "ImageToText":
#             processor_type = "image"
#             # Process visual content (image or video)
#             result = processor_fn(
#                 text=data.prompt,
#                 model=data.model,
#                 temperature=data.temperature,
#                 max_new_tokens=data.max_tokens,
#                 quantization=data.quantization,
#                 keep_model_loaded=data.keep_model_loaded,
#                 video_path=data.video_path,
#                 source_path=data.image_path,
#                 attention=data.attention
#             )
#         else:
#             # Process audio
#             result = processor_fn(
#                 text=data.prompt,
#                 model=data.model,
#                 temperature=data.temperature,
#                 max_new_tokens=data.max_tokens,
#                 quantization=data.quantization
#             )

#         # Calculate processing time
#         processing_time = time.time() - start_time

#         # Format result for response
#         if isinstance(result, dict):
#             response_data = {
#                 "processor": data.processor,
#                 "model": data.model if data.model else result.get('model', 'unknown'),
#                 "result": result.get('result', ''),
#                 "time_to_process": result.get('time_to_process', processing_time),
#                 "error": result.get('error', False)
#             }
#         else:
#             response_data = {
#                 "processor": data.processor,
#                 "model": data.model,
#                 "result": result,
#                 "time_to_process": processing_time,
#                 "error": False
#             }

#         # Add request ID and store result in database asynchronously
#         request_id = db_manager(response_data, processor_type)
#         response_data["request_id"] = request_id

#         return response_data

#     except Exception as e:
#         logger.error(f"Error processing with {data.processor}: {str(e)}")
#         processing_time = time.time() - start_time

#         error_response = {
#             "processor": data.processor,
#             "model": data.model,
#             "result": f"Error: {str(e)}",
#             "time_to_process": processing_time,
#             "error": True,
#             "request_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         }

#         # Store error in database
#         processor_type = "audio"
#         if data.processor == "ImageToText":
#             processor_type = "image"
#         background_tasks.add_task(db_manager, error_response, processor_type)

#         return error_response
