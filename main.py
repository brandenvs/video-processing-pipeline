from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import tempfile
import os
import time
import json
import logging
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

# Import services
from services import SERVICES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters - these should be stored in environment variables in production
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

class BaseProcessor(BaseModel):
    processor: str
    model: Optional[str] = "Qwen2.5-VL-3B-Instruct"
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    quantization: Optional[str] = "8bit"
    video_path: Optional[str] = None
    image_path: Optional[str] = None
    attention: Optional[str] = "eager"
    keep_model_loaded: Optional[bool] = False

class ProcessingResponse(BaseModel):
    processor: str
    model: str
    result: str
    time_to_process: float
    video_id: Optional[str] = None
    request_id: str
    error: Optional[bool] = False
    
app = FastAPI(title="ADP Video Pipeline API", 
             description="API for processing video and audio for StadPrin",
             version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Video Processing Pipeline - StadPrin"}

def insert_result_to_db(result: Dict[str, Any], processor_type: str) -> str:
    """Insert processing result into PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Generate a unique ID for this request
        request_id = f"{processor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # SQL for inserting the result
        sql = """
        INSERT INTO processing_results 
        (request_id, processor_type, model, result_json, processing_time, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        # Format result data for insertion
        values = (
            request_id,
            processor_type,
            result.get('model', 'unknown'),
            Json({"result": result.get('result', '')}),
            result.get('time_to_process', 0.0),
            datetime.now()
        ) 
        print(f'{datetime.now()}')
        
        cursor.execute(sql, values)
        db_id = cursor.fetchone()[0]
        conn.commit()
        
        logger.info(f"Successfully inserted result with ID: {db_id}")
        return request_id
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        # Don't raise the exception - we want the API to return a result even if DB insert fails
        return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    finally:
        if conn:
            cursor.close()
            conn.close()

@app.post("/audio-processor", response_model=ProcessingResponse)
async def process_audio(data: BaseProcessor, background_tasks: BackgroundTasks):
    """Process audio content and return results"""
    start_time = time.time()
    processor_fn = SERVICES.get(data.processor)
    
    if not processor_fn:
        raise HTTPException(status_code=400, detail=f"Processor {data.processor} not found")
    
    try:
        # Process based on processor type
        processor_type = "audio"
        if data.processor == "ImageToText":
            processor_type = "image"
            # Process visual content (image or video)
            result = processor_fn(
                text=data.prompt,
                model=data.model,
                temperature=data.temperature,
                max_new_tokens=data.max_tokens,
                quantization=data.quantization,
                keep_model_loaded=data.keep_model_loaded,
                video_path=data.video_path,
                source_path=data.image_path,
                attention=data.attention
            )
        else:
            # Process audio
            result = processor_fn(
                text=data.prompt,
                model=data.model,
                temperature=data.temperature,
                max_new_tokens=data.max_tokens,
                quantization=data.quantization
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format result for response
        if isinstance(result, dict):
            response_data = {
                "processor": data.processor,
                "model": data.model if data.model else result.get('model', 'unknown'),
                "result": result.get('result', ''),
                "time_to_process": result.get('time_to_process', processing_time),
                "error": result.get('error', False)
            }
        else:
            # Handle string result case
            response_data = {
                "processor": data.processor,
                "model": data.model,
                "result": result,
                "time_to_process": processing_time,
                "error": False
            }
        
        # Add request ID and store result in database asynchronously
        request_id = insert_result_to_db(response_data, processor_type)
        response_data["request_id"] = request_id
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing with {data.processor}: {str(e)}")
        processing_time = time.time() - start_time
        
        error_response = {
            "processor": data.processor,
            "model": data.model,
            "result": f"Error: {str(e)}",
            "time_to_process": processing_time,
            "error": True,
            "request_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Store error in database
        processor_type = "audio"
        if data.processor == "ImageToText":
            processor_type = "image"
        background_tasks.add_task(insert_result_to_db, error_response, processor_type)
        
        return error_response

@app.post("/video-processor")
async def get_video_to_text(data: BaseProcessor, background_tasks: BackgroundTasks):
    """Process video and return analysis results"""
    # Default to ImageToText processor if not specified
    if not data.processor or data.processor not in SERVICES:
        data.processor = "ImageToText"
    
    return await process_audio(data, background_tasks)

@app.get("/schema")
async def get_document_to_schema():
    return {"message": "Video Processing Pipeline - StadPrin"}