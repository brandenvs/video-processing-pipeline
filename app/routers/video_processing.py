import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import torch
import numpy as np
import gc
import cv2
import asyncio
import logging
from datetime import datetime
from app.routers.db_functions import DB_CONFIG # imported db_config
import psycopg2

from torchvision.transforms import ToPILImage
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)
from PIL import Image
from qwen_vl_utils import process_vision_info

from app.routers import model_management as mm

from fastapi import APIRouter, HTTPException


from concurrent.futures import ThreadPoolExecutor
import atexit
import functools

from moviepy import VideoFileClip


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# startd here ---------------------------------------------------------------

# Have a dictionary with DB_CONFIG here ...
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# Write a function that will connect to "PostgreSQL" database here ...
def ConnectToDb():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"Database connection established to {DB_CONFIG['database']} at {DB_CONFIG['host']}")
        return conn, cursor
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        print(f"Failed to connect to database: {e}")
        return None, None

# end here --------------------------------------------------------------------------

class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = "Perform a detailed analysis of the given data"
    temperature: Optional[float] = 0.5
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None

router = APIRouter()

router = APIRouter(
    prefix="/process",
    tags=["process"],
    responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)


def cleanup_executor():
    executor.shutdown(wait=True)


atexit.register(cleanup_executor)


def check_memory(device=mm.get_torch_device()):
    print("Device Loaded: ", device)

    total_mem = mm.get_total_memory()
    print(f"GPU has {total_mem} GBs")

    free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
    return (free_mem_gb, total_mem)


@router.post("/process/")
async def process_video(request_body: BaseProcessor):
    loop = asyncio.get_running_loop()

    torch.cuda.empty_cache()
    gc.collect()
    check_memory()

# startd here ---------------------------------------------------------------


    try: 
        inference_task = functools.partial(model_manager.inference, **request_body.model_dump())
        generated_data = await loop.run_in_executor(executor, inference_task)
        response = json.loads(generated_data)

        if isinstance(response, list) and len(response) > 0:
            try:
                # Try to parse as JSON first
                try:
                    analysis_data = json.loads(response[0])
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, use text format
                    analysis_data = {
                        'Frame description': response[0],
                        'License plates': [],
                        'Identification documents': [],
                        'Scene sentiment': {},
                        'People nearby': [],
                        'Risk analysis': {},
                    }
            except Exception as e:
                # Handle potential exceptions, such as IndexError
                analysis_data = {'error': str(e)}
        else:
            analysis_data = response

        # Store in database
        db_helper = HelperFunctionDb()
        analysis_id = db_helper.store_analysis_results(
            analysis_data, 
            source_path=request_body.source_path
        )
        
        if analysis_id:
            analysis_data['id'] = analysis_id
            
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "results": analysis_data,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")   

# end here --------------------------------------------------------------------------

# start here ----------------------------------------------------------------------

class HelperFunctionDb:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish a database connection"""
        import psycopg2
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def close(self):
        """Close database connection properly"""
        if self.cursor:
            self.cursor.close()
        if self.conn and not self.conn.closed:
            self.conn.close()

    def format_with_timestamp(self, data, current_time):
        """Format data items with timestamps if they don't have one"""
        from datetime import datetime
        
        if current_time is None:
            current_time = datetime.now().isoformat()
            
        if isinstance(data, dict):
            if 'timestamp' not in data:
                data['timestamp'] = current_time
            return data
        else:
            return {
                'value': str(data),
                'timestamp': current_time
            }

    def store_analysis_results(self, analysis_data, source_path=None):
        current_time = datetime.now().isoformat()
        
        try:
            if not self.connect():
                return None
                
            # Format frame descriptions
            frame_desc = analysis_data.get('Frame description', '')
            if isinstance(frame_desc, str):
                frame_descriptions = [{
                    'description': frame_desc,
                    'timestamp': current_time
                }]
            elif isinstance(frame_desc, list):
                frame_descriptions = [self.format_with_timestamp(desc, current_time) for desc in frame_desc]
            else:
                frame_descriptions = []
                
            # Format license plates
            license_plates = []
            for plate in analysis_data.get('License plates', []):
                if plate:
                    license_plates.append(self.format_with_timestamp(plate, current_time))
                    
            # Format identification documents
            id_documents = []
            for doc in analysis_data.get('Identification documents', []):
                if doc:
                    id_documents.append(self.format_with_timestamp(doc, current_time))
                    
            # Format scene sentiment
            sentiment = analysis_data.get('Scene sentiment', {})
            if isinstance(sentiment, dict):
                sentiment_data = {
                    'assessment': sentiment.get('assessment', ''),
                    'justification': sentiment.get('justification', ''),
                    'timestamp': current_time
                }
            elif isinstance(sentiment, str):
                sentiment_data = {
                    'assessment': sentiment,
                    'justification': '',
                    'timestamp': current_time
                }
            else:
                sentiment_data = {
                    'assessment': '',
                    'justification': '',
                    'timestamp': current_time
                }
                
            # Format people nearby
            people_data = []
            for person in analysis_data.get('People nearby', []):
                if person:
                    people_data.append(self.format_with_timestamp(person, current_time))
                    
            # Format risk analysis
            risk = analysis_data.get('Risk analysis', {})
            if isinstance(risk, dict):
                risk_data = {
                    'risks': risk.get('risks', []),
                    'severity': risk.get('severity', 'low'),
                    'timestamp': current_time
                }
            else:
                risk_data = {
                    'risks': [],
                    'severity': 'low',
                    'timestamp': current_time
                }
                
            # Insert formatted data into database
            insert_sql = """
            INSERT INTO visual_analysis 
            (frame_description, license_plates, scene_sentiment, 
            sentiment_justification, people_nearby, risk_analysis, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            # Extract sentiment values
            sentiment_value = sentiment_data.get('assessment', '')
            sentiment_justify = sentiment_data.get('justification', '')
            
            values = (
                [json.dumps(fd) for fd in frame_descriptions],  # Convert dicts to JSON strings in array
                [json.dumps(lp) for lp in license_plates],      # Convert dicts to JSON strings in array
                [sentiment_value],                              # Text array
                [sentiment_justify],                            # Text array
                json.dumps(people_data),                        # JSONB
                json.dumps(risk_data),                          # JSONB
                datetime.now()
            )
            
            self.cursor.execute(insert_sql, values)
            analysis_id = self.cursor.fetchone()[0]
            
            # Store source path in analysis_messages
            if source_path:
                message_sql = """
                INSERT INTO analysis_messages
                (analysis_id, role, content, created_at)
                VALUES (%s, %s, %s, %s);
                """
                
                content = json.dumps({
                    "source_path": source_path,
                    "id_documents": id_documents
                })
                
                self.cursor.execute(message_sql, (
                    analysis_id,
                    'system',
                    content,
                    datetime.now()
                ))
            
            self.conn.commit()
            return analysis_id
            
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Database error: {e}")
            return None
        finally:
            self.close()
            
    def get_analysis_by_id(self, analysis_id):
        """
        Retrieve analysis results by ID
        
        Args:
            analysis_id (int): The analysis ID
            
        Returns:
            dict: The analysis data or None if not found
        """
        import json
        
        try:
            if not self.connect():
                return None
                
            sql = """
            SELECT id, frame_description, license_plates, scene_sentiment, 
                   sentiment_justification, people_nearby, risk_analysis, created_at
            FROM visual_analysis
            WHERE id = %s
            """
            
            self.cursor.execute(sql, (analysis_id,))
            row = self.cursor.fetchone()
            
            if not row:
                return None
                
            # Fetch any ID documents from analysis_messages
            id_docs_sql = """
            SELECT content FROM analysis_messages
            WHERE analysis_id = %s AND role = 'system'
            """
            
            self.cursor.execute(id_docs_sql, (analysis_id,))
            message_row = self.cursor.fetchone()
            
            id_documents = []
            if message_row:
                message_content = json.loads(message_row[0])
                id_documents = message_content.get('id_documents', [])
            
            # Convert JSON strings in arrays back to objects
            frame_desc = [json.loads(fd) for fd in row[1]] if row[1] else []
            license_plates = [json.loads(lp) for lp in row[2]] if row[2] else []
            
            # Parse JSONB fields
            people_nearby = json.loads(row[5]) if row[5] else []
            risk_analysis = json.loads(row[6]) if row[6] else {}
            
            return {
                'id': row[0],
                'Frame description': [fd.get('description', '') for fd in frame_desc],
                'License plates': license_plates,
                'Identification documents': id_documents,
                'Scene sentiment': {
                    'assessment': row[3][0] if row[3] and len(row[3]) > 0 else '',
                    'justification': row[4][0] if row[4] and len(row[4]) > 0 else ''
                },
                'People nearby': people_nearby,
                'Risk analysis': risk_analysis,
                'created_at': row[7].isoformat() if row[7] else None
            }
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            self.close()

# end here --------------------------------------------------------------------------


class Qwen2_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = mm.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self._model_loaded = False
        self._last_used = 0

    def load_model(self, quantization=None, attention="eager"):
        if self._model_loaded:
            return

        torch.cuda.empty_cache()
        gc.collect()
        check_memory()

        torch.manual_seed(42)

        model_id = "qwen/Qwen2.5-VL-3B-Instruct"
        self.model_checkpoint = os.path.join(
            "models/prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            min_pixels=153600,
            max_pixels=409600,
        )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if self.bf16_support else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        torch.cuda.empty_cache()
        gc.collect()
        check_memory(self.device)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
            device_map=self.device,
            attn_implementation=attention,
            quantization_config=quantization_config,
        )

        self._model_loaded = True

    def inference(
        self,
        system_prompt,
        temperature,
        max_new_tokens,
        source_path=None,
    ):
        video = VideoFileClip(source_path)

        video_duration = video.duration
        batch_duration = 2

        # Video batching
        batches = []
        for start_time in range(0, int(video_duration), batch_duration):
            end_time = min(start_time + batch_duration, video_duration)
            batch = video.subclipped(start_time, end_time)
            batches.append(batch)

        for idx, batch in enumerate(batches):
            batch_filename = f"data/batch_{idx+1}.mp4"
            batch.write_videofile(batch_filename, codec="libx264")

        data_dir = Path("./data")

        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)  # Video file path

        if not self._model_loaded:
            print("Model not loaded!")
            self.load_model()

        with torch.no_grad():
            messages = [
                {
                    "role": "system",
                    "content": """You are a detailed visual analysis system.
                    Analyze the provided visual content and output the following information in structured JSON format: 
                    Frame description: Detailed description of all visible objects, locations, lighting, and activity. 
                    License plates: All visible car license plates exactly as they appear (or partially if obscured). 
                    Identification documents: All visible ID documents, there ID numbers and there time
                    Scene sentiment: Assessment of whether the environment appears peaceful, neutral, or dangerous, with justification. 
                    People nearby: Description of all people in focus, including estimated age range, clothing, behavior, and interactions. 
                    Risk analysis: Any signs of risk, conflict, or abnormal activity.""",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": file_path},
                        {"type": "text", "text": system_prompt},
                    ],
                },
            ]

            # Preparation for inference
            system_prompts = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            check_memory(self.device)

            inputs = self.processor(
                text=system_prompts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.device)

            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=1.2,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            stringified = json.dumps(result)

            print(stringified)
            return stringified


model_manager = Qwen2_VQA()
