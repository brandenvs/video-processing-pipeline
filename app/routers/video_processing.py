import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import torch
import numpy as np
import gc
import asyncio
from datetime import datetime
from app.routers.db_functions import DB_CONFIG # imported db_config
import psycopg2

from torchvision.transforms import ToPILImage
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,

)
from PIL import Image
from qwen_vl_utils import process_vision_info

from app.routers import model_management as mm

from fastapi import APIRouter


from concurrent.futures import ThreadPoolExecutor
import atexit
import functools

from moviepy import VideoFileClip


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Dictionary with DB_CONFIG
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# Connect to "PostgreSQL" database
def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"Database connection established to {DB_CONFIG['database']} at {DB_CONFIG['host']}")
        return conn, cursor
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return None, None


class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = "Identify key data and fillout the given system schema"
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

    total_mem = mm.get_total_memory() / (1024 * 1024 * 1024)
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

    inference_task = functools.partial(model_manager.inference, **request_body.model_dump())
    results = await loop.run_in_executor(executor, inference_task)
    

    for analysis_data in results:
        analysis_ids = []

        # Store in database
        db_helper = Db_helper()
        analysis_id = db_helper.insert_analysis(analysis_data, source_path=request_body.source_path)

        if analysis_id:
            # analysis_data['id'] = analysis_id
            analysis_ids.append(analysis_id)
        
    return {
        "status": "success",
        "analysis_ids": analysis_ids,
        "results": results,
    }

# document to text in the database, and the audio to text, 

class Db_helper:
    def __init__(self):
        self.db_config = DB_CONFIG # Dylan -> Hook up Db
        self.conn = None
        self.cursor = None

    def insert_analysis(self, analysis_data, source_path=None):
        try:
            self.conn = psycopg2.connect(**self.db_config) # Dylan -> Hook up Db
            self.cursor = self.conn.cursor()
            
            # Extract data from analysis_data
            frame_description = analysis_data.get('Frame description', '')
            objects_detected = analysis_data.get('Objects dectected', [])
            objects_detected = ", ".join(objects_detected)

            license_plates = analysis_data.get('License plates', [])
            license_plates = ", ".join(license_plates)

            scene_sentiment = analysis_data.get('Scene sentiment', '')
            risk_analysis = analysis_data.get('Risk analysis', '')
            
            insert_sql = """
            INSERT INTO visual_analysis
            (frame_description, objects_detected, license_plates, scene_sentiment, 
            risk_analysis, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            values = (
                frame_description,
                objects_detected,
                scene_sentiment,
                license_plates,
                risk_analysis,
                datetime.now()
            )
            print(frame_description, objects_detected, license_plates, scene_sentiment, 
            risk_analysis, datetime.now())
            
            self.cursor.execute(insert_sql, values)
            analysis_id = self.cursor.fetchone()[0]
            
            # Store ID documents in messages table
            # if source_path and 'Identification documents' in analysis_data:
            #     message_sql = """
            #     INSERT INTO analysis_messages
            #     (analysis_id, role, content, created_at)
            #     VALUES (%s, %s, %s, %s);
            #     """
                
            #     content = json.dumps({
            #         "source_path": source_path,
            #         "id_documents": analysis_data.get('Identification documents', [])
            #     })
                
            #     self.cursor.execute(message_sql, (
            #         analysis_id,
            #         'system',
            #         content,
            #         datetime.now()
            #     ))
            
            self.conn.commit()
            return analysis_id
            
        except Exception as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.rollback()
            return None
            
        finally:
            # Always close connections
            if self.cursor:
                self.cursor.close()
            if self.conn and not self.conn.closed:
                self.conn.close()


class Qwen2_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = mm.get_torch_device()
        self.dtype = None
        self._model_loaded = False

    def process_generated_response(self, generated_response: str, sequence_no: int):
        if generated_response.startswith("```json"):
            try:
                lines = generated_response.strip().splitlines()
                json_content = "\n".join(lines[1:-1])
    
                json_response = json.loads(json_content)
                spread_response = {**json_response, **{"sequence_no": sequence_no}}
                return spread_response

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return e

        return generated_response

    def load_model(self):
        if self._model_loaded:
            return

        torch.cuda.empty_cache()
        print(gc.collect())
        check_memory()
        torch.manual_seed(42)

        model_id = "qwen/Qwen2.5-VL-3B-Instruct"
        self.model_checkpoint = os.path.join("models/prompt_generator", os.path.basename(model_id))

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        self.dtype = (
            torch.float16 if mm.should_use_fp16(self.device)
            else torch.bfloat16 if mm.should_use_bf16(self.device)
            else torch.float32
        )

        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=self.dtype,
            device_map=self.device,
            quantization_config=quantization_config
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint
        )
        self._model_loaded = True

    def inference(
        self,
        system_prompt,
        max_new_tokens,
        source_path=None,
    ):
        video = VideoFileClip(source_path)

        results = []

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

        if not self._model_loaded:
            self.load_model()

        sequence_no = 0
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)  # Video file path
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert visual analysis system.
                    Analyze the video and structure a concise JSON response structured as follows.
                    Frame activity: (A concise description of is happening within the video).
                    Objects dectected: (A list of objects identified).
                    License plates: (A list of car license plates and the car make and year).
                    Scene sentiment: (Either 'neutral', 'dangerous' or 'unknown').
                    Risk analysis: (Any signs of risk, conflict, or abnormal activity).""",
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
                videos=video_inputs,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            generated_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            sequence_no += 1
            result = self.process_generated_response(generated_response[0], sequence_no)
            results.append(result)
        return results

model_manager = Qwen2_VQA()
