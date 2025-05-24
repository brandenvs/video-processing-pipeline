import csv
import json
import os
from pathlib import Path
import re
import time
from typing import Optional
import concurrent
from pydantic import BaseModel
import torch
import gc
import asyncio
from torch.nn.attention import SDPBackend, sdpa_kernel
from app.routers.database_service import Db_helper
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

import subprocess

# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     AutoProcessor,
#     BitsAndBytesConfig,
# )
from qwen_vl_utils import process_vision_info

from app.routers import model_management as mm

from fastapi import APIRouter


from concurrent.futures import ThreadPoolExecutor
import atexit
import functools

from moviepy import VideoFileClip


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = (
        "Identify key data and fillout the given system schema"
    )
    max_tokens: Optional[int] = 512
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


def check_memory(device=mm.get_torch_device):
    print("Device Loaded: ", device)

    total_mem = mm.get_total_memory(device) / (1024 * 1024 * 1024)
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

    scene_detection = functools.partial(
        batch_scene_detection, request_body.source_path
    )
    scene_detection_response = await loop.run_in_executor(executor, scene_detection)
    
    infer_obj = {
        **scene_detection_response,
        **request_body.model_dump(),
    }
    print(infer_obj)
    process_video = functools.partial(
        model_manager.inference_helper, **infer_obj
    )
    processed_video_response = await loop.run_in_executor(executor, process_video)

    # for analysis_data in results:
    #     analysis_ids = []

    #     # Store in database
    #     db_helper = Db_helper()
    #     analysis_id = db_helper.video_analysis(
    #         analysis_data, source_path=request_body.source_path
    #     )

    #     if analysis_id:
    #         # analysis_data['id'] = analysis_id
    #         analysis_ids.append(analysis_id)

    return {
        "status": "success",
        "scene_detection_response": scene_detection_response,
        "processed_video_response": processed_video_response
        # "analysis_ids": analysis_ids,
    }

def calculate_mean_content_val(stats_file: str) -> float:
    content_vals = []
    
    with open(f'segments/{stats_file}', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content_vals.append(float(row['content_val']))
    
    return sum(content_vals) / len(content_vals)


def get_content_val_stats(stats_file: str) -> dict:
    content_vals = []
    
    with open(stats_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content_vals.append(float(row['content_val']))
    
    content_vals.sort()
    n = len(content_vals)
    
    mean_val = sum(content_vals) / n
    median_val = content_vals[n // 2] if n % 2 == 1 else (content_vals[n // 2 - 1] + content_vals[n // 2]) / 2
    variance = sum((x - mean_val) ** 2 for x in content_vals) / n
    
    return {
        'count': n,
        'min': min(content_vals),
        'max': max(content_vals),
        'mean': mean_val,
        'median': median_val,
        'std': variance ** 0.5,
        'p25': content_vals[n // 4],
        'p75': content_vals[3 * n // 4],
        'p90': content_vals[int(0.9 * n)]
    }


def batch_scene_detection(video):
    response = subprocess.run([
        'scenedetect', '--config', 'scenedetect.cfg',
        '-i', video, 
        '--output', 'segments',
        'detect-content', 'split-video', '--filename', '$SCENE_NUMBER', '--copy'
    ], check=True, capture_output=True, text=True)
    segments = [f for f in os.listdir('segments') if f.endswith('.mp4')]
    
    for segment in segments:
        stats_file = f'{segment}.stats.csv'
        
        response = subprocess.run([
            'scenedetect', '--config', 'scenedetect.cfg',
            '-i', f'segments/{segment}',
            '--output', 'segments',         
            '--stats', stats_file,
            'detect-content', 'split-video', '--filename', '$SCENE_NUMBER', '--copy'

        ], check=True, capture_output=True, text=True)
    
        mean_content_val = calculate_mean_content_val(stats_file)
    
        content_threshold = 15
        if mean_content_val < content_threshold:
            _ = segments.pop(segments.index(segment))

    response = {  
        'segments': segments,
        "mean_content_val": round(mean_content_val, 3)
    }
    # print(json.dumps(response, indent=2))
    return response

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
        self.model_checkpoint = os.path.join(
            "models/prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        # MARK: Precision
        if mm.should_use_fp16(self.device):
            self.dtype = torch.float16
        elif mm.should_use_fp16(self.device):
            self.dtype = torch.float16
        else:
            torch.float32

        print(f'>>> Selected DType: {self.dtype}')

        if self.dtype != torch.float16:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Get architecture details if CUDA is available
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            print(f'>>> CUDA Device: {device_name}')
            print(f'>>> Compute Capability: {capability[0]}.{capability[1]}')

            # Classify architecture
            major = capability[0]
            if major >= 8:
                print(">>> Architecture: Ampere or newer (FlashAttention-compatible)")
            elif major == 7:
                print(">>> Architecture: Turing or Volta (not compatible with FlashAttention)")
            else:
                print(">>> Architecture: Older (not compatible)")

        else:
            print(">>> CUDA not available")
    
        print(f'>>> Selected Device: {self.device}')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
            quantization_config=quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            min_pixels = 256*28*28,
            max_pixels = 1280*28*28,
        )

        self._model_loaded = True

    def inference_helper(self, system_prompt, max_tokens, segments: list, mean_content_val, source_path):
        responses = []
        sequence = 0
        
        if not self._model_loaded:
            print('>>> Loading model into memory')
            self.load_model()

        segments.sort(reverse=True)

        for seq, segment in enumerate(segments):
            sequence += 1
            segment_path = os.path.join('segments', segment)
            subprocess.run([
                'ffmpeg', '-y', '-i', segment_path,
                '-r', '5',
                '-c:v', 'libx264',
                '-crf', '23',
                segment_path
            ], check=True)
            responses.append(self.inference(segment_path, system_prompt, max_tokens, seq))
        return responses

        
    def inference(self, segment_path, system_prompt, max_tokens, seq):
        start_time = time.time() # timer

        # MARK: Batching
        # for start in range(0, int(video_duration), batch):
        #     end = min(start + batch, video_duration)
        #     segments.append((start, end))
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     print('>>> Processing segments')
        #     video_to_batches = {
        #         executor.submit(self.process_batch, 
        #             start, end, source_path, idx):
        #             idx for idx, (start, end) in enumerate(segments)
        #     }

        # for future in concurrent.futures.as_completed(video_to_batches):
        #     batch = future.result()
        #     if batch:
        #         batches.append(batch)

        # elapsed_time = time.time() - start_time # timer logged
        # print(f"Batching completed in {elapsed_time:.2f} seconds")

        # batches.sort(key=lambda filename: int(filename.split('_')[-1].split('.')[0]))

        messages = [
            {
                "role": "system",
                "content": """You are an expert visual analysis system.
                Analyze the video and structure a concise JSON response structured as follows.

                Frame activity - A concise description of is happening within the video.
                Objects detected - A list of objects identified within a close proximity.
                Cars detected - A list of JSON objects with the following properties: Car license plate(if visible), Color and Model.
                People detected - A list of JSON objects with the following properties: Estimated Height, Age, Race, Emotional state, and proximity
                Scene sentiment - Either 'neutral', 'dangerous' or 'unknown'.
                ID cards detected - A list of JSON objects with the following properties: Surname, Names, Sex, Nationality, Identity Number, Date of Birth, Country of Birth, Status""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": segment_path},
                    {"type": "text", "text": system_prompt},
                ],
            },
        ]

        print('>>> Preparation for inference')
        system_prompts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        check_memory(self.device)

        print('Preparing inputs ....')
        inputs = self.processor(
            text=system_prompts,
            videos=video_inputs,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs =  inputs.to(self.device)

        print('>>> Inference: Generation of the output')
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        #     outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        generated_response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_response = self.process_generated_response(generated_response[0], seq)

        finished_in = time.time() - start_time
        response = {
            **generated_response,
            "finished_in": round(finished_in, 3)
        }
        return response


model_manager = Qwen2_VQA()

