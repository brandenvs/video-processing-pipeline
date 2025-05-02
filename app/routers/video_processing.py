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

# Have a dictionary with DB_CONFIG here ...

class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = "Perform a detailed analysis of the given data"
    temperature: Optional[float] = 0.5
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None

# Write a function that will connect to PostgreSQL database here ...

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

    # inference_task = functools.partial(
    #     model_manager.inference, **request_body.model_dump()
    # )
    # generated_data = await loop.run_in_executor(executor, inference_task)
    # response = json.loads(generated_data)

    data = {'Frame description': 'Some Text',}

    # Db writes here ...

    # Columns: Table(Videos Processed)
    # Frame description
    # License plates
    # People nearby
    # Risk analysis

    # return [{"results": response}]


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
