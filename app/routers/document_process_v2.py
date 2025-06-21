

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
from typing import Dict, Optional
import uuid
from fastapi import APIRouter
from pydantic import BaseModel
import urllib

import torch
from routers import model_management as mm

from app.routers.video_processing import FieldDefinition
import functools
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig # type: ignore


class BaseProcessor(BaseModel):
  system_prompt: Optional[str] = "Process the document using the structured output"
  max_tokens: Optional[int] = 512
  model_id: Optional[str] = "qwen/Qwen2.5-VL-3B-Instruct"
  source_path: str

router = APIRouter()

router = APIRouter(
  prefix="/document",
  tags=["process"],
  responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def cleanup_executor():
  executor.shutdown(wait=True)

atexit.register(cleanup_executor)

@router.post('/process/')
async def process_document(request_body: BaseProcessor):
  print(request_body)
  
  filename = f'/home/ubuntu/adp-video-pipeline/source_data/{uuid.uuid4()}document.pdf'
  result = urllib.request.urlretrieve(request_body.source_path, filename)
  
  loop = asyncio.get_running_loop()  
  if (result):
    process_document = functools.partial(model_manager.inference, filename)    
    processed_document_response = await loop.run_in_executor(executor, process_document)

  return {
    "status": "success",
    "processed_document_response": processed_document_response
  }


class Qwen2_VQA:
  def __init__(self):
    self.model_checkpoint = None
    self.processor = None
    self.model = None
    self.device = mm.get_torch_device()
    self.dtype = None
    self._model_loaded = False

  def load_model(self, model_id: str):
    if self._model_loaded:
      return

    torch.cuda.empty_cache()
    print('Total Garbage Collected', gc.collect())
    torch.manual_seed(46)

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

    if self.dtype != torch.float16:
      quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=self.dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
      )
    else:
      quantization_config = None

    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      self.model_checkpoint,
      torch_dtype=self.dtype,
      attn_implementation="eager",
      device_map=self.device,
      quantization_config=quantization_config,
    )

    self.processor = AutoProcessor.from_pretrained(
      self.model_checkpoint,
      min_pixels = 256*28*28,
      max_pixels = 1280*28*28,
    )

    self._model_loaded = True

  def inference(self, system_prompt: str, max_token: int, model_id: str, source_path: str):
    if not self._model_loaded:
      print('>>> Loading model into memory')
      self.load_model(model_id)
    
    # Build your logic here
    messages = [
      {
        "role": "system",
        "content": f"You are an expert document analysis system. Analyze the PDF and extract structured data using the following JSON schema",
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": source_path},  # For PDF pages as images
          {"type": "text", "text": system_prompt},
        ],
      },
    ]



model_manager = Qwen2_VQA()