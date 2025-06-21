import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import uuid
from fastapi import APIRouter
from pydantic import BaseModel
import urllib.request
import json
import re
import time

import torch
from routers import model_management as mm

from app.routers.video_processing import FieldDefinition
import functools
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig # type: ignore


class BaseProcessor(BaseModel):
  system_prompt: Optional[str] = "Analyze this document and extract all key information as a structured JSON output"
  max_tokens: Optional[int] = 1024
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
    process_document = functools.partial(
      model_manager.inference, 
      request_body.system_prompt or "Analyze this document and extract all key information as a structured JSON output",
      request_body.max_tokens or 1024,
      request_body.model_id or "qwen/Qwen2.5-VL-3B-Instruct",
      filename
    )    
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
    self.tokenizer = None
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
    else:
      self.dtype = torch.float32

    # Optimize memory usage with 4-bit quantization for non-float16 models
    if self.dtype != torch.float16:
      quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=self.dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
      )
    else:
      quantization_config = None

    # Load model with optimized configuration
    print(f"Loading model with dtype: {self.dtype} on device: {self.device}")
    
    # Use context manager to ensure efficient memory handling
    with torch.cuda.amp.autocast(enabled=self.dtype==torch.float16):
      self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        self.model_checkpoint,
        torch_dtype=self.dtype,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
        device_map=self.device,
        quantization_config=quantization_config,
      )

    self.processor = AutoProcessor.from_pretrained(
      self.model_checkpoint,
      min_pixels = 256*28*28,
      max_pixels = 1280*28*28,
    )
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
    self._model_loaded = True

  def _extract_json_from_text(self, text: str) -> Dict:
    """Extract JSON object from text response"""
    # Try to find JSON structure in response
    json_pattern = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
    if json_pattern:
        try:
            json_str = json_pattern.group(0)
            # Handle potential trailing text by finding last closing brace
            if json_str.count('{') != json_str.count('}'):
                # Find position of the last complete JSON object
                open_braces = 0
                for i, char in enumerate(json_str):
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            json_str = json_str[:i+1]
                            break
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If JSONDecodeError, try more aggressive cleaning
            try:
                # Remove any trailing or leading text outside of braces
                cleaner_json = re.search(r'(\{.*\})', json_str, re.DOTALL).group(1)
                return json.loads(cleaner_json)
            except (json.JSONDecodeError, AttributeError):
                pass
    
    # Fallback: If we can't extract valid JSON, return the raw text
    return {"raw_text": text}

  def inference(self, system_prompt: str, max_token: int, model_id: str, source_path: str) -> Dict[str, Any]:
    """
    Optimized document processing using a single model call to handle both document 
    type detection and content extraction.
    """
    start_time = time.time()
    
    if not self._model_loaded:
      print('>>> Loading model into memory')
      self.load_model(model_id)
    
    # Create a single comprehensive prompt that handles document type detection and content extraction
    combined_prompt = f"""
    You are an expert document analyzer. Your task is to:

    1. Identify the document type (e.g., incident_report, invoice, form, etc.)
    2. Extract all information from the document in a structured JSON format
    3. Process the information according to these instructions: {system_prompt}

    IMPORTANT: 
    - Your response must be a valid JSON object
    - Include a "document_type" field that identifies the type of document
    - Include all visible fields and their values from the document
    - If a field has no value, include it with an empty string
    - Use appropriate data types for values (strings, numbers, arrays, etc.)
    - Organize information logically and hierarchically when appropriate
    """
    
    # Create a single message for processing
    messages = [
      {
        "role": "system",
        "content": combined_prompt,
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": source_path},
          {"type": "text", "text": "Analyze this document and provide a comprehensive JSON structure with document type and all content."},
        ],
      },
    ]
    
    # Process the messages
    print("Processing document with combined prompt...")
    
    # Use torch.no_grad to optimize memory usage during inference
    with torch.no_grad():
      inputs = self.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
      ).to(self.device)
      
      # Generate response with optimized settings
      outputs = self.model.generate(
        inputs,
        max_new_tokens=max_token,
        do_sample=False,
        # Additional optimization parameters
        temperature=0.1,  # Lower temperature for more deterministic outputs
        repetition_penalty=1.1,  # Slight penalty to avoid repetitions
      )
    
    # Decode and extract the structured response
    response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract JSON from response
    result = self._extract_json_from_text(response_text)
    
    # Add processing time for monitoring
    end_time = time.time()
    result["processing_time_seconds"] = round(end_time - start_time, 2)
    
    # Ensure document_type is always included
    if "document_type" not in result and "tag" not in result:
        # Try to infer document type from result keys if missing
        possible_types = ["report", "form", "invoice", "letter", "document"]
        for key in result.keys():
            for doc_type in possible_types:
                if doc_type in key.lower():
                    result["document_type"] = doc_type
                    break
            if "document_type" in result:
                break
        
        if "document_type" not in result:
            result["document_type"] = "document"
    
    # For backward compatibility
    if "document_type" in result and "tag" not in result:
        result["tag"] = result["document_type"]
    elif "tag" in result and "document_type" not in result:
        result["document_type"] = result["tag"]
    
    print(f"Document processed in {result['processing_time_seconds']} seconds")
    return result


model_manager = Qwen2_VQA()