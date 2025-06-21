import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
import uuid
from fastapi import APIRouter
from pydantic import BaseModel
import urllib.request
import json
import time
import logging
from pypdf import PdfReader
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path

import torch
from app.routers import model_management as mm

import functools
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig


class BaseProcessor(BaseModel):
  system_prompt: Optional[str] = "Analyze this document and extract all key information as a structured JSON output"
  max_tokens: Optional[int] = 1024
  model_id: Optional[str] = "qwen/Qwen2.5-VL-3B-Instruct"
  dpi: Optional[int] = 150
  source_path: str


def normalize_field_name(field_name: str) -> str:
    if not field_name:
        return ""
    
    field_name = field_name.lower().strip()
    field_name = " ".join(field_name.split())
    field_name = field_name.replace(" ", "_")
    return field_name

def convert_to_human_readable_label(field_name: str) -> str:
    if not field_name:
        return ""
    
    readable = field_name.replace('_', ' ')
    readable = ' '.join(word.capitalize() for word in readable.split())
    
    return readable

def convert_pdf_to_images(pdf_path: str, dpi: int = 150) -> List[tuple]:
  image_paths = convert_from_path(
    pdf_path,
    dpi=dpi,
    fmt="jpeg",
    output_folder='pdf_pages', # MARK Todo
    paths_only=True
  )
  return image_paths

router = APIRouter()

router = APIRouter(
  prefix="/document",
  tags=["process"],
  responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=2)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def cleanup_executor():
  executor.shutdown(wait=True)

atexit.register(cleanup_executor)

@router.post('/process/')
async def process_document(request_body: BaseProcessor): 
  filename = f'/home/ubuntu/adp-video-pipeline/source_data/{uuid.uuid4()}document.pdf'
  source_path, res = urllib.request.urlretrieve(request_body.source_path, filename)
  
  loop = asyncio.get_running_loop()  
  if (source_path):
    process_document = functools.partial(
      model_manager.inference, 
      request_body.system_prompt,
      request_body.max_tokens,
      request_body.model_id,
      source_path,
      request_body.dpi
    )    
    processed_document_response = await loop.run_in_executor(executor, process_document)
    final_schema = join_sequences(processed_document_response)
  
  if len(processed_document_response) > 0:
    gc.collect()
    return {
      "status": "success",
      "processed_document_response": processed_document_response,
      "final_schema": final_schema
    }
  else:
    return {
      "status": "error",
      "Message": "Something went wrong (～￣(OO)￣)ブ"
    } 

def join_sequences(sequences):
    combined = {}
    for seq in sequences:
        # Remove sequence_no and merge the rest
        seq_copy = seq.copy()
        seq_copy.pop('sequence_no', None)
        combined.update(seq_copy)
    return combined


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
    torch.manual_seed(46)

    self.model_checkpoint = os.path.join(
        "models/prompt_generator", os.path.basename(model_id)
    )

    if not os.path.exists(self.model_checkpoint):
      from huggingface_hub import snapshot_download
      snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

    if mm.should_use_fp16(self.device):
      self.dtype = torch.float16
    else:
      self.dtype = torch.float32

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
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
    self._model_loaded = True

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
        return {
          "error": f"JSON parsing error: {str(e)}",
          "sequence_no": sequence_no
        } 

    try:
      json_response = json.loads(generated_response)
      spread_response = {**json_response, **{"sequence_no": sequence_no}}
      return spread_response
    except json.JSONDecodeError as e:
      print(f"JSON decode error: {e}")
      return {
        "error": f"JSON parsing error: {str(e)}",
        "sequence_no": sequence_no,
        "raw_sample": generated_response
      }

  def process_page(self, image_path: str, system_prompt: str, max_token: int, page_num: int) -> Dict[str, Any]:
    page_specific_prompt = """
    You are an expert document analyzer. 
    Examine this document content and then contextual extract the following data using structured output:
    - label: The field name.
    - description: A concise description that a Video Analysis model can use to interpret the field more accurately.
    - field type: Either 'text', 'number', or 'list'.
    - options: A list of the options for checkboxes (can be and empty list).
    - required: wether or not the field should be made mandatory.
    
    Use headings and subheadings as context but do not include them with your response.
    
    Use the following Structured Output Schema: 
    { [fieldVar: string]: { label: string, field_type: string, description: string, options: [], required: boolean }
    For example: { preventive_measures: { label: "Preventative Measures",  description: "What preventive measures can be implemented to avoid similar incidents in the future?", options: [], required: "true" }}
    NOTE: fieldVar must match the label but should be all lowercase letters and whitespaces should be replaced with underscores - 
    """
    messages = [
      {
        "role": "system",
        "content": page_specific_prompt,
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": image_path},
          {"type": "text", "text": system_prompt},
        ],
      },
    ]    
    
    image_inputs, video_inputs = process_vision_info(messages)  

    system_prompts = self.processor.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=False,
      return_tensors="pt"
    )
      
    inputs = self.processor(
      text=system_prompts,
      images=image_inputs,
      padding=True,
      return_tensors="pt",
    )
    inputs = inputs.to(self.device)
      
    outputs = self.model.generate(**inputs, max_new_tokens=max_token)
    
    generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]

    generated_response = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    generated_response = self.process_generated_response(generated_response[0], page_num)
    print('generated_response', generated_response)
    return {
      **generated_response,
      "page_number": page_num
    }

  def inference(self, system_prompt: str, max_token: int, model_id: str, source_path: str, dpi: int = 150) -> List:
    start_time = time.time()
    page_paths = []
    page_results = []

    try:
      if not self._model_loaded:
        self.load_model(model_id)
      
      if source_path.endswith('.pdf'):
        page_paths = convert_pdf_to_images(source_path, dpi=dpi)

        if not page_paths:
          return {"error": "Failed to convert PDF to images", "processing_time_seconds": 0}

      for idx, image_path in enumerate(page_paths):          
        result = self.process_page(image_path, system_prompt, max_token, idx + 1)
        page_results.append(result)

      # Clean up PDF images
      [os.remove(os.path.join('pdf_pages', f)) for f in os.listdir('pdf_pages') if os.path.isfile(os.path.join('pdf_pages', f))]

      joined_page_results = [item for item in page_results if item is not None]
      return joined_page_results

    except Exception as e:
      logging.exception(f"Error processing document: {e}")
      return {"error": str(e), "processing_time_seconds": round(time.time() - start_time, 2)}


model_manager = Qwen2_VQA()