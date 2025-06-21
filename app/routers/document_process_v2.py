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
import time
import tempfile
import logging
from pypdf import PdfReader
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
import shutil

import torch
from app.routers import model_management as mm

from app.routers.video_processing import FieldDefinition
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
  print(request_body)
  
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

    print(f"Loading model with dtype: {self.dtype} on device: {self.device}")
    
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
      max_pixels = 1024*28*28,
    )
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
    self._model_loaded = True

  def process_generated_response(self, generated_response: str, sequence_no: int):
    print(f'GENERATED RESPONSE FOR BATCH: {sequence_no}', generated_response)
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
    You are an expert document analyzer. Examine this document and provide a complete structured JSON output of all information present in the document. 
    Use the following JSON Structured Output Schema for each field: { label: string; field_type: string; description: string; required: boolean }
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
      
    outputs = self.model.generate(
      **inputs,
      max_new_tokens=max_token,
      do_sample=False,
      temperature=0.1,
      repetition_penalty=1.1,
    )
    
    generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]

    generated_response = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    generated_response = self.process_generated_response(generated_response[0], page_num)
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
        print('>>> Loading model into memory')
        self.load_model(model_id)
      
      if source_path.endswith('.pdf'):
        page_paths = convert_pdf_to_images(source_path, dpi=dpi)

        if not page_paths:
          return {"error": "Failed to convert PDF to images", "processing_time_seconds": 0}

      for idx, image_path in enumerate(page_paths):          
        result = self.process_page(image_path, system_prompt, max_token, idx + 1)
        print(json.dumps(result, indent=2))
        page_results.append(result)

      # Clean up PDF images
      [os.remove(os.path.join('pdf_pages', f)) for f in os.listdir('pdf_pages') if os.path.isfile(os.path.join('pdf_pages', f))]

      joined_page_results = [item for item in page_results if item is not None]
      return joined_page_results

    except Exception as e:
      logging.exception(f"Error processing document: {e}")
      return {"error": str(e), "processing_time_seconds": round(time.time() - start_time, 2)}

  def _extract_document_text(self, file_path: str) -> str:
    if not os.path.exists(file_path) or not file_path.lower().endswith('.pdf'):
        return ""
        
    try:
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""
  
  def _extract_fields_with_ai(self, document_text: str, document_type: str = "form") -> List[str]:
    prompt = f"""
    Analyze this {document_type} document text and identify all form field names. 
    Return ONLY the normalized field names (snake_case) as a Python list.
    Example: ["first_name", "last_name", "date_of_birth"]
    
    Document text:
    {document_text[:4000]}
    """
    
    if not self._model_loaded:
        self.load_model("qwen/Qwen2.5-VL-3B-Instruct")
        
    try:
        messages = [
            {"role": "system", "content": "You are a document field extraction assistant. Extract form fields from documents."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=500,
                do_sample=False,
                temperature=0.1
            )
        
        response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            if start_idx >= 0 and end_idx > start_idx:
                list_text = response_text[start_idx:end_idx+1]
                fields_list = json.loads(list_text.replace("'", '"'))
                if isinstance(fields_list, list):
                    return fields_list
        except:
            pass
        
        return []
        
    except Exception as e:
        logging.error(f"Error extracting fields with AI: {e}")
        return []

  def _extract_json_from_text(self, text: str) -> Dict:
    try:
        start_idx = text.find('{')
        if start_idx == -1:
            return {"raw_text": text}
        
        open_braces = 0
        json_end = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                open_braces += 1
            elif text[i] == '}':
                open_braces -= 1
                if open_braces == 0:
                    json_end = i + 1
                    break
        
        if open_braces != 0:
            return {"raw_text": text}
            
        json_str = text[start_idx:json_end]
        return json.loads(json_str)
    except:
        return {"raw_text": text}
  
  def _combine_page_results(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not page_results:
      return {"error": "No page results to combine"}
    
    combined = {
      "document_type": self._determine_document_type(page_results),
      "pages": page_results,
      "combined_content": {}
    }
    
    first_page = page_results[0]
    for key, value in first_page.items():
      if key not in ["page_number", "raw_text", "processing_time_seconds"] and key not in combined:
        combined[key] = value
    
    all_fields = {}
    
    for page in page_results:
      page_num = page.get("page_number", 0)
      
      for key, value in page.items():
        if key in ["page_number", "document_type", "raw_text", "processing_time_seconds"]:
          continue
          
        if key not in all_fields:
          all_fields[key] = {
            "value": value,
            "sources": [page_num]
          }
        else:
          if all_fields[key]["value"] != value and value:
            if isinstance(all_fields[key]["value"], list):
              if value not in all_fields[key]["value"]:
                all_fields[key]["value"].append(value)
            else:
              all_fields[key]["value"] = [all_fields[key]["value"], value]
            
          if page_num not in all_fields[key]["sources"]:
            all_fields[key]["sources"].append(page_num)
    
    for key, data in all_fields.items():
      combined["combined_content"][key] = data["value"]
    
    return combined
  
  def _determine_document_type(self, page_results: List[Dict[str, Any]]) -> str:
    type_counts = {}
    
    for page in page_results:
      doc_type = page.get("document_type", "")
      if not doc_type and "tag" in page:
        doc_type = page.get("tag", "")
      
      if doc_type:
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    if type_counts:
      most_common = max(type_counts.items(), key=lambda x: x[1])[0]
      return most_common
    
    return "document"

class QwenDocumentIntegrator(Qwen2_VQA):
    pass


model_manager = Qwen2_VQA()