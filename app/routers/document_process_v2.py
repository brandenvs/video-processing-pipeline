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
  source_path: str
  document_type: Optional[str] = "form"
  document_key: Optional[str] = None
  client_id: Optional[str] = None
  max_pages: Optional[int] = None
  batch_size: Optional[int] = 3
  dpi: Optional[int] = 150


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

def convert_pdf_to_images(pdf_path: str, max_pages: Optional[int] = None, dpi: int = 150, batch_size: int = 3) -> List[tuple]:
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "pdf_pages")
        os.makedirs(temp_dir, exist_ok=True)
        
        unique_id = uuid.uuid4().hex[:8]
        
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            total_pages = len(pdf.pages)
        
        first_page = 1
        last_page = total_pages if max_pages is None else min(max_pages, total_pages)
        total_to_process = last_page - first_page + 1
        
        print(f"Converting PDF to images (processing {total_to_process} of {total_pages} total pages)")
        
        result_paths = []
        
        for batch_start in range(first_page, last_page + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, last_page)
            print(f"Converting batch: pages {batch_start} to {batch_end}")
            
            batch_images = convert_from_path(
                pdf_path, 
                first_page=batch_start, 
                last_page=batch_end,
                dpi=dpi,
                fmt="jpeg",
                output_folder=temp_dir,
                output_file=f"page_{unique_id}_",
                paths_only=True
            )
            
            for i, img_path in enumerate(batch_images):
                page_num = batch_start + i
                result_paths.append((page_num, img_path))
            
            gc.collect()
        
        print(f"Created {len(result_paths)} images from PDF")
        return result_paths
        
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []

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
  result = urllib.request.urlretrieve(request_body.source_path, filename)
  
  loop = asyncio.get_running_loop()  
  if (result):
    process_document = functools.partial(
      model_manager.inference, 
      request_body.system_prompt or "Analyze this document and extract all key information as a structured JSON output",
      request_body.max_tokens or 1024,
      request_body.model_id or "qwen/Qwen2.5-VL-3B-Instruct",
      filename,
      request_body.max_pages,
      request_body.batch_size or 3,
      request_body.dpi or 150
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

  def _process_single_page(self, image_path: str, system_prompt: str, max_token: int, page_num: int) -> Dict[str, Any]:
    page_specific_prompt = f"""
    You are analyzing page {page_num} of a document. Extract all information visible on this page into structured JSON.
    
    Additional instructions: {system_prompt}

    IMPORTANT: 
    - Response must be valid JSON with fields and values from this page
    - Include "page_number": {page_num} in your JSON
    - If you can identify the document type, include "document_type" field
    - Use descriptive field names for all extracted information
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
          {"type": "text", "text": f"Extract all information from this document page into JSON."},
        ],
      },
    ]
    
    print(f"Processing document page {page_num}")
    
    with torch.no_grad():
      inputs = self.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
      ).to(self.device)
      
      outputs = self.model.generate(
        inputs,
        max_new_tokens=max_token,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.1,
      )
    
    response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    result = self._extract_json_from_text(response_text)
    
    if "page_number" not in result:
        result["page_number"] = page_num
        
    return result

  def inference(self, system_prompt: str, max_token: int, model_id: str, source_path: str, 
                max_pages: Optional[int] = None, batch_size: int = 3, dpi: int = 150) -> Dict[str, Any]:
    start_time = time.time()
    page_image_tuples = []
    page_results = []
    
    result_id = uuid.uuid4().hex
    temp_results_file = os.path.join(tempfile.gettempdir(), f"doc_results_{result_id}.json")
    
    try:
        if not self._model_loaded:
          print('>>> Loading model into memory')
          self.load_model(model_id)
        
        if source_path.lower().endswith('.pdf'):
          page_image_tuples = convert_pdf_to_images(
              source_path, 
              max_pages=max_pages, 
              dpi=dpi,
              batch_size=batch_size
          )
          if not page_image_tuples:
            return {"error": "Failed to convert PDF to images", "processing_time_seconds": 0}
        else:
          page_image_tuples = [(1, source_path)]
        
        current_batch = []
        
        for page_num, image_path in page_image_tuples:
            print(f"Processing page {page_num} of {len(page_image_tuples)}")
            
            page_result = self._process_single_page(
                image_path, system_prompt, max_token, page_num
            )
            
            current_batch.append(page_result)
            page_results.append(page_result)
            
            if len(current_batch) >= batch_size:
                self._save_intermediate_results(page_results, temp_results_file)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                current_batch = []
                gc.collect()
            
            if source_path.lower().endswith('.pdf'):
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except Exception as e:
                    logging.warning(f"Failed to delete temporary image: {e}")
        
        combined_result = self._combine_page_results(page_results)
        
        end_time = time.time()
        combined_result["processing_time_seconds"] = round(end_time - start_time, 2)
        combined_result["total_pages_processed"] = len(page_results)
        
        print(f"Document processed in {combined_result['processing_time_seconds']} seconds, {len(page_results)} pages")
        return combined_result
        
    except Exception as e:
        logging.exception(f"Error processing document: {e}")
        return {"error": str(e), "processing_time_seconds": round(time.time() - start_time, 2)}
    
    finally:
        self._cleanup_temp_files(page_image_tuples, temp_results_file, source_path.lower().endswith('.pdf'))
        gc.collect()
    
  def _save_intermediate_results(self, results: List[Dict], output_file: str) -> None:
    try:
        with open(output_file, 'w') as f:
            json.dump({"pages": results}, f)
    except Exception as e:
        logging.warning(f"Failed to save intermediate results: {e}")
  
  def _cleanup_temp_files(self, image_tuples: List[tuple], results_file: str, delete_images: bool = True) -> None:
    if delete_images:
        for _, img_path in image_tuples:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except:
                pass
    
    try:
        if os.path.exists(results_file):
            os.remove(results_file)
    except:
        pass
  
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