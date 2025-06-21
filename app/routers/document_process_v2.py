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
    elif mm.should_use_fp16(self.device):
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
      attn_implementation="eager",
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

  def _identify_document_type(self, source_path: str) -> str:
    """Identify the type of document from its content"""
    if not self._model_loaded:
      print('>>> Loading model into memory for document type identification')
      self.load_model("qwen/Qwen2.5-VL-3B-Instruct")
      
    # Create messages asking specifically for document type
    type_detection_messages = [
      {
        "role": "system",
        "content": "You are an expert document classifier. Examine this document and identify its type.",
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": source_path},
          {"type": "text", "text": "What type of document is this? Respond with a single specific type like 'incident_report', 'patrol_report', 'security_form', 'invoice', etc."},
        ],
      },
    ]
    
    # Process messages to identify document type
    inputs = self.processor.apply_chat_template(
      type_detection_messages,
      add_generation_prompt=True,
      tokenize=True,
      return_tensors="pt"
    ).to(self.device)
    
    # Generate response for document type
    with torch.no_grad():
      outputs = self.model.generate(
        inputs,
        max_new_tokens=128,  # Shorter response for document type
        do_sample=False,
      )
    
    # Decode response
    type_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean and normalize the document type
    doc_type = type_response.lower().strip()
    
    # Extract just the document type, removing any explanation
    simple_type_match = re.search(r'([a-z_]+_?report|[a-z_]+_?form|[a-z_]+_?invoice|[a-z_]+_?document)', doc_type)
    if simple_type_match:
        return simple_type_match.group(1)
    
    # If no specific pattern found, take the first word or default to generic document
    words = re.findall(r'\b[a-z_]+\b', doc_type)
    if words:
        return words[0]
    else:
        return "generic_document"

  def inference(self, system_prompt: str, max_token: int, model_id: str, source_path: str) -> Dict[str, Any]:
    if not self._model_loaded:
      print('>>> Loading model into memory')
      self.load_model(model_id)
    
    # First, identify the document type
    document_type = self._identify_document_type(source_path)
    
    # Step 1: First analyze the document to determine structure
    analyze_messages = [
      {
        "role": "system",
        "content": "You are an expert document analyzer. Examine this document and provide a complete structured JSON output of all information present in the document. Include all fields and content visible in the document.",
      },
      {
        "role": "user",
        "content": [
          {"type": "image", "image": source_path},
          {"type": "text", "text": "Analyze this document and output a comprehensive JSON structure with all visible fields and their values. If a field has no value, include it with an empty string."},
        ],
      },
    ]
    
    # Process the analyze messages
    inputs = self.processor.apply_chat_template(
      analyze_messages,
      add_generation_prompt=True,
      tokenize=True,
      return_tensors="pt"
    ).to(self.device)
    
    # Generate response for document analysis
    with torch.no_grad():
      outputs = self.model.generate(
        inputs,
        max_new_tokens=max_token,
        do_sample=False,
      )
    
    # Decode and extract the structured response
    analysis_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract JSON from analysis
    extracted_data = self._extract_json_from_text(analysis_text)
    
    # Add the document type tag
    extracted_data["tag"] = document_type
    
    # Step 2: Process the extracted information with the user's system_prompt for customization
    if system_prompt and system_prompt != "Analyze this document and extract all key information as a structured JSON output":
      # Extract initial JSON from analysis
      initial_json = extracted_data
      
      # Use the system_prompt to further process or customize the output
      custom_messages = [
        {
          "role": "system",
          "content": f"You are an expert document processor. Use the following JSON data extracted from a document and refine it according to these instructions: {system_prompt}",
        },
        {
          "role": "user",
          "content": f"Here's the extracted document data: {json.dumps(initial_json)}. Process this data according to the instructions and return a refined JSON output. IMPORTANT: Make sure to preserve the 'tag' field that identifies the document type.",
        },
      ]
      
      inputs = self.processor.apply_chat_template(
        custom_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
      ).to(self.device)
      
      with torch.no_grad():
        outputs = self.model.generate(
          inputs,
          max_new_tokens=max_token,
          do_sample=False,
        )
      
      final_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
      result = self._extract_json_from_text(final_response)
      
      # Ensure the tag is preserved in the final result
      if "tag" not in result:
        result["tag"] = document_type
        
      return result
    else:
      return extracted_data


model_manager = Qwen2_VQA()