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
import tempfile
import logging
from pypdf import PdfReader

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
  document_type: Optional[str] = "form"
  document_key: Optional[str] = None
  client_id: Optional[str] = None

# Utility functions needed for document_flow.py
def normalize_field_name(field_name: str) -> str:
    """Normalize field name to snake_case and clean it up"""
    if not field_name:
        return ""
    
    # Convert to lowercase and replace spaces/special chars with underscores
    normalized = re.sub(r'[^\w\s]', ' ', field_name.lower())
    normalized = re.sub(r'\s+', '_', normalized.strip())
    
    # Remove multiple underscores and trailing/leading underscores
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    
    return normalized

def convert_to_human_readable_label(field_name: str) -> str:
    """Convert normalized field name to human readable form"""
    if not field_name:
        return ""
    
    # Replace underscores with spaces
    readable = field_name.replace('_', ' ')
    
    # Capitalize first letter of each word
    readable = ' '.join(word.capitalize() for word in readable.split())
    
    return readable

def preprocess_document_text(text: str) -> str:
    """Preprocess extracted document text to fix common OCR issues"""
    if not text:
        return ""
    
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_field_names_improved(text: str) -> List[str]:
    """Extract field names using improved pattern matching"""
    fields = []
    
    # Look for form-like patterns (field: value)
    form_patterns = [
        r'([A-Z][A-Za-z\s]{2,25}):(?:\s*|$)',  # Field name followed by colon
        r'([A-Z][A-Za-z\s]{2,25})\s*\(\s*[X✓]\s*\)',  # Field with checkbox
        r'([A-Z][A-Za-z\s]{2,25})[_]{2,}',  # Field followed by underline
    ]
    
    for pattern in form_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            field = match.group(1).strip()
            if field and len(field) > 2 and field.lower() not in ['the', 'and', 'for', 'this', 'that']:
                fields.append(normalize_field_name(field))
    
    # Remove duplicates while preserving order
    unique_fields = []
    seen = set()
    for field in fields:
        if field not in seen:
            seen.add(field)
            unique_fields.append(field)
    
    return unique_fields

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
    
  # Document extraction functions for compatibility with document_flow.py
  def _extract_document_text(self, file_path: str) -> str:
    """Extract text from a PDF document"""
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
    """Extract field names from document text using AI model"""
    prompt = f"""
    Analyze this {document_type} document text and identify all form field names. 
    Return ONLY the normalized field names (snake_case) as a Python list.
    Example: ["first_name", "last_name", "date_of_birth"]
    
    Document text:
    {document_text[:4000]}  # Limit text length
    """
    
    if not self._model_loaded:
        self.load_model("qwen/Qwen2.5-VL-3B-Instruct")
        
    try:
        messages = [
            {"role": "system", "content": "You are a document field extraction assistant. Extract form fields from documents."},
            {"role": "user", "content": prompt}
        ]
        
        # Process the prompt
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response with optimized settings
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=500,
                do_sample=False,
                temperature=0.1
            )
        
        response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract list from response
        list_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if list_match:
            try:
                fields_list = json.loads(list_match.group(0).replace("'", '"'))
                if isinstance(fields_list, list):
                    return fields_list
            except:
                pass
                
        # Fallback: extract potential field names
        fields = []
        lines = response_text.strip().split('\n')
        for line in lines:
            # Look for field name patterns
            if '"' in line or "'" in line:
                field_match = re.search(r'["\']([a-z0-9_]+)["\']', line)
                if field_match:
                    fields.append(field_match.group(1))
            elif re.match(r'^[a-z0-9_]+$', line.strip()):
                fields.append(line.strip())
        
        return fields
        
    except Exception as e:
        logging.error(f"Error extracting fields with AI: {e}")
        return []
        
  def _extract_additional_fields(self, document_text: str) -> List[str]:
    """Extract additional fields using pattern matching"""
    return extract_field_names_improved(document_text)
    
  def _extract_fields_comprehensive(self, source_path: Optional[str], document_text: str, document_type: str) -> List[str]:
    """Comprehensive field extraction combining multiple methods"""
    fields = extract_field_names_improved(document_text)
    
    # If we have an image and not enough fields, use vision model
    if source_path and (not fields or len(fields) < 5) and os.path.exists(source_path):
        try:
            vision_prompt = f"Identify all form fields in this {document_type} document. List them in order of appearance."
            vision_fields = self._extract_fields_from_image(source_path, vision_prompt)
            
            # Combine fields
            existing_fields = set(fields)
            for field in vision_fields:
                if field not in existing_fields:
                    fields.append(field)
        except:
            pass
            
    # If still not enough fields, use generic fields based on document type
    if not fields or len(fields) < 3:
        if document_type == "form":
            fields.extend(["name", "date", "signature"])
        elif document_type == "invoice":
            fields.extend(["invoice_number", "date", "amount", "customer_name"])
        elif document_type == "report":
            fields.extend(["report_date", "title", "author", "summary"])
    
    # Remove duplicates
    unique_fields = []
    seen = set()
    for field in fields:
        norm_field = normalize_field_name(field)
        if norm_field and norm_field not in seen:
            seen.add(norm_field)
            unique_fields.append(norm_field)
    
    return unique_fields
    
  def _extract_fields_from_image(self, image_path: str, prompt: str) -> List[str]:
    """Extract fields directly from document image"""
    if not self._model_loaded:
        self.load_model("qwen/Qwen2.5-VL-3B-Instruct")
        
    try:
        messages = [
            {"role": "system", "content": "You are a document field extraction assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Process the image prompt
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=500,
                do_sample=False
            )
        
        response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract field names from the response
        fields = []
        
        # Look for field names in bullet points, numbered lists, or quotes
        patterns = [
            r'[\•\-\*]\s*["\']?([A-Za-z0-9\s]+)["\']?',  # Bullet points
            r'\d+\.\s*["\']?([A-Za-z0-9\s]+)["\']?',     # Numbered list
            r'["\']([A-Za-z0-9\s]+)["\']'                # Quoted fields
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response_text)
            for match in matches:
                field = match.group(1).strip()
                if field and len(field) > 2:
                    fields.append(normalize_field_name(field))
        
        # If no structured fields found, split by lines and extract potential field names
        if not fields:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 2 and len(line) < 30 and not line.startswith(('The', 'This', 'I ', 'Here')):
                    fields.append(normalize_field_name(line))
        
        return fields
    
    except Exception as e:
        logging.error(f"Error extracting fields from image: {e}")
        return []

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


# Create a class that's compatible with QwenDocumentIntegrator for document_flow.py
class QwenDocumentIntegrator(Qwen2_VQA):
    """Compatibility class for document_flow.py that extends Qwen2_VQA"""
    pass


model_manager = Qwen2_VQA()