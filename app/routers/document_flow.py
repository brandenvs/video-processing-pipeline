from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import tempfile
import requests
from urllib.parse import urlparse
import time
import re
import logging
from typing import Optional, List, Dict, Any
import torch
from pydantic import BaseModel
import urllib
from app.routers.document_process_v2 import (
    QwenDocumentIntegrator, 
    BaseProcessor, 
    normalize_field_name,
    convert_to_human_readable_label,
    preprocess_document_text
)

class DocumentFlowRequest(BaseProcessor):
    pass

def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False
        
def download_file_from_url(url, output_path=None):
    if not output_path:
        file_name = os.path.basename(urlparse(url).path) or f"download_{uuid.uuid4()}.pdf"
        output_path = os.path.join(tempfile.gettempdir(), file_name)
    
    print(f"Downloading file from URL: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
#     with open(output_path, 'wb') as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
    
#     print(f"Downloaded URL to: {output_path}")
#     return output_path

# router = APIRouter(
#     prefix="/document-flow",
#     tags=["document-flow"],
#     responses={404: {"description": "Not found"}},
# )

def detect_field_type(field_name: str) -> str:
    """Detect field type based on field name patterns"""
    field_lower = field_name.lower()
    
    # Date/time fields
    if any(date_word in field_lower for date_word in ["date", "time", "when", "dob", "birth"]):
        return "date" if "date" in field_lower else "time"
    
    # Number fields
    elif any(num_word in field_lower for num_word in ["number", "id", "badge", "age", "count", "amount"]):
        return "number"
    
    # Textarea fields
    elif any(text_word in field_lower for text_word in ["description", "narrative", "details", "notes", "summary"]):
        return "textarea"
    
    # Email fields
    elif "email" in field_lower or "e_mail" in field_lower:
        return "email"
    
    # Phone fields
    elif any(phone_word in field_lower for phone_word in ["phone", "telephone", "contact", "mobile", "cell"]):
        return "phone"
    
    # Select fields
    elif any(list_word in field_lower for list_word in ["type", "category", "status", "classification"]):
        return "select"
    
    # Default
    return "text"

def generate_field_schema(fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """Generate a complete field schema with types, labels and descriptions"""
    schema_fields = {}
    
    for field_name in fields:
        field_type = detect_field_type(field_name)
        human_readable_label = convert_to_human_readable_label(field_name)
        description = f"Information related to {field_name.replace('_', ' ')}"
        
        schema_fields[field_name] = {
            "label": human_readable_label,
            "value": "",
            "field_type": field_type,
            "required": True,
            "description": description,
            "fieldDataVar": field_name
        }
    
    return schema_fields

@router.post("/extract/")
async def extract_document_fields(
    request_body: DocumentFlowRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Process a document to extract fields and generate a document schema.
    """    
    start_time = time.time()
    temp_files = []
    
    # Source validation and document retrieval
    source_path = request_body.source_path
    if not source_path:
        raise HTTPException(status_code=400, detail="Source path is required")
    
    if is_url(source_path):
        try:
            temp_file_path = download_file_from_url(source_path)
            temp_files.append(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download file from URL: {e}")
    else:
        if not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail=f"File not found: {source_path}")
        temp_file_path = source_path
    
#     if not temp_file_path.lower().endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     document_type = request_body.document_type or "form"
#     client_id = request_body.client_id or "default"
    
    try:
        # Create document integrator and load model
        integrator = QwenDocumentIntegrator()
        
        # Enhanced prompt that focuses on comprehensive field extraction
        system_prompt = f"""
        Analyze this {document_type} document and:
        1. Thoroughly identify ALL form fields present in the document
        2. Extract values for each field if present
        3. Determine the appropriate field types (text, number, date, etc.)
        4. Return a structured JSON with complete field information
        
        IMPORTANT: Be extremely thorough and extract every possible field from the document,
        even if it requires analyzing the document layout, formatting, or implied fields.
        Include field names that might be represented by labels, boxes, or spaces for filling in information.
        """
        
        # Load model if not already loaded
        if not integrator._model_loaded:
            integrator.load_model("qwen/Qwen2.5-VL-3B-Instruct")
        
        # Process document with model
        model_response = integrator.inference(
            system_prompt=system_prompt,
            max_token=1500,  # Increased token count for more thorough extraction
            model_id="qwen/Qwen2.5-VL-3B-Instruct",
            source_path=temp_file_path
        )
        
        # Extract document text for reference only
        document_text = integrator._extract_document_text(temp_file_path)
        preprocessed_text = preprocess_document_text(document_text)
        
        # Extract fields exclusively from model response
        fields_from_response = []
        
        # Extract fields from model response with enhanced field detection
        if isinstance(model_response, dict):
            # Try all possible field locations in model response
            if "fields" in model_response:
                if isinstance(model_response["fields"], list):
                    fields_from_response = model_response["fields"]
                elif isinstance(model_response["fields"], dict):
                    fields_from_response = list(model_response["fields"].keys())
            elif "form_fields" in model_response:
                if isinstance(model_response["form_fields"], list):
                    fields_from_response = model_response["form_fields"]
                elif isinstance(model_response["form_fields"], dict):
                    fields_from_response = list(model_response["form_fields"].keys())
            elif "extracted_fields" in model_response:
                if isinstance(model_response["extracted_fields"], list):
                    fields_from_response = model_response["extracted_fields"]
                elif isinstance(model_response["extracted_fields"], dict):
                    fields_from_response = list(model_response["extracted_fields"].keys())
            else:
                # Look for any key-value pairs in the response that could be fields
                for key, value in model_response.items():
                    if key not in ["document_type", "tag", "raw_text", "processing_time_seconds"]:
                        fields_from_response.append(key)
        
        # Process field names from response
        all_fields = [normalize_field_name(field) for field in fields_from_response if field]
        
        # Remove duplicates while preserving order
        unique_fields = []
        seen_fields = set()
        for field in all_fields:
            if field and field not in seen_fields:
                unique_fields.append(field)
                seen_fields.add(field)
        
        all_fields = unique_fields
        
        # If no fields were found, try a second call with an even more explicit prompt
        if not all_fields:
            fallback_prompt = f"""
            This is a critical task to identify ALL form fields in this {document_type} document.
            
            Please carefully examine the entire document and list EVERY SINGLE field name or label 
            that could be filled in or that contains information. Include:
            
            - Pre-filled fields with values
            - Empty fields waiting to be filled
            - Section headers and subheaders
            - Any label followed by a line, box, or space for information
            
            Format your response as a JSON object with a "fields" key containing an array of field names.
            Example: {{"fields": ["first_name", "last_name", "date_of_birth", ...]}}
            """
            
            fallback_response = integrator.inference(
                system_prompt=fallback_prompt,
                max_token=1500,
                model_id="qwen/Qwen2.5-VL-3B-Instruct",
                source_path=temp_file_path
            )
            
            # Extract fields from fallback response
            if isinstance(fallback_response, dict):
                if "fields" in fallback_response and isinstance(fallback_response["fields"], list):
                    fallback_fields = [normalize_field_name(field) for field in fallback_response["fields"] if field]
                    for field in fallback_fields:
                        if field and field not in seen_fields:
                            all_fields.append(field)
                            seen_fields.add(field)
        
        if not all_fields:
            raise HTTPException(status_code=422, detail="Failed to identify any fields in the document using the model")
        
        # Generate complete schema
        schema_fields = generate_field_schema(all_fields)
        
        # Prepare response
        response = {
            "status": "success",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "results": {
                "field_names": all_fields,
                "document_schema": {
                    "client_id": client_id,
                    "document_title": os.path.basename(source_path),
                    "document_type": document_type,
                    "is_active": True,
                    "schema_fields": schema_fields,
                    "created_by": "system"
                },
                "extraction_method": "model_only_extraction",
                "extracted_content": model_response
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logging.exception("Error processing document")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if background_tasks and temp_files:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    background_tasks.add_task(lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)
