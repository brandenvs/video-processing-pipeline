from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import tempfile
import requests
from urllib.parse import urlparse
import time
import logging
from typing import Optional, List, Dict, Any
import torch
from pydantic import BaseModel
import urllib
from app.routers.document_process_v2 import (
    BaseProcessor, 
    normalize_field_name,
    convert_to_human_readable_label
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
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded URL to: {output_path}")
    return output_path

router = APIRouter(
    prefix="/document-flow",
    tags=["document-flow"],
    responses={404: {"description": "Not found"}},
)

def detect_field_type(field_name: str) -> str:
    field_lower = field_name.lower()
    
    if any(date_word in field_lower for date_word in ["date", "time", "when", "dob", "birth"]):
        return "date" if "date" in field_lower else "time"
    
    elif any(num_word in field_lower for num_word in ["number", "id", "badge", "age", "count", "amount"]):
        return "number"
    
    elif any(text_word in field_lower for text_word in ["description", "narrative", "details", "notes", "summary"]):
        return "textarea"
    
    elif "email" in field_lower or "e_mail" in field_lower:
        return "email"
    
    elif any(phone_word in field_lower for phone_word in ["phone", "telephone", "contact", "mobile", "cell"]):
        return "phone"
    
    elif any(list_word in field_lower for list_word in ["type", "category", "status", "classification"]):
        return "select"
    
    return "text"

def generate_field_schema(fields: List[str]) -> Dict[str, Dict[str, Any]]:
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
    start_time = time.time()
    temp_files = []
    
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
    
    if not temp_file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_type = request_body.document_type or "form"
    client_id = request_body.client_id or "default"
    dpi = request_body.dpi or 150
    batch_size = request_body.batch_size or 3
    
    try:
        integrator = {}
        
        system_prompt = f"""
        Analyze this {document_type} document and:
        1. Thoroughly identify ALL form fields present in the document
        2. Extract values for each field if present
        3. Determine the appropriate field types (text, number, date, etc.)
        4. Return a structured JSON with complete field information
        
        IMPORTANT: Be extremely thorough and extract every possible field from the document.
        Include field names that might be represented by labels, boxes, or spaces for filling in information.
        """
        
        model_response = integrator.inference(
            system_prompt=system_prompt,
            max_token=1500,
            model_id="qwen/Qwen2.5-VL-3B-Instruct",
            source_path=temp_file_path,
            max_pages=None,
            batch_size=batch_size,
            dpi=dpi
        )
        
        fields_from_response = []
        
        if isinstance(model_response, dict):
            if "combined_content" in model_response and isinstance(model_response["combined_content"], dict):
                fields_from_response = list(model_response["combined_content"].keys())
            
            if not fields_from_response and "pages" in model_response and isinstance(model_response["pages"], list):
                field_set = set()
                for page in model_response["pages"]:
                    if isinstance(page, dict):
                        for key in page.keys():
                            if key not in ["page_number", "document_type", "raw_text", "processing_time_seconds"]:
                                field_set.add(normalize_field_name(key))
                
                fields_from_response = list(field_set)
                
            if not fields_from_response:
                for key in model_response.keys():
                    if key not in ["document_type", "pages", "combined_content", "tag", "raw_text", "processing_time_seconds", "total_pages_processed"]:
                        fields_from_response.append(key)
        
        all_fields = [normalize_field_name(field) for field in fields_from_response if field]
        
        unique_fields = []
        seen_fields = set()
        for field in all_fields:
            if field and field not in seen_fields:
                unique_fields.append(field)
                seen_fields.add(field)
        
        all_fields = unique_fields
        
        if not all_fields:
            fallback_prompt = f"""
            This is a critical task to identify ALL form fields in this {document_type} document.
            
            List EVERY field name or label that could be filled in or that contains information.
            
            Format your response as a JSON object with a "fields" key containing an array of field names.
            Example: {{"fields": ["first_name", "last_name", "date_of_birth", ...]}}
            """
            
            if "pages" in model_response and len(model_response["pages"]) > 0:
                first_page_num = model_response["pages"][0].get("page_number", 1)
                
                with torch.no_grad():
                    messages = [
                        {"role": "system", "content": "Extract all field names from this document."},
                        {"role": "user", "content": [
                            {"type": "text", "text": fallback_prompt}
                        ]}
                    ]
                    
                    inputs = integrator.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt"
                    ).to(integrator.device)
                    
                    outputs = integrator.model.generate(
                        inputs,
                        max_new_tokens=1000,
                        do_sample=False,
                        temperature=0.1
                    )
                    
                    response_text = integrator.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    try:
                        start_idx = response_text.find('[')
                        end_idx = response_text.rfind(']')
                        if start_idx >= 0 and end_idx > start_idx:
                            list_text = response_text[start_idx:end_idx+1]
                            import json
                            fields_list = json.loads(list_text.replace("'", '"'))
                            for field in fields_list:
                                norm_field = normalize_field_name(field)
                                if norm_field and norm_field not in seen_fields:
                                    all_fields.append(norm_field)
                                    seen_fields.add(norm_field)
                    except:
                        pass
        
        if not all_fields:
            fallback_fields = []
            if document_type == "form":
                fallback_fields = ["name", "date", "signature"]
            elif document_type == "invoice":
                fallback_fields = ["invoice_number", "date", "amount", "customer_name"]
            elif document_type == "report":
                fallback_fields = ["report_date", "title", "author", "summary"]
            else:
                fallback_fields = ["title", "date", "content"]
                
            for field in fallback_fields:
                if field not in seen_fields:
                    all_fields.append(field)
                    seen_fields.add(field)
        
        schema_fields = generate_field_schema(all_fields)
        
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
        if background_tasks and temp_files:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    background_tasks.add_task(lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)
