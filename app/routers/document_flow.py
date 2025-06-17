#!/usr/bin/env python3
"""
API endpoint for streamlined document field extraction flow:
1. Extract text from PDF
2. Pass text to model to identify fields and types
3. Return fields in required format
"""

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
from typing import Optional, List, Dict
from pydantic import BaseModel
from app.routers.document_processing import QwenDocumentIntegrator, normalize_field_name, convert_to_human_readable_label, BaseProcessor

class DocumentFlowRequest(BaseProcessor):
    """Request model for the document flow endpoint that extends BaseProcessor"""
    pass

def is_url(path):
    """Check if a path is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False
        
def download_file_from_url(url, output_path=None):
    """Download a file from URL to a local path"""
    if not output_path:
        # Create a temp file with the correct extension
        file_name = os.path.basename(urlparse(url).path) or f"download_{uuid.uuid4()}.pdf"
        output_path = os.path.join(tempfile.gettempdir(), file_name)
    
    print(f"Downloading file from URL: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
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

@router.post("/extract/")
async def extract_document_fields(
    request_body: DocumentFlowRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Process a document using the streamlined 3-step flow:
    1. Extract text from PDF
    2. Pass text to model to identify fields and types
    3. Return fields in required format
    
    Supports both local file paths and URLs as source
    """    
    start_time = time.time()
    temp_files = []  # Track temporary files for cleanup
    
    # Get the file from the source path
    source_path = request_body.source_path
    if not source_path:
        raise HTTPException(status_code=400, detail="Source path is required")
    
    if is_url(source_path):
        # If source_path is a URL, download the file
        try:
            print(f"Detected URL: {source_path}")
            temp_file_path = download_file_from_url(source_path)
            temp_files.append(temp_file_path)  # Track for cleanup
        except Exception as e:
            print(f"Failed to download file from URL: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download file from URL: {e}")
    else:
        # Check if local file exists
        if not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail=f"File not found: {source_path}")
        temp_file_path = source_path
    
    if not temp_file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_type = request_body.document_type or "form"
    client_id = request_body.client_id or "default"
    
    try:
        # Step 1: Extract text from the PDF
        integrator = QwenDocumentIntegrator()
        document_text = integrator._extract_document_text(temp_file_path)
        
        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(status_code=422, detail="Failed to extract meaningful text from the document")
            
        # Preprocess document text to handle line breaks and fragmented words
        preprocessed_text = preprocess_document_text(document_text)
        print("Text preprocessing applied to fix fragmented field names")
        
        # Step 2: Use AI to identify fields with their types
        fields = []
        
        # First try AI extraction
        try:
            if not integrator._model_loaded:
                integrator.load_model()
                
            if integrator._model_loaded:
                fields = integrator._extract_fields_with_ai(preprocessed_text, document_type)
        except Exception as e:
            print(f"AI extraction error: {e}")
            
        # Use our improved field extraction method
        improved_fields = extract_field_names_improved(document_text)
        if improved_fields:
            print(f"Improved extraction found {len(improved_fields)} fields")
            # Add to existing fields or use as base
            if not fields:
                # Use improved fields directly since they're already in document order
                fields = improved_fields
            else:
                # Since the improved fields are in document order, we want to preserve that order
                # Keep existing AI fields at their current positions but insert improved fields
                # where they naturally occur in the document
                combined_fields = []
                field_set = set(fields)
                
                # Add all improved fields first (in their document order)
                for field in improved_fields:
                    if field not in field_set:
                        combined_fields.append(field)
                    
                # Then add any remaining AI fields that weren't in improved fields
                for field in fields:
                    if field not in combined_fields:
                        combined_fields.append(field)
                
                fields = combined_fields
            
        # If still not enough fields, use standard extraction methods
        if not fields or len(fields) < 3:
            dynamic_fields = integrator._extract_additional_fields(preprocessed_text)
            
            if not fields:
                fields = dynamic_fields
            else:
                # Merge while preserving order
                field_set = set(fields)
                for field in dynamic_fields:
                    if field not in field_set:
                        fields.append(field)
        
        # If still not enough fields, use comprehensive extraction
        if not fields or len(fields) < 3:
            fields = integrator._extract_fields_comprehensive(
                source_path=None, 
                document_text=preprocessed_text,
                document_type=document_type
            )
        
        if not fields:
            raise HTTPException(status_code=422, detail="Failed to identify any fields in the document")
        
        # Step 3: Format fields for output
        # Determine field types
        field_types = {}
        field_descriptions = {}
        
        for field in fields:
            field_lower = field.lower()
            
            # Apply field type detection heuristics
            if any(date_word in field_lower for date_word in ["date", "time", "when", "dob", "birth"]):
                field_types[field] = "date" if "date" in field_lower else "time"
                
            elif any(num_word in field_lower for num_word in ["number", "id", "badge", "age", "count", "amount"]):
                field_types[field] = "number"
                
            elif any(text_word in field_lower for text_word in ["description", "narrative", "details", "notes", "summary"]):
                field_types[field] = "textarea"
                
            elif "email" in field_lower or "e_mail" in field_lower:
                field_types[field] = "email"
                
            elif any(phone_word in field_lower for phone_word in ["phone", "telephone", "contact", "mobile", "cell"]):
                field_types[field] = "phone"
                
            elif any(list_word in field_lower for list_word in ["type", "category", "status", "classification"]):
                field_types[field] = "select"
                
            else:
                field_types[field] = "text"
            
            # Generate descriptions
            field_descriptions[field] = f"Information related to {field.replace('_', ' ')}"
        
        # Build schema fields
        schema_fields = {}
        for field_name in fields:
            human_readable_label = convert_to_human_readable_label(field_name)
            schema_fields[field_name] = {
                "label": human_readable_label,
                "value": "",
                "field_type": field_types.get(field_name, "text"),
                "required": True,  # Default to required
                "description": field_descriptions.get(field_name, ""),
                "fieldDataVar": field_name
            }
        
        # Prepare the final response
        response = {
            "status": "success",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "results": {
                "field_names": fields,
                "field_types": field_types,
                "field_descriptions": field_descriptions,
                "document_schema": {
                    "client_id": client_id,
                    "document_title": os.path.basename(source_path),
                    "document_type": document_type,
                    "is_active": True,
                    "schema_fields": schema_fields,
                    "created_by": "system"
                },
                "extraction_method": "ai_dynamic_flow"
            }
        }
        
        # Return the response
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up any temporary files downloaded from URLs
        if background_tasks and temp_files:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    background_tasks.add_task(lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)
