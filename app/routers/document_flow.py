#!/usr/bin/env python3
"""
API endpoint for streamlined document field extraction flow:
1. Extract text from PDF
2. Pass text to model to identify fields and types
3. Return fields in required format
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import time
import re
from typing import Optional, List, Dict
from pydantic import BaseModel
from app.routers.document_processing import QwenDocumentIntegrator, normalize_field_name, convert_to_human_readable_label, BaseProcessor

class DocumentFlowRequest(BaseProcessor):
    """Request model for the document flow endpoint that extends BaseProcessor"""
    pass

router = APIRouter(
    prefix="/document-flow",
    tags=["document-flow"],
    responses={404: {"description": "Not found"}},
)

@router.post("/extract/")
async def extract_document_fields(request_body: DocumentFlowRequest):
    """
    Process a document using the streamlined 3-step flow:
    1. Extract text from PDF
    2. Pass text to model to identify fields and types
    3. Return fields in required format
    """    
    start_time = time.time()
    
    # Get the file from the source path
    source_path = request_body.source_path
    if not source_path:
        raise HTTPException(status_code=400, detail="Source path is required")
    
    if not source_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
   
    document_type = request_body.document_type or "form"
    client_id = request_body.client_id or "default"
    

    temp_file_path = source_path
    try:
        integrator = QwenDocumentIntegrator()
        document_text = integrator._extract_document_text(temp_file_path)
        
        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(status_code=422, detail="Failed to extract meaningful text from the document")
            
        # Preprocess document text to handle line breaks and fragmented words
        preprocessed_text = preprocess_document_text(document_text)
        print("Text preprocessing applied to fix fragmented field names")
        

        fields = []
        
        # First try model extraction
        try:
            if not integrator._model_loaded:
                integrator.load_model()
                
            if integrator._model_loaded:
                fields = integrator._extract_fields_with_ai(preprocessed_text, document_type)
        except Exception as e:
            print(f"AI extraction error: {e}")
              # Use  improved field extraction method
        improved_fields = extract_field_names_improved(document_text)
        if improved_fields:
            print(f"Improved extraction found {len(improved_fields)} fields")
            # Add to existing fields or use as base
            if not fields:
                # Use improved fields directly since they're already in document order
                fields = improved_fields
            else:
                combined_fields = []
                field_set = set(fields)
                
                # Add all improved fields first (in their document order)
                for field in improved_fields:
                    if field not in field_set:
                        combined_fields.append(field)
                    
                # Then add any remaining model fields that weren't in improved fields
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
                "field_descriptions": field_descriptions,                "document_schema": {
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
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # We don't delete the source file since it's provided by the user
        pass

def preprocess_document_text(document_text: str) -> str:
    """
    Preprocess document text to fix common extraction issues:
    1. Fix broken words across lines
    2. Normalize whitespace
    3. Improve form field detection
    4. Handle mixed case and special characters
    5. Join field names split across lines
    """
    if not document_text:
        return ""
    
    # Replace common hyphenation patterns where words are split across lines
    # Example: "secur-\nity" -> "security"
    processed = re.sub(r'([a-zA-Z])-\s*\n\s*([a-zA-Z])', r'\1\2', document_text)
    
    # Fix field names cut off by colons at the end of lines
    # Example: "Field Name:\nValue" -> "Field Name: Value"
    processed = re.sub(r'([a-zA-Z])\s*:\s*\n\s*', r'\1: ', processed)
    
    # Fix field names cut off by parentheses
    # Example: "Field Name (\nrequired)" -> "Field Name (required)"
    processed = re.sub(r'([a-zA-Z])\s*\(\s*\n\s*([a-zA-Z])', r'\1 (\2', processed)
    
    # Fix common word breaks without hyphens
    # Example: "Secur\nity" -> "Security"
    lines = processed.splitlines()
    result_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].rstrip()
        
        # Look ahead multiple lines to detect field name fragments
        merged_line = current_line
        lookahead = 0
        max_lookahead = 3  # Look up to 3 lines ahead
        
        # Check if this line might be part of a field label that continues across multiple lines
        potential_field_start = bool(re.search(r'[A-Z][a-z]+\s+[A-Z]?[a-z]*$', current_line))
        
        while (i + lookahead + 1 < len(lines) and 
               lookahead < max_lookahead and 
               potential_field_start):
               
            next_line = lines[i + lookahead + 1].strip()
            
            # If next line has field indicators or looks like a continuation of a field name
            if (re.match(r'^[a-z]+\s*[:\(\[]', next_line) or  # Starts with lowercase + field marker
                re.match(r'^[A-Z][a-z]+\s*[:\(\[]', next_line) or  # Starts with capitalized word + field marker
                re.match(r'^[a-z]+\s+[A-Z]', next_line)):  # Lowercase followed by capitalized (continued name)
                
                # Merge with space if it appears to be part of a multi-word field name
                merged_line += " " + next_line
                lookahead += 1
            else:
                break
        
        # If we merged lines, skip the ones we merged
        if lookahead > 0:
            result_lines.append(merged_line)
            i += lookahead + 1
        else:
            if (i + 1 < len(lines) and 
                len(current_line) > 0 and 
                len(lines[i+1].strip()) > 0):
                
                # More aggressive joining for field labels
                if re.search(r'[A-Za-z]\s*$', current_line):
                    next_line = lines[i+1].lstrip()
                    next_has_field_marker = any(marker in next_line[:30] for marker in [':', '_', '[', '(', '-'])
                    
                    if next_has_field_marker:
                        # Add space and combine with next line
                        result_lines.append(current_line + " " + next_line)
                        i += 2
                        continue
            
            result_lines.append(current_line)
            i += 1
    
    preprocessed_text = "\n".join(result_lines)
    

    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text)
    
    return preprocessed_text

def extract_field_names_improved(document_text: str) -> List[str]:
    """
    Improved field name extraction that handles fragmented field names
    and detects field labels across multiple lines.
    Addresses complex patterns and multiple line breaks.
    Preserves the original order of fields as they appear in the document.
    """

    field_positions = {}  # {field_name: position_in_document}
    
    # First, preprocess the text to fix common extraction issues
    processed_text = preprocess_document_text(document_text)
    

    field_patterns = [
        # General field name pattern with colon - more flexible
        r'([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50}):\s*[_\.\-\s]*(?:\n|$)', 
        
        # Checkbox patterns with improved word boundary detection
        r'(?:☐|□|\[\s*\]|☑|✓|✔)\s*([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50})\b',
        
        # Underline patterns with improved capture
        r'([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50})\s*[_]{2,}',
        
        # Numbered or bulleted field patterns
        r'(?:\d+\.|\*|\-|\•)\s*([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50})(?::|$)',
        
        # Form-style label patterns (common in PDF forms)
        r'(?:^|\n)([A-Z][A-Za-z0-9\s\-\'\/\.\,]{2,50})(?=\s*\(?(?:required|optional|MM\/DD\/YYYY|select one))',
        
        # Labeled input field patterns
        r'([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50})\s*(?:\[_+\]|\(_+\)|\|_+\||□|☐)',
    ]
    
    # Apply all patterns to the processed text
    for pattern in field_patterns:
        matches = re.finditer(pattern, processed_text, re.MULTILINE | re.DOTALL)
        for match in matches:
            field_name = match.group(1).strip()
            # Additional validation to filter out partial/incomplete matches
            if len(field_name.split()) > 1 or len(field_name) > 5:  # At least 2 words or longer single word
                normalized = normalize_field_name(field_name)
                if normalized and len(normalized) > 2:
                    # Store the field with its position if we haven't seen it before
                    # or if this occurrence is earlier in the document
                    position = match.start()
                    if normalized not in field_positions or position < field_positions[normalized]:
                        field_positions[normalized] = position
    
    
    text_blocks = re.split(r'(?:\n\s*\n|\n\s{3,}\n)', processed_text)  # Split by paragraph breaks
    
    for block in text_blocks:
        if len(block.strip()) < 5:  # Skip very short blocks
            continue
            
        # Extract field-like patterns within each text block
        potential_fields = re.findall(r'([A-Z][A-Za-z0-9\s\-\'\/\.\,]{2,30})[:\.\)\(]', block)
        for field in potential_fields:
            if len(field.strip().split()) >= 2:  # Require at least two words for better accuracy
                normalized = normalize_field_name(field.strip())
                if normalized and len(normalized) > 2:
                    # Calculate position in document
                    position = processed_text.find(field.strip())
                    if position < 0:  # If not found exactly, estimate
                        position = processed_text.find(block)
                    if normalized not in field_positions or position < field_positions[normalized]:
                        field_positions[normalized] = position
        
        lines = block.splitlines()
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i+1].strip()
            

            if not current_line or not next_line:
                continue
                
            # Check if lines might form a field name pattern when combined
            # Look for capital letters starting a line followed by lowercase, which might indicate a field name
            if (re.search(r'[A-Z][a-z]+\s*$', current_line) and 
                (next_line.startswith(tuple('abcdefghijklmnopqrstuvwxyz')) or 
                 any(marker in next_line[:15] for marker in [':', '(', '[', '_', '-']))):
                
                # Combine lines and check for field patterns
                combined = current_line + " " + next_line
                field_matches = re.findall(r'([A-Za-z][A-Za-z0-9\s\-\'\/\.\,]{2,50})(?::|$|\s*\(|\s*\[)', combined)
                for field in field_matches:
                    normalized = normalize_field_name(field.strip())
                    if normalized and len(normalized) > 2:
                        # Use the position of current line as the field's position
                        position = processed_text.find(current_line)
                        if position < 0:  # Fallback if not found
                            position = processed_text.find(block)
                        if normalized not in field_positions or position < field_positions[normalized]:
                            field_positions[normalized] = position
    
    
    form_field_blocks = re.findall(r'([A-Z][a-zA-Z\s]{2,40})\s*[:_]+[\s_]{3,}', processed_text)
    for field in form_field_blocks:
        normalized = normalize_field_name(field.strip())
        if normalized and len(normalized) > 2:
            position = processed_text.find(field.strip())
            if normalized not in field_positions or position < field_positions[normalized]:
                field_positions[normalized] = position
    
    header_patterns = [
        r'^\|\s*([A-Z][a-zA-Z\s]{2,30})\s*\|',  # Table headers with | delimiters
        r'^\+[-]+\+\s*([A-Z][a-zA-Z\s]{2,30})\s*\+',  # ASCII table headers
    ]
    
    for pattern in header_patterns:
        matches = re.finditer(pattern, processed_text, re.MULTILINE)
        for match in matches:
            field = match.group(1).strip()
            normalized = normalize_field_name(field)
            if normalized and len(normalized) > 2:
                position = match.start()
                if normalized not in field_positions or position < field_positions[normalized]:
                    field_positions[normalized] = position
    
    # Unduplicate and handle conflicting fields
    # Some fields might be detected with slight variations
    final_fields = list(field_positions.keys())
    
    # Filter out substrings of other fields to reduce redundancy
    # For example, if we have "incident date" and "incident date time", keep only the longer one
    filtered_fields_with_positions = {}
    for field in sorted(final_fields, key=len, reverse=True):  # Process longer fields first
        keep = True
        for existing_field in filtered_fields_with_positions.keys():
            # If field is a substring of an existing field and shorter (meaning it's likely a partial match)
            if field in existing_field and len(field) < len(existing_field):
                keep = False
                break
        if keep:
            filtered_fields_with_positions[field] = field_positions[field]
    
    # Sort fields by their position in the document to preserve original order
    return [field for field, _ in sorted(filtered_fields_with_positions.items(), key=lambda x: x[1])]
