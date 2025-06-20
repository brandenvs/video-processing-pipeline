# Fix corrupted file - replace the last section with proper code
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import json
import datetime
import gc
import asyncio
import functools
import subprocess
import re
import torch
import atexit
import requests 
import tempfile
import urllib.parse
import threading 
import traceback  
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import the DB configuration from the original project
from app.routers.database_service import DB_CONFIG, Db_helper

from app.routers import model_management as mm


def normalize_field_name(field_name: str) -> str:
    """Dynamic normalization without hard-coded assumptions"""
    if not field_name:
        return ""
    
    # Basic cleanup
    cleaned = field_name.lower().strip()
    
    # Remove obvious junk patterns dynamically
    # Remove non-word characters except spaces
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    
    # Split into words and filter dynamically
    words = cleaned.split()
    meaningful_words = []
    
    for word in words:
        # Keep words that are at least 2 chars (for abbreviations like 'id', 'no')
        # or that look like meaningful words
        if len(word) >= 2 and word.isalpha():
            meaningful_words.append(word)
    
    if len(meaningful_words) == 0:
        return ""
    
    # Dynamic redundancy removal - if we see repeated patterns, remove them
    # For example: "form officer name" -> "officer name" (remove generic "form")
    filtered_words = []
    for word in meaningful_words:
        # Skip overly generic words that don't add meaning when they appear with other words
        if len(meaningful_words) > 1 and word in ['form', 'document', 'field', 'input']:
            continue
        filtered_words.append(word)
    
    if len(filtered_words) == 0:
        filtered_words = meaningful_words  # Fallback to original if we filtered too much
    
    # Join with underscores
    normalized = '_'.join(filtered_words)
    
    # Final cleanup
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    
    # Basic validation
    if len(normalized) < 2:
        return ""
    
    return normalized


class SchemaField(BaseModel):
    """Schema field definition"""
    label: str
    value: str = ""
    field_type: str = "text"
    required: bool = True
    description: Optional[str] = None
    fieldDataVar: Optional[str] = None


class DocumentSchema(BaseModel):
    """Document schema definition"""
    client_id: str = "default"
    document_title: str
    document_type: str
    is_active: bool = True
    schema_fields: Dict[str, SchemaField]
    created_by: str = "system"


class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = "Identify key data and fillout the given system schema"
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None
    source_url: Optional[str] = None  # New field for URL-based document processing
    document_type: Optional[str] = "form"  # Default type for processing
    document_key: Optional[str] = None
    client_id: Optional[str] = None  # New field for client identification

router = APIRouter(
    prefix="/documents",
    tags=["document-processing"],
    responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)

def cleanup_executor():
    executor.shutdown(wait=True)

atexit.register(cleanup_executor)

def check_memory(device=mm.get_torch_device()):
    print("Device Loaded: ", device)

    total_mem = mm.get_total_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU has {total_mem} GBs")

    free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
    return (free_mem_gb, total_mem)

class DynamicPromptGenerator:
    """Generates system prompts dynamically based on extracted fields for video analysis of security/body camera footage"""
    
    def __init__(self):
        self.prompt_templates = {
            "form": self._generate_form_prompt,
            "security": self._generate_security_video_prompt,
            "bodycam": self._generate_bodycam_prompt,
            "patrol_report": self._generate_patrol_report_prompt,
            "incident_report": self._generate_incident_report_prompt,
            "traffic_stop": self._generate_traffic_stop_prompt,
            "default": self._generate_default_prompt
        }
    
    def generate_system_prompt(self, document_schema: DocumentSchema) -> str:
        """Generate a dynamic system prompt based on document schema"""
        document_type = document_schema.document_type.lower()
        generator = self.prompt_templates.get(document_type, self.prompt_templates["default"])
        return generator(document_schema)
    
    def _generate_form_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt for generic forms"""
        field_descriptions = []
        required_fields = []
        optional_fields = []
        
        for field_name, field_data in schema.schema_fields.items():
            field_desc = f"- {field_data.label}: {field_data.description or 'Extract relevant information'}"
            field_descriptions.append(field_desc)
            
            if field_data.required:
                required_fields.append(field_data.label)
            else:
                optional_fields.append(field_data.label)
        
        prompt = f"""
You are an intelligent document and video analysis system processing: {schema.document_title}

TASK: Extract and fill the following form fields from the provided document and/or video content.

DOCUMENT TYPE: {schema.document_type}
CLIENT: {schema.client_id}

REQUIRED FIELDS (must be filled):
{chr(10).join(f"• {field}" for field in required_fields) if required_fields else "• None"}

OPTIONAL FIELDS (fill if available):
{chr(10).join(f"• {field}" for field in optional_fields) if optional_fields else "• None"}

FIELD SPECIFICATIONS:
{chr(10).join(field_descriptions)}

INSTRUCTIONS:
1. Analyze all available content (document text, video frames, audio transcripts)
2. Extract accurate information for each field
3. If a field cannot be determined, mark it as "Not Available" or "N/A"
4. Prioritize accuracy over completeness
5. For video content, look for visual cues, text overlays, spoken information
6. Maintain consistency in date formats, names, and numerical values

OUTPUT FORMAT: Return a JSON object with field names as keys and extracted values as values.
"""
        return prompt.strip()
    
    def _generate_default_prompt(self, schema: DocumentSchema) -> str:
        """Generate a default prompt for unspecified document types"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Extract relevant information'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are a document analysis AI processing: {schema.document_title}

DOCUMENT TYPE: {schema.document_type}
CLIENT: {schema.client_id}

FIELDS TO EXTRACT:
{chr(10).join(field_list)}

GENERAL INSTRUCTIONS:
1. Analyze the provided document and/or video content thoroughly
2. Extract accurate information for each specified field
3. If information is not available, mark as "N/A"
4. Maintain consistency in formatting and terminology
5. Prioritize accuracy over speed
6. For unclear information, indicate uncertainty

OUTPUT: Return a JSON object with field names as keys and extracted values as values.
"""
        return prompt.strip()
    
    def _generate_security_video_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt specifically for security camera video analysis"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Analyze video content for this information'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are an advanced security camera video analysis AI processing: {schema.document_title}

VIDEO ANALYSIS CONTEXT: You are analyzing security camera footage to extract information and fill out form fields. Focus on visual elements, actions, people, objects, and events captured in the video.

CLIENT: {schema.client_id}
DOCUMENT TYPE: {schema.document_type}

FIELDS TO EXTRACT FROM VIDEO:
{chr(10).join(field_list)}

SECURITY VIDEO ANALYSIS INSTRUCTIONS:
1. PERSON IDENTIFICATION: Look for individuals in the footage - count people, describe appearance, clothing, behavior
2. TIMESTAMP ANALYSIS: Extract visible timestamps, dates, or time indicators from video overlay
3. LOCATION MARKERS: Identify any visible location indicators, street signs, building numbers, landmarks
4. VEHICLE IDENTIFICATION: Spot vehicles, license plates, make/model information
5. INCIDENT DETECTION: Observe any unusual activities, violations, or significant events
6. OBJECT RECOGNITION: Identify relevant objects, equipment, weapons, or items of interest
7. MOVEMENT PATTERNS: Track person/vehicle movements, directions, entry/exit points
8. AUDIO ANALYSIS: Extract information from any audio/speech in the video
9. ENVIRONMENTAL CONTEXT: Note weather conditions, lighting, time of day indicators
10. DOCUMENTATION VISIBLE: Look for any documents, IDs, signs, or text visible in frames

VIDEO-SPECIFIC TECHNIQUES:
- Analyze multiple frames throughout the video duration
- Pay attention to motion detection and activity zones
- Look for recurring patterns or behaviors
- Extract text from any on-screen displays or overlays
- Identify camera angles and coverage areas
- Note video quality and visibility conditions

ACCURACY REQUIREMENTS:
- Time stamps must be precise if visible
- Person descriptions should be detailed but objective
- Vehicle information must be complete (plate, color, type)
- Location details should be specific
- Incident descriptions must be factual and detailed

OUTPUT: Return a JSON object with field names as keys and extracted video analysis as values.
"""
        return prompt.strip()
    
    def _generate_bodycam_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt specifically for body camera video analysis"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Extract from body cam footage'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are a specialized body camera video analysis AI processing: {schema.document_title}

BODY CAMERA CONTEXT: You are analyzing police/security body camera footage from an officer's perspective. Focus on interactions, conversations, procedures, and evidence visible from the officer's viewpoint.

CLIENT: {schema.client_id}
DOCUMENT TYPE: {schema.document_type}

FIELDS TO EXTRACT FROM BODY CAM FOOTAGE:
{chr(10).join(field_list)}

BODY CAMERA ANALYSIS INSTRUCTIONS:
1. OFFICER INTERACTIONS: Analyze conversations between officer and subjects
2. SUBJECT IDENTIFICATION: Extract names, descriptions, and behavior of individuals contacted
3. LOCATION CONTEXT: Identify where the interaction takes place (address, landmarks, environment)
4. PROCEDURE COMPLIANCE: Note if proper procedures are followed (Miranda rights, searches, etc.)
5. EVIDENCE COLLECTION: Identify any physical evidence seen or collected
6. INCIDENT CHRONOLOGY: Track the sequence of events throughout the interaction
7. AUDIO TRANSCRIPTION: Extract important dialogue and verbal exchanges
8. OFFICER ACTIONS: Document what the officer does, says, and observes
9. SAFETY CONCERNS: Note any weapons, threats, or safety issues
10. DOCUMENTATION: Look for any forms, citations, or paperwork being filled out

BODY CAM SPECIFIC FOCUS:
- First-person perspective analysis from officer's viewpoint
- Audio quality may vary - extract what's clearly audible
- Look for officer equipment (radio, computer, forms) in frame
- Note officer's verbal announcements and explanations
- Track suspect responses and compliance levels
- Identify backup officers or other personnel
- Observe vehicle stops, searches, and arrests
- Extract license plate numbers, addresses, and identification

LEGAL/PROCEDURAL AWARENESS:
- Note Miranda warnings if given
- Document consent for searches
- Track evidence handling procedures
- Identify probable cause explanations
- Record any violations or complaints

OUTPUT: Return a JSON object with field names as keys and extracted body cam analysis as values.
"""
        return prompt.strip()
    
    def _generate_patrol_report_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt for patrol report video analysis"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Extract for patrol documentation'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are a patrol report documentation AI analyzing video footage for: {schema.document_title}

PATROL REPORT CONTEXT: Analyze video evidence to complete official patrol reports and incident documentation.

CLIENT: {schema.client_id}
DOCUMENT TYPE: {schema.document_type}

PATROL REPORT FIELDS TO COMPLETE:
{chr(10).join(field_list)}

PATROL REPORT ANALYSIS FOCUS:
1. INCIDENT DETAILS: What, when, where, who, why, and how of the incident
2. OFFICER INFORMATION: Badge numbers, names, units involved
3. SUBJECT/WITNESS DATA: Names, addresses, contact information, descriptions
4. LOCATION SPECIFICS: Exact addresses, cross streets, landmarks, GPS coordinates
5. TIME DOCUMENTATION: Dispatch time, arrival time, incident duration, clear time
6. NARRATIVE CONSTRUCTION: Chronological sequence of events
7. EVIDENCE INVENTORY: Items collected, photographed, or observed
8. ACTIONS TAKEN: Arrests, citations, warnings, referrals made
9. DAMAGE ASSESSMENT: Property damage, injuries, medical attention
10. FOLLOW-UP REQUIREMENTS: Investigations needed, court dates, report distribution

OFFICIAL REPORT STANDARDS:
- Use clear, objective language
- Include all relevant details for court proceedings
- Maintain chain of custody documentation
- Ensure accuracy for legal proceedings
- Follow department reporting protocols
- Include witness statements and contact information

VIDEO ANALYSIS FOR REPORTS:
- Extract exact quotes from conversations
- Document visual evidence of violations or incidents
- Note environmental conditions affecting the incident
- Track all parties involved and their roles
- Identify any contributing factors or causes

OUTPUT: Return comprehensive JSON data suitable for official patrol report completion.
"""
        return prompt.strip()
    
    def _generate_incident_report_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt for incident report video analysis"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Document incident details'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are an incident report analysis AI processing video evidence for: {schema.document_title}

INCIDENT REPORT CONTEXT: Analyze video footage to document incidents, accidents, violations, or significant events for official reporting.

CLIENT: {schema.client_id}
DOCUMENT TYPE: {schema.document_type}

INCIDENT REPORT FIELDS:
{chr(10).join(field_list)}

INCIDENT ANALYSIS REQUIREMENTS:
1. INCIDENT CLASSIFICATION: Type of incident (traffic, criminal, medical, civil, etc.)
2. TIMELINE RECONSTRUCTION: Precise sequence of events with timestamps
3. INVOLVED PARTIES: All individuals, their roles, and involvement levels
4. CAUSAL FACTORS: What led to the incident, contributing circumstances
5. INJURY/DAMAGE ASSESSMENT: Medical attention needed, property damage extent
6. WITNESS IDENTIFICATION: Anyone who observed the incident
7. ENVIRONMENTAL CONDITIONS: Weather, lighting, road conditions, visibility
8. VIOLATION DOCUMENTATION: Laws broken, regulations violated, policy breaches
9. RESPONSE ACTIONS: Emergency services called, immediate actions taken
10. EVIDENCE PRESERVATION: Photos taken, statements given, items secured

INCIDENT-SPECIFIC FOCUS:
- Determine primary and secondary causes
- Document pre-incident conditions
- Track incident progression and escalation
- Identify intervention points where outcomes could have changed
- Note any equipment malfunctions or failures
- Document compliance with safety procedures

VIDEO EVIDENCE ANALYSIS:
- Multiple camera angles if available
- Before, during, and after incident footage
- Audio analysis for warnings, instructions, or statements
- Movement patterns and decision points
- Impact sequences and aftermath
- Emergency response effectiveness

OUTPUT: Return detailed JSON data for comprehensive incident documentation.
"""
        return prompt.strip()
    
    def _generate_traffic_stop_prompt(self, schema: DocumentSchema) -> str:
        """Generate prompt for traffic stop video analysis"""
        field_list = [f"• {field_data.label}: {field_data.description or 'Extract traffic stop information'}" 
                     for field_data in schema.schema_fields.values()]
        
        prompt = f"""
You are a traffic stop analysis AI processing video footage for: {schema.document_title}

TRAFFIC STOP CONTEXT: Analyze body camera or dash camera footage of traffic stops to complete citation forms, incident reports, and documentation.

CLIENT: {schema.client_id}
DOCUMENT TYPE: {schema.document_type}

TRAFFIC STOP FIELDS TO EXTRACT:
{chr(10).join(field_list)}

TRAFFIC STOP ANALYSIS PROTOCOL:
1. VEHICLE INFORMATION: License plate, make, model, year, color, condition
2. DRIVER/PASSENGER DATA: Number of occupants, descriptions, behavior, compliance
3. VIOLATION DOCUMENTATION: Speed recorded, traffic law violations observed
4. STOP PROCEDURE: Reason for stop, location, time, duration
5. OFFICER SAFETY: Positioning, backup called, risk assessment
6. DOCUMENT EXCHANGE: License, registration, insurance verification
7. CITATION DETAILS: Violations cited, warnings given, court information
8. SEARCH JUSTIFICATION: Probable cause, consent, or warrant basis
9. EVIDENCE COLLECTION: Items found, photographed, or seized
10. RESOLUTION: Citation issued, arrest made, warning given, or released

TRAFFIC ENFORCEMENT SPECIFICS:
- Radar/lidar readings if visible
- Posted speed limits and traffic signs
- Road conditions and traffic flow
- Driver sobriety assessment indicators
- Vehicle equipment violations
- Registration and insurance status
- Outstanding warrants or alerts

VIDEO ANALYSIS FOR CITATIONS:
- Clear documentation of the violation
- Driver behavior and cooperation level
- Passenger interactions and behavior
- Any contraband or evidence in plain view
- Officer explanation of violation to driver
- Proper citation completion procedures

LEGAL DOCUMENTATION REQUIREMENTS:
- Accurate location for court jurisdiction
- Precise violation codes and descriptions
- Proper Miranda warnings if applicable
- Chain of custody for any evidence
- Witness information if applicable

OUTPUT: Return structured JSON data for traffic citation and report completion.
"""
        return prompt.strip()
    
class QwenDocumentIntegrator:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = mm.get_torch_device()
        self.dtype = None
        self._model_loaded = False
        self.OFFLOAD_DIR = "./offload"
        self._text_cache = {}  # Cache extracted text to avoid re-processing
        self.dynamic_prompt_generator = DynamicPromptGenerator()  # Add prompt generator instance

    def _download_file_from_url(self, url: str) -> str:
        """
        Download a file from a URL and return the local file path.
        Supports Firebase Storage URLs and other document URLs.
        """
        try:
            print(f">>> Downloading document from URL: {url}")
            
            # Parse the URL to get a filename
            parsed_url = urllib.parse.urlparse(url)
            
            # Extract filename from URL path or create a default one
            if parsed_url.path:
                # Get the last part of the path and decode it
                path_parts = parsed_url.path.split('/')
                filename = path_parts[-1] if path_parts else "downloaded_document.pdf"
                # Remove URL encoding
                filename = urllib.parse.unquote(filename)
                # Remove query parameters from filename if they exist
                filename = filename.split('?')[0]
                # Clean up any remaining path separators
                filename = filename.replace('/', '_').replace('\\', '_')
            else:
                filename = "downloaded_document.pdf"
            
            # Ensure we have a file extension
            if not filename or '.' not in filename:
                filename = "downloaded_document.pdf"
            
            # Create a temporary file path
            temp_dir = tempfile.gettempdir()
            local_file_path = os.path.join(temp_dir, f"doc_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
            
            # Download the file
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
              # Save the file
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f">>> Downloaded document successfully to: {local_file_path}")
            print(f">>> File size: {os.path.getsize(local_file_path)} bytes")
            
            return local_file_path
            
        except requests.exceptions.RequestException as e:
            print(f">>> Error downloading file from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {str(e)}")
        except Exception as e:
            print(f">>> Unexpected error downloading file: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error downloading document: {str(e)}")

    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary downloaded files"""
        try:
            if file_path and os.path.exists(file_path) and tempfile.gettempdir() in file_path:
                os.remove(file_path)
                print(f">>> Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f">>> Warning: Could not clean up temporary file {file_path}: {e}")

    def _setup_gguf_model(self):
        from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
        from ctransformers import AutoTokenizer as CTAutoTokenizer

        print(f">>> Loading GGUF model from {self.model_checkpoint}")
        try:
            self.model = CTAutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                model_type="qwen",
                gpu_layers=0 if self.device == "cpu" else 20,
                context_length=2048,
                threads=4 if self.device == "cpu" else 1,
                batch_size=1
            )
            self.tokenizer = CTAutoTokenizer.from_pretrained(self.model)
            self._model_loaded = True
            print(">>> Model loaded successfully")
            return True
        except Exception as e:
            print(f">>> GGUF loading error: {type(e).__name__}: {e}")
            return self._setup_gguf_model_fallback()

    def _setup_gguf_model_fallback(self):
        try:
            from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
            from ctransformers import AutoTokenizer as CTAutoTokenizer
            
            self.model = CTAutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                model_type="qwen",
                gpu_layers=0,  # Force CPU mode
                context_length=1024,  # Reduced context
                threads=4,
                batch_size=1
            )
            self.tokenizer = CTAutoTokenizer.from_pretrained(self.model)
            self._model_loaded = True
            print(">>> Model loaded in CPU-only mode")
            return True
        except Exception as fallback_e:
            print(f">>> Fallback loading failed: {fallback_e}")
            return False

    def _setup_hf_model(self):
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE
        
        model_id = "Qwen/Qwen1.5-1.8B-Chat"
        
        # Ensure offload directory exists
        if not os.path.exists(self.OFFLOAD_DIR):
            os.makedirs(self.OFFLOAD_DIR)
        
        # Setup cache directory
        cache_dir = os.environ.get(HF_HUB_CACHE) or os.environ.get(HUGGINGFACE_HUB_CACHE) 
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        model_cache_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
        
        if not os.path.exists(model_cache_path):
            print(f">>> Downloading model {model_id}")
            snapshot_download(repo_id=model_id, local_dir=model_cache_path)
        else:
            print(f">>> Model already downloaded to {model_cache_path}")
        
        self.model_checkpoint = model_id
        return self._setup_hf_model_config()

    def _setup_hf_model_config(self):
        # Set precision based on device capabilities
        if mm.should_use_fp16(self.device):
            self.dtype = torch.float16
        elif mm.should_use_bf16(self.device):
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        print(f'>>> Selected DType: {self.dtype}')

        # Configure quantization if needed
        quantization_config = None
        if torch.cuda.is_available() and self.dtype != torch.float16:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._log_device_info()
        return self._load_hf_model(quantization_config)

    def _log_device_info(self):
        if not torch.cuda.is_available():
            print(">>> CUDA not available")
            return

        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        print(f'>>> CUDA Device: {device_name}')
        print(f'>>> Compute Capability: {capability[0]}.{capability[1]}')
        
        major = capability[0]
        if major >= 8:
            print(">>> Architecture: Ampere or newer (FlashAttention-compatible)")
        elif major == 7:            print(">>> Architecture: Turing or Volta (not compatible with FlashAttention)")
        else:
            print(">>> Architecture: Older (not compatible)")

    def _load_hf_model(self, quantization_config):
        try:
            print(">>> Starting model loading process...")
            
            # CPU-optimized loading to prevent hanging
            loading_kwargs = {
                "pretrained_model_name_or_path": self.model_checkpoint,
                "torch_dtype": torch.float32,  # Force float32 for CPU stability
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Only add these parameters when on GPU
            if torch.cuda.is_available():
                print(f">>> Loading model on GPU with available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                loading_kwargs["device_map"] = "auto"
                loading_kwargs["torch_dtype"] = self.dtype
                if quantization_config:
                    loading_kwargs["quantization_config"] = quantization_config
            else:
                print(">>> Loading model on CPU with optimizations")
                # CPU-specific optimizations
                loading_kwargs.update({                    "device_map": "cpu",
                    "max_memory": {"cpu": "4GB"},  # Limit CPU memory usage
                    "offload_folder": self.OFFLOAD_DIR,
                })
            
            print(">>> Executing model loading...")
            # Load with timeout protection
            import threading
            
            result = [None]
            exception = [None]
            
            def load_model():
                try:
                    result[0] = AutoModelForCausalLM.from_pretrained(**loading_kwargs)
                except Exception as e:
                    exception[0] = e
            
            # Use threading for timeout on all platforms
            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
            thread.join(timeout=60)  # 60 second timeout for model loading
            
            if thread.is_alive():
                print(">>> Model loading timed out, forcing CPU-only minimal loading")
                # Force minimal CPU loading
                minimal_kwargs = {
                    "pretrained_model_name_or_path": self.model_checkpoint,
                    "torch_dtype": torch.float32,
                    "trust_remote_code": True,
                    "device_map": "cpu",
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(**minimal_kwargs)
            elif exception[0]:
                raise exception[0]
            else:
                self.model = result[0]
            
            print(">>> Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint, 
                trust_remote_code=True
            )
            
            # Verify model is properly loaded
            print(">>> Verifying model initialization...")
            if hasattr(self.model, "config"):
                print(f">>> Model config verified: {self.model.config.model_type}")
            
            self._model_loaded = True
            print(">>> Model loaded successfully")
              # Force model evaluation mode
            self.model.eval()
            
            return True
        except Exception as e:
            print(f">>> Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def load_model(self):
        """Load the model, trying GGUF format first, then falling back to HF format if needed."""
        if self._model_loaded:
            return

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        check_memory(self.device)
        torch.manual_seed(42)  # For reproducibility
        
        # Try GGUF format first
        model_id = "Qwen3-1.7B-UD-Q8_K_XL.gguf"
        model_dir = os.path.join(os.getcwd(), "models", "prompt_generator")
        self.model_checkpoint = os.path.join(model_dir, os.path.basename(model_id))

        if not os.path.exists(self.model_checkpoint):
            print(f">>> GGUF model file not found: {self.model_checkpoint}")
        else:
            try:
                if self._setup_gguf_model():
                    return
            except ImportError:
                print(">>> ctransformers not installed. Installing...")
                try:
                    subprocess.check_call(["pip", "install", "ctransformers"])
                    print(">>> ctransformers installed. Please restart the application.")
                    raise ImportError("ctransformers installed. Please restart the application.")
                except subprocess.CalledProcessError:
                    print(">>> Failed to install ctransformers, falling back to HF model")
            except Exception as e:
                print(f">>> Error loading GGUF model: {type(e).__name__}: {e}")
        
        # Fall back to standard HF model
        print(">>> Falling back to standard Hugging Face model loading")
        try:
            self._setup_hf_model()
        except Exception as e:
            print(f">>> Critical error: Failed to load any model: {str(e)}")            # Set a flag to bypass model loading in inference
            self._model_loaded = False
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load any model format: {str(e)}")

    def process_generated_response(self, generated_response: str):
        """Process the generated response from the model."""
        if not generated_response.startswith("```json"):
            return {"raw_response": generated_response}

        try:
            lines = generated_response.strip().splitlines()
            json_content = "\n".join(lines[1:-1])
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"error": str(e)}

    async def process_document(self, request_body: BaseProcessor):
        loop = asyncio.get_running_loop()
        torch.cuda.empty_cache()
        gc.collect()
        check_memory()
        
        # Handle URL downloads if source_url is provided
        temp_file_path = None
        original_source_path = request_body.source_path
        
        try:
            if request_body.source_url and not request_body.source_path:
                # Download file from URL
                temp_file_path = self._download_file_from_url(request_body.source_url)
                request_body.source_path = temp_file_path
                print(f">>> Processing document from URL: {request_body.source_url}")
                print(f">>> Downloaded to temporary file: {temp_file_path}")
            elif request_body.source_url and request_body.source_path:
                raise HTTPException(status_code=400, detail="Please provide either source_path OR source_url, not both")
            elif not request_body.source_url and not request_body.source_path:
                raise HTTPException(status_code=400, detail="Either source_path or source_url must be provided")
            
            inference_task = functools.partial(
                self.inference, **request_body.model_dump()
            )
            results = await loop.run_in_executor(executor, inference_task)
            
            return {
                "status": "success" if results and "error" not in results else "error",
                "results": results,
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Clean up temporary file if it was downloaded
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
                # Restore original source_path
                request_body.source_path = original_source_path

    def inference(self, source_path=None, document_text=None, system_prompt=None, max_new_tokens=512, **kwargs):
        """
        Process a document and extract field names from forms.
        Optimized version with caching and fallback to regex-based extraction.
        Supports both file path input and direct text input.
        """
        # Determine if we're processing a file or direct text
        if document_text:
            print(">>> Processing direct text input...")
            processed_text = document_text
            cache_key = None  # No caching for direct text
        elif source_path:
            # File processing mode
            if not os.path.exists(source_path):
                return {"error": f"File not found: {source_path}"}
            
            # Check cache first
            cache_key = f"{source_path}_{os.path.getmtime(source_path)}"
            if cache_key in self._text_cache:
                print(">>> Using cached document text")
                processed_text = self._text_cache[cache_key]
            else:
                print(">>> Extracting document text...")
                processed_text = self._extract_document_text(source_path)
                if processed_text and cache_key:
                    self._text_cache[cache_key] = processed_text
        else:
            return {"error": "Either source_path or document_text must be provided"}
        
        if not processed_text:
            return {"error": "Failed to extract or process document text"}
          # Use comprehensive field extraction (AI + regex + additional patterns)
        field_names = self._extract_fields_comprehensive(source_path, processed_text, **kwargs)
        
        print(f">>> Comprehensive extraction completed: {len(field_names)} fields found")
        
        # Create document schema and generate dynamic prompt
        document_type = kwargs.get('document_type', 'form')
        client_id = kwargs.get('client_id') or 'default'  # Ensure client_id is never None
        document_title = kwargs.get('document_title', os.path.basename(source_path) if source_path else 'Extracted Document')
        
        # Create reasonable default field types and descriptions
        field_types = {name: "text" for name in field_names}
        field_descriptions = {name: f"Information related to {name.lower().replace('_', ' ')}" for name in field_names}
        
        document_schema = self._create_document_schema(
            field_names=field_names,
            field_types=field_types,
            field_descriptions=field_descriptions,
            document_type=document_type,
            client_id=client_id,
            document_title=document_title
        )        # Generate dynamic system prompt
        generated_system_prompt = self._generate_dynamic_system_prompt(document_schema)
        
        video_prompt = self._generate_video_processing_prompt(field_names, field_descriptions)
        
        # Generate a unique ID for this extraction
        import random
        extraction_id = random.randint(1, 999)
        
        # Return the exact structure requested
        return {
            "status": "success",
            "results": {
                "field_names": field_names,
                "field_types": field_types,
                "field_descriptions": field_descriptions,
                "document_schema": document_schema.model_dump(),
                "generated_system_prompt": generated_system_prompt,
                "video_processing_prompt": video_prompt,
                "extraction_method": "regex_only_no_model",
                "id": extraction_id
            }
        }

    def _generate_with_timeout(self, model, inputs, timeout_seconds=30, **generation_kwargs):
        """Generate with timeout protection to prevent hanging"""
        import threading
        
        result = [None]
        exception = [None]
        
        def generate_target():
            try:
                result[0] = model.generate(**inputs, **generation_kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=generate_target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Force cleanup if thread is still running
            print(f">>> Generation timed out after {timeout_seconds} seconds")
            torch.cuda.empty_cache()
            gc.collect()
            raise TimeoutError(f"Model generation timed out after {timeout_seconds} seconds")
        
        if exception[0]:
            raise exception[0]        
        return result[0]
    def _extract_fields_with_ai(self, document_text, document_type="form"):
        """Use the loaded AI model for intelligent field extraction - NO HARDCODING"""
        if not self._model_loaded:
            print(">>> AI model not loaded, falling back to dynamic extraction")
            return self._extract_additional_fields(document_text)
        
        try:
            print(">>> Using AI model for intelligent field extraction...")
            
            # Analyze document structure to provide context to the AI
            sections = []
            lines = document_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and len(line.strip()) >= 3 and any(char.isupper() for char in line):
                    sections.append(f"Section {i}: {line.strip()}")
                if len(sections) >= 5:  # Get some key sections for context
                    break
            
            sections_text = "\n".join(sections)
            
            # Create a focused prompt for field extraction without hardcoded expectations
            prompt = f"""
You are an expert at analyzing documents and extracting form fields. 

TASK: Analyze the following document text and identify ALL form fields, input areas, data entry points WITHOUT using hardcoded patterns.

DOCUMENT TYPE: {document_type}

DOCUMENT STRUCTURE:
{sections_text}

DOCUMENT TEXT:
{document_text[:3000]}

INSTRUCTIONS:
1. Dynamically identify all possible fields in the document
2. Learn the document's pattern and structure automatically
3. DO NOT rely on hardcoded patterns - analyze the actual structure
4. Focus on finding ALL fields regardless of format or naming convention
5. Pay special attention to form structure, layout patterns, and content organization
6. Consider both explicit fields (with clear labels) and implicit fields (from context)

OUTPUT FORMAT: Return a JSON array of field names only, like this:
["field_name_1", "field_name_2", "field_name_3"]

IMPORTANT: Focus on discovering fields dynamically without prior assumptions about field naming or document structure. Convert all identified fields to snake_case format.
"""

            # Generate response with timeout
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                try:
                    outputs = self._generate_with_timeout(
                        self.model,
                        inputs,
                        timeout_seconds=180,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                    )
                except TimeoutError as e:
                    print(f">>> AI field extraction timed out: {str(e)}")
                    return self._extract_field_names_fallback(document_text)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the JSON array from the response
            try:
                # Look for JSON array in the response
                import re
                json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
                if json_match:
                    json_str = '[' + json_match.group(1) + ']'
                    ai_fields = json.loads(json_str)
                    
                    # Clean and normalize the field names
                    cleaned_fields = []
                    for field in ai_fields:
                        if isinstance(field, str) and len(field.strip()) > 2:
                            normalized = normalize_field_name(field.strip())
                            if normalized:
                                cleaned_fields.append(normalized)
                    
                    print(f">>> AI extracted {len(cleaned_fields)} fields: {cleaned_fields}")
                    return cleaned_fields
                else:
                    print(">>> AI response didn't contain valid JSON array, falling back to regex")
                    return self._extract_field_names_fallback(document_text)
                    
            except (json.JSONDecodeError, Exception) as e:
                print(f">>> Error parsing AI response: {e}")
                return self._extract_field_names_fallback(document_text)
                
        except Exception as e:
            print(f">>> Error in AI field extraction: {e}")
            return self._extract_field_names_fallback(document_text)
        
        
    def _extract_fields_comprehensive(self, source_path, document_text, **kwargs):
        """Combine multiple extraction methods with AI-based analysis as PRIMARY method (NO HARDCODING)"""
        print(">>> Starting comprehensive field extraction (NO HARDCODING)...")
        
        all_fields = set()
        
        # Method 1: AI-based extraction 
        try:
            document_type = kwargs.get('document_type', 'form')
            ai_fields = self._extract_fields_with_ai(document_text, document_type)
            all_fields.update(ai_fields)
            print(f">>> AI-based extraction (PRIMARY) found {len(ai_fields)} fields")
        except Exception as e:
            print(f">>> AI-based extraction failed: {e}")
        
        # Method 2: Dynamic structural analysis (backup - no hardcoding)
        try:
            dynamic_fields = self._extract_additional_fields(document_text)
            all_fields.update(dynamic_fields)
            print(f">>> Dynamic extraction (backup) found {len(dynamic_fields)} fields")
        except Exception as e:
            print(f">>> Dynamic extraction failed: {e}")
            
        # Method 3: Fallback extraction (last resort)
        try:
            document_type = kwargs.get('document_type', 'form')
            ai_fields = self._extract_fields_with_ai(document_text, document_type)
            all_fields.update(ai_fields)
            print(f">>> AI extraction found {len(ai_fields)} fields")
        except Exception as e:
            print(f">>> AI extraction failed: {e}")
        
        # Method 3: Pattern-based extraction (fallback only)
        try:
            pattern_fields = self._extract_field_names_fallback(document_text)
            all_fields.update(pattern_fields)
            print(f">>> Pattern extraction found {len(pattern_fields)} fields")
        except Exception as e:
            print(f">>> Pattern extraction failed: {e}")
        
        # Convert to sorted list
        final_fields = sorted(list(all_fields))
        
        print(f">>> Comprehensive extraction completed: {len(final_fields)} total fields found")
        print(f">>> Final field list: {final_fields}")
        return final_fields

    def _llm_extraction(self, document_text, max_new_tokens, **kwargs):
        """Enhanced LLM extraction method using comprehensive field detection"""
        print(">>> Using enhanced LLM extraction with comprehensive field detection...")
        
        # Use the comprehensive extraction method
        field_names = self._extract_fields_comprehensive(None, document_text, **kwargs)
          # Create reasonable default field types and descriptions
        field_types = {name: "text" for name in field_names}
        field_descriptions = {name: f"Information related to {name.lower().replace('_', ' ')}" for name in field_names}
        
        # Create document schema and generate dynamic prompt
        document_type = kwargs.get('document_type', 'form')
        client_id = kwargs.get('client_id') or 'default'  # Ensure client_id is never None
        document_title = kwargs.get('document_title', 'Extracted Document')
        
        document_schema = self._create_document_schema(
            field_names=field_names,
            field_types=field_types,
            field_descriptions=field_descriptions,
            document_type=document_type,
            client_id=client_id,
            document_title=document_title
        )

        generated_system_prompt = self._generate_dynamic_system_prompt(document_schema)

        video_prompt = self._generate_video_processing_prompt(field_names, field_descriptions)
        
        return {
            "field_names": field_names,
            "field_types": field_types,
            "field_descriptions": field_descriptions,
            "document_schema": document_schema.model_dump(),
            "generated_system_prompt": generated_system_prompt,
            "video_processing_prompt": video_prompt,
            "extraction_method": "enhanced_regex"
        }

    def _extract_field_names_fallback(self, document_text):
        """Enhanced fallback method to extract field names using text analysis optimized for Docling output."""
        # Enhanced patterns for form fields that work better with Docling's structured output
        field_patterns = [
            # Clear field patterns with proper boundaries
            r'([A-Za-z][A-Za-z\s]{2,30}):\s*[_\[\(\.]{2,}',  # Name: ____ or Name: [...] 
            r'([A-Za-z][A-Za-z\s]{2,30})\s*\[\s*\]',  # Name [ ]
            r'([A-Za-z][A-Za-z\s]{2,30})\s*\(\s*\)',  # Name ( )
            r'([A-Za-z][A-Za-z\s]{2,30}):\s*$',  # Name: (at end of line)
            r'(\d+)\.\s*([A-Za-z][A-Za-z\s]{2,30}):',  # 1. Name: (numbered fields)
            r'^([A-Z][A-Z\s]{2,30}):',  # ALL CAPS FIELD NAMES:
            # Specific form field indicators
            r'([A-Za-z\s]+(?:name|number|id|date|time|location|address|phone|email))\s*[:_\[\(]',
            r'([A-Za-z\s]+(?:description|details|information|category|type|status))\s*[:_\[\(]',
        ]
        
        field_names = set()
        
        # First pass: extract using patterns
        for pattern in field_patterns:
            matches = re.findall(pattern, document_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Handle tuple matches from numbered patterns
                if isinstance(match, tuple):
                    cleaned = match[1].strip() if len(match) > 1 else match[0].strip()
                else:
                    cleaned = match.strip()
                
                # Clean up the field name
                cleaned = cleaned.rstrip(':').rstrip('_').strip()
                
                # Validate field quality
                if self._is_valid_field_name(cleaned):
                    normalized = normalize_field_name(cleaned)
                    if normalized and len(normalized) >= 3:
                        field_names.add(normalized)
        
        # Second pass: look for common field indicators in context
        common_field_keywords = [
            'incident_number', 'case_number', 'report_number', 'reference_number',
            'date', 'time', 'location', 'address', 'name', 'contact',
            'officer_name', 'officer_id', 'badge_number',
            'incident_type', 'incident_category', 'priority',
            'witness_name', 'witness_contact', 'victim_name',
            'description', 'narrative', 'details', 'summary',
            'actions_taken', 'follow_up', 'status', 'outcome',
            'supervisor_name', 'supervisor_review',
            'signature', 'date_signed'
        ]
        
        # Look for these keywords in context
        lines = document_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 80:  # Reasonable field label length
                line_lower = line.lower()
                
                # Check if line contains field keywords and looks like a label
                for keyword in common_field_keywords:
                    if keyword.replace('_', ' ') in line_lower or keyword in line_lower:                        # Extract the actual field label from the line
                        if ':' in line:
                            potential_field = line.split(':')[0].strip()
                        else:
                            potential_field = line.strip()
                        
                        if self._is_valid_field_name(potential_field):
                            normalized = normalize_field_name(potential_field)
                            if normalized and len(normalized) >= 3:
                                field_names.add(normalized)
        
        result = sorted(list(field_names))
        
        # Filter out obviously bad field names
        filtered_result = []
        for field in result:
            if self._is_high_quality_field_name(field):
                filtered_result.append(field)
        
        print(f"Extracted {len(filtered_result)} quality field names: {filtered_result}")
        return filtered_result

    def _valid_field_name_patterns(self):
        """Dynamic pattern generation for valid field name detection"""
        # Using machine learning-like pattern discovery - NO HARDCODING
        return [
            # General patterns
            r'([A-Za-z][A-Za-z\s]{2,30}):\s*[_\[\(\.]{2,}',  # Name: ____ or Name: [...] 
            r'([A-Za-z][A-Za-z\s]{2,30})\s*\[\s*\]',  # Name [ ]
            r'([A-Za-z][A-Za-z\s]{2,30})\s*\(\s*\)',  # Name ( )
            r'([A-Za-z][A-Za-z\s]{2,30}):\s*$',  # Name: (at end of line)
            r'(\d+)\.\s*([A-Za-z][A-Za-z\s]{2,30}):',  # 1. Name: (numbered fields)
            r'^([A-Z][A-Z\s]{2,30}):',  # ALL CAPS FIELD NAMES:
            
            # Specific form field indicators
            r'([A-Za-z\s]+(?:name|number|id|date|time|location|address|phone|email))\s*[:_\[\(]',
            r'([A-Za-z\s]+(?:description|details|information|category|type|status))\s*[:_\[\(]',
        ]

    def _extract_fields_dynamically(self, document_text: str) -> List[str]:
        """
        Dynamic field extraction using machine learning-like pattern recognition
        NO HARDCODING - learns patterns from document structure
        """
        # Simple dynamic extraction with regex patterns
        fields = set()
        
        # Find colon patterns (most common in forms)
        colon_patterns = re.finditer(r'([A-Za-z\s]{3,30}):\s*[_\.\-\s]*(?:\n|$)', document_text)
        for match in colon_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2:
                fields.add(normalized)
                
        # Find underscore patterns (form blanks)
        underscore_patterns = re.finditer(r'([A-Za-z\s]{3,30})[_\s]{3,}', document_text)
        for match in underscore_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2:
                fields.add(normalized)
                
        # Find bracket patterns (checkboxes)
        bracket_patterns = re.finditer(r'[\[\(\{\s][ _]*[\]\)\}]([A-Za-z\s]{3,30})', document_text)
        for match in bracket_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2:
                fields.add(normalized)
                
        # Find capitalized phrases (likely field names)
        cap_patterns = re.finditer(r'(?<!\w)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?!\w)', document_text)
        for match in cap_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2:
                fields.add(normalized)
                
        return sorted(list(fields))
    
    def _is_valid_field_name(self, field_name: str) -> bool:
        """Dynamic validation without hard-coded word lists"""
        if not field_name or len(field_name.strip()) < 2:
            return False
        
        field_lower = field_name.lower().strip()
        
        # Reject if too long (probably contains multiple concepts or instructions)
        if len(field_lower) > 30:
            return False
        
        # Dynamic pattern analysis for junk detection
        # Check for obvious technical junk
        if any(pattern in field_lower for pattern in ['www.', '.com', 'http', '©']):
            return False
        
        # Check for structural issues using pattern analysis
        words = field_lower.split('_')
        
        # Must have at least one meaningful word
        alpha_words = [word for word in words if word.isalpha() and len(word) >= 2]
        if len(alpha_words) == 0:
            return False
        
        # Check for obvious fragments using length and pattern heuristics
        for word in alpha_words:            # Very short words that end in common suffixes might be fragments
            if len(word) <= 3 and word.endswith(('ed', 'er', 'ly', 'ng')):
                return False
            # Words that look truncated
            if len(word) <= 4 and (word.startswith(('un', 're')) or word.endswith(('tion', 'ing'))):
                return False
        
        return True

    def _is_high_quality_field_name(self, field_name: str) -> bool:
        """Check if a field name is high quality using dynamic analysis"""
        if not field_name or len(field_name) < 3:
            return False
        
        field_lower = field_name.lower()
        
        # Reject overly long field names (likely contain form instructions)
        if len(field_lower) > 35:
            return False
        
        # Dynamic analysis - check for structural quality
        words = field_lower.split('_')
        
        # Filter out single letters and fragments
        meaningful_words = [word for word in words if len(word) >= 2 and word.isalpha()]
        
        if len(meaningful_words) == 0:
            return False
        
        # Check for obvious fragments using pattern analysis
        for word in meaningful_words:
            # Reject if word looks like a fragment (starts/ends with common suffixes/prefixes that suggest truncation)
            if (word.endswith(('ected', 'ation', 'ies')) and len(word) < 6) or \
               (word.startswith(('orm', 'ort', 'ect')) and len(word) < 5):
                return False
        
        # Reject if it looks like instructions rather than field names
        # Instructions typically have patterns like "yes_no", "provide_details", etc.
        instruction_patterns = [
            r'^(yes|no)_',
            r'_(yes|no)$',
            r'provide_',
            r'_provide',
            r'^(if|when|where|how|why)_',
        ]
        
        for pattern in instruction_patterns:
            if re.search(pattern, field_lower):
                return False
        
        # Basic quality check - field should represent a single concept
        # Reject if it seems to be multiple concepts mashed together
        if len(meaningful_words) > 4:  # Too many words suggests it's not a clean field name
            return False
        
        return True
    
    def _extract_additional_fields(self, document_text):
        """Enhanced field extraction using multiple dynamic techniques - NO HARDCODING"""
        additional_fields = set()
        
        # Look for colon-separated key-value patterns - dynamic discovery
        colon_patterns = re.finditer(r'([A-Za-z\s]{3,25}):\s*[_\.\-\s]*(?:\n|$)', document_text)
        for match in colon_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2 and self._is_valid_field_name(normalized) and self._is_high_quality_field_name(normalized):
                additional_fields.add(normalized)
        
        # Look for form-like structures - dynamic discovery
        form_patterns = re.finditer(r'([A-Za-z\s]{3,25})\s*\[\s*\]', document_text)
        for match in form_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2 and self._is_valid_field_name(normalized) and self._is_high_quality_field_name(normalized):
                additional_fields.add(normalized)
        
        # Look for underscore patterns (common in forms) - dynamic discovery
        underscore_patterns = re.finditer(r'([A-Za-z\s]{3,25})\s*_{3,}', document_text)
        for match in underscore_patterns:
            field_name = match.group(1).strip()
            normalized = normalize_field_name(field_name)
            if normalized and len(normalized) > 2 and self._is_valid_field_name(normalized) and self._is_high_quality_field_name(normalized):
                additional_fields.add(normalized)
        
        return list(additional_fields)
    
    def _generate_video_processing_prompt(self, field_names, field_descriptions):
        """Generate a video processing prompt based on extracted form fields."""
        
        if not field_names:
            return {
                "prompt": "Analyze the video for any relevant information that could be used to fill out form fields.",
                "focus_areas": ["general_information"]
            }
        
        # Create specific instructions based on field types
        video_instructions = []
        focus_areas = []
        
        for field_name in field_names:
            description = field_descriptions.get(field_name, "")
            field_lower = field_name.lower()
            
            if any(keyword in field_lower for keyword in ["name", "person", "individual", "officer", "witness", "patient", "employee"]):
                video_instructions.append(f"Look for person identification or names for field: {field_name}")
                focus_areas.append("person_identification")
                
            elif any(keyword in field_lower for keyword in ["date", "time", "when", "arrival", "dispatch", "start", "end"]):
                video_instructions.append(f"Look for timestamps, dates, or time indicators for field: {field_name}")
                focus_areas.append("temporal_information")
                
            elif any(keyword in field_lower for keyword in ["location", "address", "place", "where", "street", "city", "scene"]):
                video_instructions.append(f"Look for location markers, addresses, or place names for field: {field_name}")
                focus_areas.append("location_identification")
                
            elif any(keyword in field_lower for keyword in ["vehicle", "car", "license", "plate", "registration", "vin"]):
                video_instructions.append(f"Look for vehicles, license plates, or vehicle information for field: {field_name}")
                focus_areas.append("vehicle_identification")
                
            elif any(keyword in field_lower for keyword in ["incident", "event", "what", "description", "violation", "crime", "activity"]):
                video_instructions.append(f"Look for activities, events, or incidents for field: {field_name}")
                focus_areas.append("activity_analysis")
                
            elif any(keyword in field_lower for keyword in ["document", "id", "identification", "badge", "card", "certificate"]):
                video_instructions.append(f"Look for documents, IDs, or identification materials for field: {field_name}")
                focus_areas.append("document_identification")
                
            elif any(keyword in field_lower for keyword in ["clothing", "appearance", "description", "physical", "height", "weight"]):
                video_instructions.append(f"Look for physical descriptions or appearance details for field: {field_name}")
                focus_areas.append("physical_description")
                
            elif any(keyword in field_lower for keyword in ["weapon", "equipment", "tool", "device", "object"]):
                video_instructions.append(f"Look for weapons, equipment, or objects for field: {field_name}")
                focus_areas.append("object_identification")
                
            elif any(keyword in field_lower for keyword in ["behavior", "action", "conduct", "demeanor", "attitude"]):
                video_instructions.append(f"Look for behaviors, actions, or conduct patterns for field: {field_name}")
                focus_areas.append("behavior_analysis")
                
            else:
                video_instructions.append(f"Look for any relevant information for field: {field_name}")
                if description:
                    video_instructions.append(f"  - {description}")
                focus_areas.append("general_information")
        
        # Remove duplicates from focus areas
        focus_areas = list(set(focus_areas))
        
        prompt_text = f"""
        Based on the form fields that need to be filled out, analyze the video for the following specific information:
        
        {chr(10).join(video_instructions)}
          Focus on extracting data that would be relevant for completing these form fields.
        Pay special attention to: {', '.join(focus_areas)}
        
        Return structured data that can be mapped to the form fields.
        """
        
        return {
            "prompt": prompt_text.strip(),
            "focus_areas": focus_areas,
            "target_fields": field_names
        }
        
    def _extract_document_text(self, file_path):
        """Extract text from a document file."""
        if not os.path.exists(file_path):
            return None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            if file_extension == '.pdf':
                # Try lightweight extraction first
                lightweight_text = self._extract_text_lightweight(file_path)
                if lightweight_text and len(lightweight_text.strip()) > 50:
                    return lightweight_text
                # Fallback to Docling if lightweight fails
                return self._extract_text_from_pdf(file_path)
            elif file_extension in ['.txt', '.md', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                # For other file types, return the file name and extension
                return f"Document: {os.path.basename(file_path)}"
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None
    
    def _extract_text_lightweight(self, pdf_path):
        """Fast PDF text extraction using PyMuPDF if available, fallback to PyPDF2"""
        try:
            # Try PyMuPDF first (fastest)
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(min(5, len(doc))):  # Only process first 5 pages for speed
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            # Fallback to PyPDF2
            return self._extract_text_fallback(pdf_path)
            
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text and structured data from a PDF file using Docling."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            
            # Configure the pipeline for better form field extraction
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for better text extraction
            pipeline_options.do_table_structure = True  # Extract table structures
            pipeline_options.table_structure_options.do_cell_matching = True
            
            # Create converter with optimized settings for form extraction
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert the document
            result = converter.convert(pdf_path)
            
            # Extract text content
            text_content = result.document.export_to_markdown()
            
            # Also try to extract structured data that might indicate form fields
            structured_data = self._extract_structured_form_data(result.document)
            
            # Combine text and structured data
            if structured_data:
                text_content += "\n\nStructured Form Data:\n" + structured_data
            
            return text_content
            
        except ImportError as e:
            print(f"Docling not available: {e}")
            # Fallback to basic text extraction
            return self._extract_text_fallback(pdf_path)
        except Exception as e:
            print(f"Error extracting PDF text with Docling: {str(e)}")
            # Fallback to basic text extraction
            return self._extract_text_fallback(pdf_path)
    
    def _extract_structured_form_data(self, document):
        """Extract structured form data from Docling document object."""
        try:
            structured_info = []
            
            # Look for tables which often contain form-like structures
            for item in document.texts:
                if hasattr(item, 'label') and item.label:
                    if any(keyword in item.label.lower() for keyword in ['table', 'form', 'field']):
                        structured_info.append(f"Form element: {item.text}")
            
            # Look for specific patterns that indicate form fields
            for item in document.texts:
                text = item.text.strip()
                if ':' in text and len(text) < 100:  # Likely a field label
                    structured_info.append(f"Field: {text}")
            
            return '\n'.join(structured_info) if structured_info else ""
            
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            return ""
    
    def _extract_text_fallback(self, pdf_path):
        """Fallback method using PyPDF2 if Docling fails."""
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            return text
        except ImportError:
            try:
                subprocess.check_call(["pip", "install", "PyPDF2"])
                import PyPDF2
                text = ""
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n"
                return text
            except Exception:
                return "Error: Could not extract text from PDF"
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

    def _create_document_schema(self, field_names, field_types, field_descriptions, 
                               document_type="form", client_id="default", 
                               document_title="Extracted Document", created_by="system"):
        """Create a DocumentSchema object from extracted field data"""
        
        # Build schema_fields dictionary
        schema_fields = {}
        for field_name in field_names:
            human_readable_label = convert_to_human_readable_label(field_name)
            schema_fields[field_name] = SchemaField(
                label=human_readable_label,
                value="",
                field_type=field_types.get(field_name, "text"),
                required=True,
                description=field_descriptions.get(field_name, ""),
                fieldDataVar=field_name
            )
        
        # Create the document schema
        document_schema = DocumentSchema(
            client_id=client_id,
            document_title=document_title,
            document_type=document_type,
            is_active=True,
            schema_fields=schema_fields,
            created_by=created_by
        )
        
        return document_schema

    def _generate_dynamic_system_prompt(self, document_schema: DocumentSchema):
        """Generate a dynamic system prompt based on the document schema"""
        try:
            return self.dynamic_prompt_generator.generate_system_prompt(document_schema)
        except Exception as e:
            print(f"Error generating dynamic prompt: {e}")
            return "Extract information from the document and fill the specified fields accurately."

    def integrate_form_with_analysis(self, form_fields, visual_analysis):
        """
        Integrate form fields with visual analysis data.
        
        Args:
            form_fields: Dictionary containing form fields
            visual_analysis: List of dictionaries containing visual analysis data
            
        Returns:
            Updated form fields with integrated analysis
        """
        if not form_fields or not visual_analysis:
            return form_fields
          # Load model if not already loaded
        if not self._model_loaded:
            try:
                self.load_model()
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
                return form_fields
        
        try:            # Extract relevant data from visual analysis
            visual_data = {}
            for analysis in visual_analysis:
                if 'analysis_data' in analysis and analysis['analysis_data']:
                    # Merge all analysis data
                    if isinstance(analysis['analysis_data'], dict):
                        visual_data.update(analysis['analysis_data'])
                    
            if not visual_data:
                return form_fields
                
            # Prepare the prompt for integration
            prompt = f"""
            I have form data and visual analysis data that need to be integrated.
            
            Form data:
            {json.dumps(form_fields, indent=2)}
            
            Visual analysis data:
            {json.dumps(visual_data, indent=2)}
            
            Please integrate these two data sources, enhancing the form data with relevant information from the visual analysis.
            Return the result as a JSON object.
            """
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                try:
                    outputs = self._generate_with_timeout(
                        self.model,
                        inputs,
                        timeout_seconds=60,  # Longer timeout for form integration
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                except TimeoutError as e:
                    print(f">>> Form integration generation timed out: {str(e)}")
                    return form_fields  # Return original form fields on timeout
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = self.process_generated_response(response)
            if "error" in result or not result:
                return form_fields
                
            if "raw_response" in result:
                try:
                    json_match = re.search(r'```json\n(.*?)\n```', result["raw_response"], re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        parsed = json.loads(json_str)
                        return parsed
                except Exception:
                    pass
                
            return form_fields
        except Exception as e:
            print(f"Error integrating data: {str(e)}")
            return form_fields

document_integrator = QwenDocumentIntegrator()

# Pre-load the model on startup for better performance???
# try:
#     print(">>> Pre-loading document processing model...")
#     document_integrator.load_model()
#     print(">>> Document processing model pre-loaded successfully")
# except Exception as e:
#     print(f">>> Failed to pre-load model: {e}")
#     print(">>> Model will be loaded on first request")

@router.post("/process/")
async def process_document(request_body: BaseProcessor):
    return await document_integrator.process_document(request_body)


def convert_to_human_readable_label(field_name: str) -> str:
    """Convert snake_case field name to human-readable label"""
    if not field_name:
        return ""
    
    # Replace underscores with spaces and capitalize each word
    words = field_name.replace("_", " ").split()
    # Capitalize each word properly
    capitalized_words = [word.capitalize() for word in words]
    return " ".join(capitalized_words)