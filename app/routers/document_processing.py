from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional

# Import the DB configuration from the original project
from app.routers.database_service import DB_CONFIG
import psycopg2

# Create router
router = APIRouter(
    prefix="/documents",
    tags=["document-processing"],
    responses={404: {"description": "Not found"}},
)

# Create a base class for our models
Base = declarative_base()

# Define the FormField model - this is now just for ORM, not for table creation
class FormField(Base):
    """Database model for storing form fields and their properties"""
    __tablename__ = 'form_fields'
    
    id = Column(Integer, primary_key=True)
    document_name = Column(String(255))
    field_name = Column(String(255))
    field_value = Column(Text)
    field_length = Column(Integer)
    field_type = Column(String(50))
    processed_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<FormField(document='{self.document_name}', field='{self.field_name}', value='{self.field_value}')>"

# Define the FormDocument model - this is now just for ORM, not for table creation
class FormDocument(Base):
    """Database model for storing form documents and their fields"""
    __tablename__ = 'form_documents'
    
    id = Column(Integer, primary_key=True)
    document_name = Column(String(255))
    document_path = Column(String(512))
    video_path = Column(String(512), nullable=True)
    fields_json = Column(JSON)  # Store all fields as JSON
    processed_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<FormDocument(document='{self.document_name}', fields={len(self.fields_json) if self.fields_json else 0})>"

# Function to initialize the database - modified to not create tables
def init_database():
    """Initialize the database connection using the project's DB config"""
    try:
        # Create PostgreSQL connection string
        db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        
        # Print connection details for debugging (remove in production)
        print(f"Connecting to database at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        
        # Create engine but don't create tables - they're created in the updated database_setup.py @branden
        engine = create_engine(db_url)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
        
        return engine, Session
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        return None, None

# Function to store form fields in the database
def store_form_fields(form_fields, document_name, session, video_path=None):
    """Store form fields in the database"""
    # ... existing code ...
    fields_data = {}
    
    for label, value in form_fields.items():
        # Clean up the label
        clean_label = label.rstrip(':')
        
        # Determine field type based on value
        if value.lower() in ['yes', 'no', 'true', 'false']:
            field_type = 'boolean'
        elif value.replace('.', '', 1).isdigit():
            field_type = 'numeric'
        else:
            field_type = 'text'
        
        # Add to fields_data
        fields_data[clean_label] = {
            'value': value,
            'length': len(value),
            'type': field_type
        }
        
        # Create a new FormField record
        form_field = FormField(
            document_name=document_name,
            field_name=clean_label,
            field_value=value,
            field_length=len(value),
            field_type=field_type
        )
        
        # Add to session
        session.add(form_field)
    
    # Create a FormDocument record with video path
    form_document = FormDocument(
        document_name=document_name,
        document_path=document_name,  # Using document_name as path for now
        video_path=video_path,        # Add video path
        fields_json=fields_data
    )
    
    # Add to session
    session.add(form_document)
    
    # Commit the session
    session.commit()
    print(f"Stored {len(form_fields)} form fields in the database")
    
    return form_document

# Function to process a document
def process_document(file_path, video_path=None):
    """Process a document and extract form fields"""
    # ... existing code ...
    try:
        from docling.document_converter import DocumentConverter
        
        # Create converter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        document = result.document
        
        # Extract form fields
        form_field_candidates = []
        if hasattr(document, 'texts') and document.texts:
            for i, text in enumerate(document.texts):
                if hasattr(text, 'text') and text.text.endswith(':'):
                    form_field_candidates.append((i, text.text))
        
        # Extract form fields
        form_fields = {}
        if form_field_candidates and hasattr(document, 'texts'):
            for idx, label in form_field_candidates:
                # Look for the value in the next few text elements
                value_found = False
                for i in range(idx + 1, min(idx + 10, len(document.texts))):
                    if i >= len(document.texts):
                        break
                    
                    text = document.texts[i]
                    if not hasattr(text, 'text'):
                        continue
                        
                    # Skip if it's another label
                    if text.text.endswith(':'):
                        break
                    
                    # Skip checkbox indicators but capture their state
                    if '/Off' in text.text:
                        form_fields[label] = "No"
                        value_found = True
                        break
                    elif '/Yes' in text.text:
                        form_fields[label] = "Yes"
                        value_found = True
                        break
                    
                    # Skip empty or whitespace-only values
                    if not text.text.strip():
                        continue
                    
                    # This is likely the value
                    form_fields[label] = text.text
                    value_found = True
                    break
                
                # If no value was found, set it as empty
                if not value_found:
                    form_fields[label] = ""
        
        # Store in database
        _, Session = init_database()
        if Session:
            session = Session()
            document_name = os.path.basename(file_path)
            form_document = store_form_fields(form_fields, document_name, session, video_path)
            session.close()
            
            return {
                "document_name": document_name,
                "fields_count": len(form_fields),
                "fields": form_fields,
                "video_path": video_path
            }
        else:
            return {
                "error": "Database connection failed",
                "document_name": os.path.basename(file_path),
                "fields_count": len(form_fields),
                "fields": form_fields
            }
            
    except Exception as e:
        return {"error": str(e)}

# API Endpoints
@router.post("/process")
async def process_document_endpoint(
    file: UploadFile = File(...),
    video_path: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Process a document file and extract form fields"""
    # ... existing code ...
    # Create input directory if it doesn't exist
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(input_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process the document
    if background_tasks:
        # Process in background
        background_tasks.add_task(process_document, file_path, video_path)
        return JSONResponse(content={"message": "Document processing started in background", "file": file.filename})
    else:
        # Process immediately
        result = process_document(file_path, video_path)
        return JSONResponse(content=result)

@router.get("/list")
async def list_documents():
    """List all processed documents"""
    # ... existing code ...
    _, Session = init_database()
    if not Session:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    session = Session()
    documents = session.query(FormDocument).all()
    
    result = []
    for doc in documents:
        result.append({
            "id": doc.id,
            "document_name": doc.document_name,
            "fields_count": len(doc.fields_json) if doc.fields_json else 0,
            "video_path": doc.video_path,
            "processed_date": doc.processed_date.isoformat()
        })
    
    session.close()
    return JSONResponse(content={"documents": result})

@router.get("/{document_id}")
async def get_document(document_id: int):
    """Get a specific document by ID"""
    # ... existing code ...
    _, Session = init_database()
    if not Session:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    session = Session()
    document = session.query(FormDocument).filter(FormDocument.id == document_id).first()
    
    if not document:
        session.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    result = {
        "id": document.id,
        "document_name": document.document_name,
        "document_path": document.document_path,
        "video_path": document.video_path,
        "fields": document.fields_json,
        "processed_date": document.processed_date.isoformat()
    }
    
    session.close()
    return JSONResponse(content=result)


class Qwen_Document_Integrator:
    def __init__(self):
        self.model_path = "C:\\Users\\liano\\OneDrive\\Documents\\REST-API\\docling\\models\\Qwen3-1.7B-UD-Q8_K_XL.gguf"
        self.model = None
        self._model_loaded = False

    def load_model(self):
        if self._model_loaded:
            return

        try:
            from llama_cpp import Llama
            
            print(f">>> Loading Qwen model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_batch=512,
                verbose=False
            )
            self._model_loaded = True
            print(">>> Model loaded successfully")
        except Exception as e:
            print(f">>> Error loading model: {e}")
            raise

    def process_generated_response(self, generated_response: str):
        if generated_response.startswith("```json"):
            try:
                lines = generated_response.strip().splitlines()
                json_content = "\n".join(lines[1:-1])

                json_response = json.loads(json_content)
                return json_response

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {"error": str(e)}

        return {"raw_response": generated_response}

    def integrate_form_with_analysis(self, form_fields, visual_analysis, audio_analysis=None):
        """
        Integrate form fields with visual and audio analysis data
        """
        if not self._model_loaded:
            self.load_model()
            
        # Find empty fields that need to be filled
        empty_fields = {}
        for field_name, field_data in form_fields.items():
            # Check if the field is empty or has minimal content
            if not field_data.get('value') or field_data.get('value').strip() == "":
                empty_fields[field_name] = field_data
        
        # If no empty fields, return the original form fields
        if not empty_fields:
            print(">>> No empty fields to fill")
            return form_fields
        
        print(f">>> Found {len(empty_fields)} empty fields to fill")
        
        # Extract the actual analysis data from the visual_analysis rows
        analysis_data = {}
        if visual_analysis and len(visual_analysis) > 0:
            for row in visual_analysis:
                if 'analysis_data' in row and row['analysis_data']:
                    # If it's already a dict, use it directly
                    if isinstance(row['analysis_data'], dict):
                        analysis_data = row['analysis_data']
                        break
        
        # If no analysis data found, return original fields
        if not analysis_data:
            print(">>> No analysis data found in visual_analysis")
            return form_fields
        
        print(">>>preparing prompt...")
        # Prepare the prompt with more specific instructions
        system_prompt = """You are an expert document analysis system.
        Your task is to fill in empty form fields using video analysis data.
        
        Examine the empty form fields and the video analysis data carefully.
        For each empty field, find the most relevant information in the analysis data.
        
        IMPORTANT: You must return a JSON object that has the EXACT SAME STRUCTURE as the original form fields,
        but with values filled in for the empty fields where relevant information exists.
        
        For example, if the original form has:
        {
        "License plate/s": {
            "type": "text",
            "value": "",
            "length": 0
        }
        }
        
        And the analysis data contains license plate information, your response should be:
        {
        "License plate/s": {
            "type": "text",
            "value": "ABC-123, XYZ-789",
            "length": 15
        }
        }
        
        Only fill in fields where there is a clear match with the analysis data.
        If there's no relevant information for a field, replace it with N/A.
        DO NOT change the structure of the form fields or add new fields.
        """
        
        # Convert data to strings for the prompt
        print(">>> Converting prompt data to strings...")
        form_fields_str = json.dumps(form_fields, indent=2)
        analysis_data_str = json.dumps(analysis_data, indent=2)
        
        user_prompt = f"""Here are the form fields with empty values that need to be filled:
        {form_fields_str}
        
        Here is the video analysis data:
        {analysis_data_str}
        
        Please fill in the empty form fields with relevant information from the video analysis data.
        Remember to maintain the exact same structure as the original form fields.
        """
        
        # Generate completion
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        print(">>> Generating integration response")
        response = self.model.create_completion(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>"]
        )
        
        generated_text = response["choices"][0]["text"]
        print(f">>> Generated response length: {len(generated_text)}")
        
        # Process the generated response
        try:
            # Try to extract JSON from the response
            if "```json" in generated_text:
                # Extract JSON from code block
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                json_content = generated_text[json_start:json_end].strip()
                updated_fields = json.loads(json_content)
            else:
                # Try to parse the whole response as JSON
                updated_fields = json.loads(generated_text.strip())
            
            # Validate the updated fields
            if not isinstance(updated_fields, dict):
                print(">>> Error: Model response is not a dictionary")
                return form_fields
            
            # Update the original form fields with the filled values
            result_fields = form_fields.copy()
            fields_updated = 0
            
            for field_name, field_data in updated_fields.items():
                if field_name in result_fields:
                    # Check if the field has been updated with a non-empty value
                    if isinstance(field_data, dict) and 'value' in field_data and field_data['value'].strip():
                        result_fields[field_name] = field_data
                        # Update the length if it wasn't updated
                        if 'length' in result_fields[field_name]:
                            result_fields[field_name]['length'] = len(field_data['value'])
                        fields_updated += 1
            
            print(f">>> Updated {fields_updated} fields with analysis data")
            return result_fields
            
        except json.JSONDecodeError as e:
            print(f">>> JSON decode error: {e}")
            print(f">>> Raw response: {generated_text[:100]}...")
            
            # Fallback: Try to manually extract field values from the text
            updated_fields = form_fields.copy()
            
            # Simple pattern matching for field values
            for field_name in empty_fields:
                field_pattern = f'"{field_name}"\\s*:\\s*{{[^}}]*"value"\\s*:\\s*"([^"]*)"'
                import re
                match = re.search(field_pattern, generated_text)
                if match:
                    value = match.group(1).strip()
                    if value:
                        updated_fields[field_name]['value'] = value
                        updated_fields[field_name]['length'] = len(value)
            
            return updated_fields

def integrate_form_with_analysis_data(document_id):
    """
    Retrieve form data and analysis data, then integrate them using the Qwen model
    """
    debug_info = {}
    conn = None
    try:
        # Get form document using SQLAlchemy (this part works fine)
        engine, Session = init_database()
        if not Session:
            return {"error": "Database connection failed"}
            
        session = Session()
        form_document = session.query(FormDocument).filter(FormDocument.id == document_id).first()
        
        if not form_document:
            session.close()
            return {"error": f"Document with ID {document_id} not found"}
            
        # Get form fields
        form_fields = form_document.fields_json
        
        # Get video path
        video_path = form_document.video_path
        if not video_path:
            session.close()
            return {"error": "No associated video path found for this document"}
        
        debug_info["video_path"] = video_path
        
        # Initialize direct database connection with psycopg2
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        
        # Use direct SQL query instead of SQLAlchemy
        query = "SELECT * FROM visual_analysis WHERE source_path = %s"
        cursor.execute(query, (video_path,))
        
        # Fetch all rows
        visual_analysis_rows = cursor.fetchall()
        
        if not visual_analysis_rows:
            session.close()
            cursor.close()
            conn.close()
            return {"error": "No visual analysis data found for the associated video", "debug": debug_info}
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        debug_info["columns"] = column_names
        
        # Convert visual analysis to dictionary
        visual_data = []
        for row in visual_analysis_rows:
            # Convert row to dict using column names
            row_dict = {column_names[i]: row[i] for i in range(len(column_names))}
            
            # Convert datetime objects to strings
            for key, value in row_dict.items():
                if isinstance(value, datetime.datetime):
                    row_dict[key] = value.isoformat()
            
            # If analysis_data is a JSON string, parse it
            if 'analysis_data' in row_dict and isinstance(row_dict['analysis_data'], str):
                try:
                    row_dict['analysis_data'] = json.loads(row_dict['analysis_data'])
                except json.JSONDecodeError:
                    pass
                    
            visual_data.append(row_dict)
        
        # Initialize integrator and process data
        integrator = Qwen_Document_Integrator()
        updated_fields = integrator.integrate_form_with_analysis(form_fields, visual_data)
        
        # Update form document with integrated data
        form_document.fields_json = updated_fields
        session.commit()
        
        session.close()
        cursor.close()
        conn.close()
        
        return {
            "document_id": document_id,
            "message": "Document fields updated with analysis data",
            "updated_fields": updated_fields,
            "debug": debug_info
        }
        
    except Exception as e:
        debug_info["error"] = str(e)
        if 'session' in locals() and session:
            session.close()
        if conn:
            conn.close()
        return {"error": str(e), "debug": debug_info}


# Fix the endpoint path - it should be consistent with the router prefix
@router.post("/{document_id}/integrate")
async def integrate_document_with_analysis(document_id: int):
    """
    Integrate document form fields with video/audio analysis data
    """
    result = integrate_form_with_analysis_data(document_id)
    return result