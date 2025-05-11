from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import json
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional, List

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
    # Create fields_data dictionary for JSON storage
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