from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

# Import routers
from app.routers import audio_processing
from app.routers import document_processing  # Add the new router

app = FastAPI(
    title="ADP Video Pipeline API",
    description="API for processing videos and documents",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_processing.router)
app.include_router(document_processing.router)  # Include the document processing router

@app.get("/")
async def root():
    return {"message": "Welcome to the ADP Video and Document Pipeline API"}

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
