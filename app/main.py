from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.routers import audio_processing
from app.routers import document_processing  # Make sure this is included

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
app.include_router(document_processing.router)  # Make sure this is included

@app.get("/")
async def root():
    return {"message": "Welcome to the ADP Video and Document Pipeline API"}

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
