
# ADP Video Pipeline

## Overview

ADP Video Pipeline is a comprehensive AI-powered media processing framework designed for StadPrin. The system processes and analyzes multiple media types including video, audio, and documents using state-of-the-art AI models to extract structured information.

## Core Components

### Video Pipeline
- Processes video content using Qwen2.5-VL-3B-Instruct model
- Implements intelligent scene detection to split videos into meaningful segments
- Analyzes content frame-by-frame to identify objects, people, activities, and sentiment
- Outputs structured JSON data for each video segment
- Database integration for storing and retrieving analysis results

### Audio Pipeline
- Leverages Qwen2-Audio-7B-Instruct for audio processing
- Transcribes speech with high accuracy
- Identifies different speakers and their emotions
- Detects background noises and audio events
- Processes audio from standalone files or extracted from video

### Document Pipeline
- Uses Qwen3-1.7B-GGUF for document processing
- Extracts structured information from PDF documents
- Converts PDFs to images for AI processing
- Parses forms, reports, and technical documents
- Maps extracted fields to structured database schemas

## Technical Stack
- **Backend**: FastAPI
- **AI Models**: Qwen series (vision, audio, language)
- **Database**: Firebase and Firestore
- **Processing**: CUDA-enabled for GPU acceleration
- **Deployment**: Docker containerization
- **Server-side**: adp-functions

## Setup Instructions

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   pip install -r additional_requirements.txt
   ```
3. Configure Firebase:
   ```
   Set up your Firebase credentials and Firestore collections
   ```
4. Start the server:
   ```
   uvicorn app.main:app --reload
   ```
5. Or use Docker:
   ```
   docker-compose up -d
   ```

## API Endpoints

- `/video/process/` - Process video content
- `/audio/process/` - Process audio content  
- `/document/process/` - Process document content

## Development Roadmap

- [x] Integrate document processing pipeline
- [x] Database integration for document processing
- [x] Convert JSON fields into queries
- [x] Integrate video processing pipeline
- [x] Database integration for video processing
- [ ] Complete audio to text pipeline integration
- [ ] Build vectorizers for efficient retrieval
- [ ] Implement RAG (Retrieval-Augmented Generation) with Firestore
- [ ] Build frontend dashboard
- [ ] Create customer upload portal

## Requirements

See `requirements.txt` and `requirements_server.txt` for detailed dependencies.


