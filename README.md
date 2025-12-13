# ADP Video Pipeline

**ADP Video Pipeline** is a comprehensive, AI-powered media processing framework designed to extract structured intelligence from unstructured data. By leveraging state-of-the-art multimodal AI models, the system ingests video, audio, and documents to identify objects, analyze sentiment, transcribe speech, and parse complex forms.

## 🚀 Key Features

### 🎥 Video Intelligence

Powered by **Qwen2.5-VL-3B-Instruct**.

  * **Scene Detection:** Intelligently splits videos into meaningful semantic segments.
  * **Frame Analysis:** Identifies objects, people, and activities frame-by-frame.
  * **Sentiment Analysis:** Detects emotional tone within visual context.
  * **Structured Output:** Exports analysis as queryable JSON data.

### 🎙️ Audio Intelligence

Powered by **Qwen2-Audio-7B-Instruct**.

  * **High-Fidelity Transcription:** Converts speech to text with high accuracy.
  * **Speaker Diarization:** Distinguishes between different speakers.
  * **Emotion & Event Detection:** Identifies speaker emotions and background audio events.
  * **Flexible Input:** Processes standalone audio files or tracks extracted from video.

### 📄 Document Intelligence

Powered by **Qwen3-1.7B-GGUF**.

  * **PDF Processing:** Converts and analyzes PDF documents as images.
  * **Form Parsing:** Extracts fields from reports, forms, and technical documents.
  * **Schema Mapping:** Automatically maps extracted data to structured database schemas.

-----

## 🛠️ Technical Stack

  * **Framework:** FastAPI
  * **AI Models:** Qwen Series (Vision, Audio, Language)
  * **Database:** Firebase & Firestore
  * **Infrastructure:** Docker & `adp-functions`
  * **Acceleration:** CUDA-enabled for GPU processing

-----

## ⚙️ Installation

### Prerequisites

  * Python 3.8+
  * NVIDIA GPU with CUDA support (recommended for model inference)
  * Firebase Project Credentials

### Local Setup

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/adp-video-pipeline.git
    cd adp-video-pipeline
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    pip install -r requirements.txt
    pip install -r additional_requirements.txt
    ```

3.  **Configuration**
    Create a `.env` file or configure your environment variables with your Firebase credentials. Ensure your Firestore collections are initialized.

-----

## 🏃 Usage

### Running Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Running with Docker

Deploy the entire stack using Docker Compose:

```bash
docker-compose up -d
```

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video/process/` | POST | Upload and process video content for scene and object detection. |
| `/audio/process/` | POST | Upload and transcribe audio files with emotion detection. |
| `/document/process/` | POST | Upload PDF documents for extraction and schema mapping. |

-----

## 🗺️ Roadmap

  - [x] **Document Pipeline:** Integration of Qwen3 for PDF parsing.
  - [x] **Video Pipeline:** Integration of Qwen2.5-VL for scene analysis.
  - [x] **Database Integration:** Full Firestore support for Video and Document results.
  - [x] **Query Engine:** Conversion of JSON fields into queryable formats.
  - [ ] **Audio Pipeline:** Complete "Audio to Text" integration.
  - [ ] **Vector Search:** Build vectorizers for semantic retrieval.
  - [ ] **RAG Implementation:** Retrieval-Augmented Generation with Firestore.

-----

## 🤝 Contributing

Contributions are welcome\!

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
