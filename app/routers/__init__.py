from app.routers import video_processing
from app.routers import audio_processing

ROUTES = {
    'VideoProcessing': video_processing,
    'AudioProcessing': audio_processing
}