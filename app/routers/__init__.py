from app.routers import document_process_v2
from app.routers import video_processing
from app.routers import audio_processing
from app.routers import model_management
from app.routers import cli_args
from app.routers import options

ROUTERS = {
    'VideoProcessing': video_processing,
    'AudioProcessing': audio_processing,
    'ModelManagement': model_management,
    'CLIArgs': cli_args,
    'Options': options,
    'DocumentProcessingV2': document_process_v2
}
