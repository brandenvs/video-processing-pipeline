from routers import document_process_v2
from routers import video_processing
from routers import audio_processing
from routers import model_management
from routers import cli_args
from routers import options
from routers import database_service
from routers import document_processing


ROUTES = {
    'VideoProcessing': video_processing,
    'AudioProcessing': audio_processing,
    'ModelManagement': model_management,
    'CLIArgs': cli_args,
    'Options': options,
    'DatabaseService': database_service,
    'DocumentProcessing': document_processing,
    'DocumentProcessingV2': document_process_v2
}