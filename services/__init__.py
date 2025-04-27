from .audio_to_text import extract_audio_for_transcription, direct_transcription
from .image_to_text import ImageToTextService
from .comfy import model_management

# Initialize image-to-text service
image_to_text_service = ImageToTextService()

SERVICES = {
    'ExtractAudio': extract_audio_for_transcription,
    'DirectTranscription': direct_transcription,
    'ImageToText': image_to_text_service.infer,
    'GetTorchDevice': model_management.get_torch_device
}