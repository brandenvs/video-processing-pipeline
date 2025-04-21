from .audio_to_text import extract_audio_for_transcription,  direct_transcription

SERVICES = {
    'ExtractAudio': extract_audio_for_transcription,
    'DirectTranscription': direct_transcription
}
