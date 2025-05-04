import subprocess
import os
import re
import whisper
from transformers import pipeline

# Set environment variables to disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Prefer offline models when available

def ensure_model_directory():
    """
    Ensures the models directory exists
    """
    os.makedirs("models", exist_ok=True)

def get_model_path(model_name):
    """
    Returns the path to the local model directory
    """
    return os.path.join("models", model_name.replace("/", "_"))

def load_pipeline(task, model_name):
    """
    Loads a pipeline, first checking if it exists locally.
    If not, downloads and saves it locally for future use.
    Uses CPU only.
    """
    model_path = get_model_path(model_name)
    
    if os.path.exists(model_path):
        print(f"Loading {task} model from local directory: {model_path}")
        return pipeline(task, model=model_path, device=-1)  # device=-1 forces CPU
    else:
        print(f"Downloading {task} model '{model_name}' and saving locally")
        ensure_model_directory()
        pipe = pipeline(task, model=model_name, device=-1)  # device=-1 forces CPU
        pipe.save_pretrained(model_path)
        return pipe

def extract_audio_for_transcription(video_path, output_path=None):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + "_transcript_ready.wav"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                  # disable video
        "-acodec", "pcm_s16le", # WAV format, 16-bit PCM
        "-ar", "16000",         # 16 kHz
        "-ac", "1",             # mono channel
        output_path,
        "-y"   
    ]

    subprocess.run(command, check=True)
    print(f"Transcription-ready audio saved to: {output_path}")
    return output_path

def transcribe_audio(audio_path, model_name='base'):
    """
    Loads the Whisper model and transcribes the given audio file.
    Checks if the model exists locally before downloading.
    """
    model_path = get_model_path(f"whisper_{model_name}")
    
    if os.path.exists(model_path):
        print(f"Loading Whisper model from local directory: {model_path}")
        model = whisper.load_model(model_name, download_root=model_path)
    else:
        print(f"Downloading Whisper model '{model_name}' and saving locally")
        ensure_model_directory()
        # First load and save the model
        model = whisper.load_model(model_name)
        os.makedirs(model_path, exist_ok=True)
        # Whisper models are saved automatically to ~/.cache/whisper
        # We'll create a symbolic link to our models directory
        whisper_cache = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(whisper_cache):
            for file in os.listdir(whisper_cache):
                if model_name in file:
                    src = os.path.join(whisper_cache, file)
                    dst = os.path.join(model_path, file)
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
    
    result = model.transcribe(audio_path)
    return result['text']

def split_into_sentences(text):
    """
    Splits text into sentences using regular expressions.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def format_transcript(text):
    """
    Formats the transcript by splitting it into sentences and joining them with newlines.
    """
    sentences = split_into_sentences(text)
    return "\n".join(sentences)

def fix_transcript_sentence(sentence, grammar_corrector):
    """
    Fixes a single sentence using the grammar correction model.
    """
    corrected = grammar_corrector(sentence, max_length=512)
    return corrected[0]['generated_text']

def fix_transcript_full(transcript):
    """
    Processes each sentence individually for grammar correction.
    """
    grammar_corrector = load_pipeline("text2text-generation", "prithivida/grammar_error_correcter_v1")
    sentences = split_into_sentences(transcript)
    fixed_sentences = [fix_transcript_sentence(sentence, grammar_corrector) for sentence in sentences]
    return "\n".join(fixed_sentences)

def hierarchical_summary(text, summarizer, chunk_size=3000):
    """
    If the text is longer than chunk_size, it splits it into chunks,
    summarizes each, and then summarizes the combined summaries.
    """
    if len(text) <= chunk_size:
        return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    else:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        combined_summary = " ".join(chunk_summaries)
        # Summarize the combined summaries if it's too long
        if len(combined_summary) > chunk_size:
            return summarizer(combined_summary, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        else:
            return combined_summary

def tone_analysis(transcript):
    """
    Analyzes the tone using a zero-shot classification pipeline.
    Candidate labels include "formal", "informal", "angry", "happy", "sad", and "neutral".
    """
    classifier = load_pipeline("zero-shot-classification", "facebook/bart-large-mnli")
    candidate_labels = ["formal", "informal", "angry", "happy", "sad", "neutral"]
    result = classifier(transcript, candidate_labels, truncation=True)
    return result

def analyze_transcript(transcript):
    """
    Performs full analysis:
      - Hierarchical summarization for overall summary.
      - Sentiment analysis.
      - Tone analysis.
      - Grammar correction per sentence.
      - Urgency check for specific keywords.
    """
    # Initialize pipelines
    summarizer = load_pipeline("summarization", "facebook/bart-large-cnn")
    sentiment_analyzer = load_pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")

    # Hierarchical Summary
    summary = hierarchical_summary(transcript, summarizer, chunk_size=3000)

    # Sentiment Analysis (with truncation to prevent overflow)
    sentiment = sentiment_analyzer(transcript, truncation=True)[0]

    # Tone Analysis
    tone = tone_analysis(transcript)

    # Fixed Transcript (grammar correction per sentence)
    fixed_transcript = fix_transcript_full(transcript)

    # Urgency check: any of these keywords present?
    urgent_keywords = ["emergency", "urgent", "immediately", "alert", "crisis"]
    contains_urgent = any(keyword in transcript.lower() for keyword in urgent_keywords)

    return {
        "summary": summary,
        "sentiment": sentiment,
        "tone": tone,
        "fixed_transcript": fixed_transcript,
        "urgent": contains_urgent,
    }

def direct_transcription(audio_file, model_name):
    print("Starting direct transcription using Whisper...\n")
    transcript = transcribe_audio(audio_file, model_name=model_name)

    # Format the original transcript (one sentence per line)
    formatted_original = format_transcript(transcript)

    analysis = analyze_transcript(transcript)

    # Nicely printed output:
    print("=== Original Transcript ===")
    print(formatted_original)
    
    print("\n=== Fixed Transcript (Grammar Correction) ===")
    print(analysis["fixed_transcript"])
    
    print("\n=== Summary of Transcript ===")
    print(analysis["summary"])
    
    print("\n=== Sentiment Analysis ===")
    print(f"Label: {analysis['sentiment']['label']} (Score: {analysis['sentiment']['score']:.4f})")
    
    print("\n=== Tone Analysis (Labels and Scores) ===")
    
    for label, score in zip(analysis["tone"]["labels"], analysis["tone"]["scores"]):
        print(f" - {label}: {score:.4f}")
        
    print("\n=== Urgency Check ===")
    if analysis["urgent"]:
        print("Urgent keywords detected in the transcript.")
    else:
        print("No urgent keywords detected in the transcript.")
        
    return formatted_original, analysis
