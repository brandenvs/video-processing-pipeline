import asyncio
import atexit
from io                 import BytesIO
import json
import os
from pathlib            import Path
import time
from typing             import Optional
import concurrent
from pydantic           import BaseModel
from routers.database_service import Db_helper
import librosa
from transformers       import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from routers        import model_management as mm
import torch
import gc
from moviepy            import VideoFileClip
from pydub              import AudioSegment
from fastapi            import APIRouter
from concurrent.futures import ThreadPoolExecutor
import functools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class AudioProcessor(BaseModel):
    system_prompt: Optional[str] = (
        "Identify speakers, emotions, and transcribe the audio content"
    )
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None
    document_key: Optional[str] = None

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
    responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)

def cleanup_executor():
    executor.shutdown(wait=True)

atexit.register(cleanup_executor)

def check_memory(device=mm.get_torch_device()):
    print("Device Loaded: ", device)

    total_mem = mm.get_total_memory() / (1024 * 1024 * 1024)
    print(f"GPU has {total_mem} GBs")

    free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
    return (free_mem_gb, total_mem)

@router.post("/process/")
async def process_audio(request_body: AudioProcessor):
    loop = asyncio.get_running_loop()
    torch.cuda.empty_cache()
    gc.collect()
    check_memory()

    inference_task = functools.partial(
        model_manager.inference, **request_body.model_dump()
    )
    results = await loop.run_in_executor(executor, inference_task)

    analysis_ids = []
    for analysis_data in results:
        # Store in database
        db_helper = Db_helper()
        analysis_id = db_helper.audio_analysis(
            analysis_data, source_path=request_body.source_path
        )

        if analysis_id:
            analysis_ids.append(analysis_id)

    return {
        "status": "success",
        "analysis_ids": analysis_ids,
        "results": results,
    }

class Qwen2_Audio:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = mm.get_torch_device()
        self.dtype = None
        self._model_loaded = False

    def process_generated_response(self, generated_response: str, sequence_no: int):
        if generated_response.startswith("```json"):
            try:
                lines = generated_response.strip().splitlines()
                json_content = "\n".join(lines[1:-1])

                json_response = json.loads(json_content)
                spread_response = {**json_response, **{"sequence_no": sequence_no}}
                return spread_response

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {"error": str(e), "sequence_no": sequence_no}

        return {"raw_response": generated_response, "sequence_no": sequence_no}

    def load_model(self):
        if self._model_loaded:
            return

        torch.cuda.empty_cache()
        print(gc.collect())
        check_memory()
        torch.manual_seed(42)

        # Use the newer Qwen model for better multimodal capabilities
        model_id = "Qwen/Qwen2.5-Omni-3B"
        self.model_checkpoint = os.path.join(
            "models/audio_processor", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        # Determine precision
        self.dtype = (
            torch.float16
            if mm.should_use_fp16(self.device)
            else torch.bfloat16 if mm.should_use_bf16(self.device) else torch.float32
        )
        print(f'>>> Selected DType: {self.dtype}')

        # Configure quantization if needed
        if self.dtype != torch.float16:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        print(f'>>> Selected Device: {self.device}')
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=self.dtype,
            device_map=self.device,
            quantization_config=quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        self._model_loaded = True

    def process_batch(self, start_ms, end_ms, audio, idx):
        """Process a batch of audio and save it to disk"""
        batch = audio[start_ms:end_ms]
        batch_filename = os.path.join('audio_segments', f"batch_{idx+1}.wav")
        batch.export(batch_filename, format="wav")
        return batch_filename

    def inference(
        self,
        system_prompt,
        max_new_tokens=512,
        source_path=None,
    ):
        results = []
        os.makedirs('audio_segments', exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Load audio file
            print(">>> Loading audio file")
            if source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Extract audio from video
                print(">>> Extracting audio from video")
                video = VideoFileClip(source_path)
                audio = AudioSegment.from_file(video.audio.filename)
                video.close()
            else:
                # Direct audio file
                audio = AudioSegment.from_file(source_path)
            
            # Set batch duration for processing
            batch_duration = 5  # 5 seconds per batch for better context
            batch_duration_ms = batch_duration * 1000
            audio_duration_ms = len(audio)
            
            # Create batches
            print(">>> Creating audio batches")
            segments = []
            for start_ms in range(0, audio_duration_ms, batch_duration_ms):
                end_ms = min(start_ms + batch_duration_ms, audio_duration_ms)
                segments.append((start_ms, end_ms, audio, len(segments)))

            # Process batches in parallel
            batches = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                print('>>> Processing audio segments')
                batch_futures = {
                    executor.submit(self.process_batch, *segment): segment[3]
                    for segment in segments
                }
                
                for future in concurrent.futures.as_completed(batch_futures):
                    batch = future.result()
                    if batch:
                        batches.append(batch)
            
            batches.sort(key=lambda filename: int(filename.split('_')[-1].split('.')[0]))
            
            elapsed_time = time.time() - start_time
            print(f"Batching completed in {elapsed_time:.2f} seconds")
            
            # Load model if not already loaded
            if not self._model_loaded:
                print('>>> Loading model into memory')
                self.load_model()
            
            sequence_no = 0
            for batch in batches:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": """You are an expert audio analysis system.
                            Analyze the audio and structure a concise JSON response with the following:
                            
                            1. Speakers: Identify different speakers and assign them IDs
                            2. Transcription: Provide a word-for-word transcription of what is said
                            3. Emotions: Detect emotional states of each speaker
                            4. Background: Identify any background noises or sounds
                            5. Confidence: Provide a confidence score for your analysis (0-100)
                            
                            Structure the response as a JSON with these keys.""",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "audio_url": batch,
                            },
                            {"type": "text", "text": system_prompt},
                            ]
                        }
                    ]
                    
                    print('>>> Preparing for inference')
                    # Prepare for inference
                    system_prompts = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    check_memory(self.device)
                    
                    audios = []
                    for message in messages:
                        if isinstance(message["content"], list):
                            for ele in message["content"]:
                                if ele["type"] == "audio":
                                    audio_path = ele["audio_url"]
                                    audio_data, sr = librosa.load(
                                        audio_path,
                                        sr=self.processor.feature_extractor.sampling_rate,
                                    )
                                    audios.append(audio_data)
                    
                    print('>>> Processing inputs')
                    inputs = self.processor(
                        text=system_prompts, audios=audios, return_tensors="pt", padding=True
                    )
                    inputs.input_ids = inputs.input_ids.to(self.device)
                    
                    # Free memory before generating
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    print('>>> Generating response')
                    generate_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                    
                    generated_response = self.processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    
                    sequence_no += 1
                    processed_response = self.process_generated_response(generated_response[0], sequence_no)
                    
                    finished_in = time.time() - start_time
                    result = {
                        **processed_response,
                        "finished_in": round(finished_in, 3)
                    }
                    print(json.dumps(result, indent=2))
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing audio batch {batch}: {str(e)}")
                finally:
                    # Delete batch file immediately after processing
                    try:
                        if os.path.exists(batch):
                            os.remove(batch)
                            print(f"Deleted processed audio batch: {batch}")
                    except Exception as cleanup_err:
                        print(f"Failed to clean up audio batch file {batch}: {str(cleanup_err)}")
                    
                    # Free memory after each batch
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Final cleanup of any remaining files (safety measure)
            try:
                if os.path.exists('audio_segments'):
                    for f in os.listdir('audio_segments'):
                        file_path = os.path.join('audio_segments', f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Cleaned up remaining audio file: {file_path}")
            except Exception as final_err:
                print(f"Error in final audio cleanup: {str(final_err)}")
                
            return results
        except Exception as e:
            print(f"Fatal error in audio inference: {str(e)}")
            # Attempt cleanup even on catastrophic failure
            try:
                if os.path.exists('audio_segments'):
                    for f in os.listdir('audio_segments'):
                        file_path = os.path.join('audio_segments', f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            except:
                pass
            raise # Will # Re-raises the original exception <-