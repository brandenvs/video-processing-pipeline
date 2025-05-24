import json
import os
from pathlib import Path
import time
from typing import Optional
import concurrent
from pydantic import BaseModel
import torch
import gc
import asyncio
from torch.nn.attention import SDPBackend, sdpa_kernel
from app.routers.database_service import Db_helper
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     AutoProcessor,
#     BitsAndBytesConfig,
# )
from qwen_vl_utils import process_vision_info

from app.routers import model_management as mm

from fastapi import APIRouter


from concurrent.futures import ThreadPoolExecutor
import atexit
import functools

from moviepy import VideoFileClip


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = (
        "Identify key data and fillout the given system schema"
    )
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None
    document_key: Optional[str] = None

router = APIRouter()

router = APIRouter(
    prefix="/process",
    tags=["process"],
    responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)


def cleanup_executor():
    executor.shutdown(wait=True)


atexit.register(cleanup_executor)


def check_memory(device=mm.get_torch_device):
    print("Device Loaded: ", device)

    total_mem = mm.get_total_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU has {total_mem} GBs")

    free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
    print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
    return (free_mem_gb, total_mem)


@router.post("/process/")
async def process_video(request_body: BaseProcessor):  
    loop = asyncio.get_running_loop()
    torch.cuda.empty_cache()
    gc.collect()
    check_memory()

    inference_task = functools.partial(
        model_manager.inference, **request_body.model_dump()
    )
    results = await loop.run_in_executor(executor, inference_task)

    for analysis_data in results:
        analysis_ids = []

        # Store in database
        db_helper = Db_helper()
        analysis_id = db_helper.video_analysis(
            analysis_data, source_path=request_body.source_path
        )

        if analysis_id:
            # analysis_data['id'] = analysis_id
            analysis_ids.append(analysis_id)

    print(results)
    return {
        "status": "success",
        "analysis_ids": analysis_ids,
        "results": results,
    }


class Qwen2_VQA:
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
                return e

        return generated_response

    def load_model(self):
        if self._model_loaded:
            return

        torch.cuda.empty_cache()
        print(gc.collect())
        check_memory()
        torch.manual_seed(42)

        model_id = "qwen/Qwen2.5-VL-3B-Instruct"
        self.model_checkpoint = os.path.join(
            "models/prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        # MARK: Precision
        self.dtype = (
            torch.float16
            if mm.should_use_fp16(self.device)
            else torch.bfloat16 if mm.should_use_bf16(self.device) else torch.float32
        )
        print(f'>>> Selected DType: {self.dtype}')

        if self.dtype != torch.float16:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Get architecture details if CUDA is available
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            capability = torch.cuda.get_device_capability(device_index)
            print(f'>>> CUDA Device: {device_name}')
            print(f'>>> Compute Capability: {capability[0]}.{capability[1]}')

            # Classify architecture
            major = capability[0]
            if major >= 8:
                print(">>> Architecture: Ampere or newer (FlashAttention-compatible)")
            elif major == 7:
                print(">>> Architecture: Turing or Volta (not compatible with FlashAttention)")
            else:
                print(">>> Architecture: Older (not compatible)")

        else:
            print(">>> CUDA not available")
    
        print(f'>>> Selected Device: {self.device}')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=self.dtype,
            # attn_implementation="flash_attention_2",
            device_map=self.device,
            quantization_config=quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            min_pixels = 256*28*28,
            max_pixels = 1280*28*28,
        )

        self._model_loaded = True

    def process_batch(self, start, end, segment, i):
        video = VideoFileClip(segment)
        segment_path = os.path.join('segments', f"batch_{i+1}.mp4")

        segment = video.subclipped(start, end)
        segment = segment.with_fps(5)
        segment.write_videofile(
            segment_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"audio/temp-audio-{i}.m4a",
            remove_temp=True, # REMOVE
            logger=None
        )

        segment.close()
        video.close()
        return segment_path

    def inference(
        self,
        system_prompt,
        max_new_tokens,
        source_path=None,
    ):
        results = []
        batches = []
        segments = []
        batch = 2

        os.makedirs('segments', exist_ok=True)
        os.makedirs('audio', exist_ok=True)  # Ensure audio temp dir exists

        start_time = time.time() # timer
        
        try:
            video = VideoFileClip(source_path)
            video_duration = video.duration
            video.close() 

            # MARK: Batching
            for start in range(0, int(video_duration), batch):
                end = min(start + batch, video_duration)
                segments.append((start, end))

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                print('>>> Processing segments')
                video_to_batches = {
                    executor.submit(self.process_batch, 
                                    start, end, source_path, idx):
                                    idx for idx, (start, end) in enumerate(segments)
                }

            for future in concurrent.futures.as_completed(video_to_batches):
                batch = future.result()
                if batch:
                    batches.append(batch)

            elapsed_time = time.time() - start_time # timer logged
            print(f"Batching completed in {elapsed_time:.2f} seconds")

            # MARK: Inference
            if not self._model_loaded:
                print('>>> Loading model into memory')
                self.load_model()

            batches.sort(key=lambda filename: int(filename.split('_')[-1].split('.')[0]))

            sequence_no = 0
            for batch in batches:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": """You are an expert visual analysis system.
                            Analyze the video and structure a concise JSON response structured as follows.

                            Frame activity - A concise description of is happening within the video.
                            Objects detected - A list of objects identified within a close proximity.
                            Cars detected - A list of JSON objects with the following properties: Car license plate(if visible), Color and Model.
                            People detected - A list of JSON objects with the following properties: Estimated Height, Age, Race, Emotional state, and proximity
                            Scene sentiment - Either 'neutral', 'dangerous' or 'unknown'.
                            ID cards detected - A list of JSON objects with the following properties: Surname, Names, Sex, Nationality, Identity Number, Date of Birth, Country of Birth, Status""",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": batch},
                                {"type": "text", "text": system_prompt},
                            ],
                        },
                    ]

                    print('>>> Preparation for inference')
                    # Preparation for inference
                    system_prompts = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)

                    check_memory(self.device)

                    print('Preparing inputs ....')
                    inputs = self.processor(
                        text=system_prompts,
                        videos=video_inputs,
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)
                    
                    # Free memory before generating
                    torch.cuda.empty_cache()
                    gc.collect()

                    print('>>> Inference: Generation of the output')
                    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]

                    generated_response = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    sequence_no += 1
                    generated_response = self.process_generated_response(generated_response[0], sequence_no)

                    finished_in = time.time() - start_time
                    result = {
                        **generated_response,
                        'sequence_no': sequence_no,
                        "finished_in": round(finished_in, 3)
                    }
                    print(json.dumps(result, indent=2))
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing batch {batch}: {str(e)}")
                finally:
                    # Delete batch file immediately after processing
                    try:
                        if os.path.exists(batch):
                            os.remove(batch)
                            print(f"Deleted processed batch: {batch}")
                    except Exception as cleanup_err:
                        print(f"Failed to clean up batch file {batch}: {str(cleanup_err)}")
                    
                    # Free memory after each batch
                    torch.cuda.empty_cache()
                    gc.collect()

            # Final cleanup of any remaining files (belt and suspenders)
            try:
                for dir_path in ['segments', 'audio']:
                    if os.path.exists(dir_path):
                        for f in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, f)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                print(f"Cleaned up remaining file: {file_path}")
            except Exception as final_err:
                print(f"Error in final cleanup: {str(final_err)}")
                
            return results
        except Exception as e:
            print(f"Fatal error in inference: {str(e)}")
            # Attempt cleanup even on catastrophic failure
            try:
                for dir_path in ['segments', 'audio']:
                    if os.path.exists(dir_path):
                        for f in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, f)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
            except:
                pass
            raise

