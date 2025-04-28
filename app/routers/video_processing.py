import os
from typing import Optional
from pydantic import BaseModel
import torch
import numpy as np
import gc
import cv2
import asyncio
import logging

from torchvision.transforms import ToPILImage
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)
from PIL import Image
from qwen_vl_utils import process_vision_info

from app.routers import model_management as mm

from fastapi import APIRouter, Depends

# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

class BaseProcessor(BaseModel):
    system_prompt: Optional[str] = 'Perform a detailed analysis of the given data'
    temperature: Optional[float] = 0.5
    max_new_tokens: Optional[int] = 512
    source_path: Optional[str] = None
    image_path: Optional[str] = None

router = APIRouter()

router = APIRouter(
    prefix="/process",
    tags=["process"],
    responses={404: {"description": "Not found"}},
)

from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
import atexit
import functools

# Create executor at app startup
executor = ThreadPoolExecutor(max_workers=4)

# Register cleanup function to ensure executor is shutdown
def cleanup_executor():
    executor.shutdown(wait=True)
    
atexit.register(cleanup_executor)

@router.post("/process/")
async def process_video(request_body: BaseProcessor):
    loop = asyncio.get_running_loop()
    try:
        # Ensure memory is cleared before running inference
        torch.cuda.empty_cache()
        gc.collect()
        
        inference_task = functools.partial(model_manager.inference, **request_body.model_dump())
        generated_data = await loop.run_in_executor(executor, inference_task)
        
        # Clear memory after inference
        torch.cuda.empty_cache()
        gc.collect()
        
        return [{"results": generated_data}]
    except Exception as e:
        # Clean memory on error
        torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/process/")
# async def process_video(request_body: BaseProcessor):
#     loop = asyncio.get_running_loop()
#     result = await loop.run_in_executor(None, qwen_vqa.inference, ...)

#     qwen_vqa = Qwen2_VQA()

#     generated_data = qwen_vqa.inference(

#     )

#     print(generated_data)
#     return [{"results":generated_data }]

# MARK: Qwen

class Qwen2_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = mm.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self._model_loaded = False
        self._last_used = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def load_model(self, quantization="4bit", attention='eager'):
        # Skip loading if already loaded
        if self._model_loaded:
            return
            
        # Clean memory before loading model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check if we have enough memory to load the model
        free_mem_gb = mm.get_free_memory(self.device) / (1024 * 1024 * 1024)
        if free_mem_gb < 6:  # Need at least 6GB for 4-bit quantized model
            logging.warning(f"Low GPU memory detected: {free_mem_gb:.2f}GB available. This may cause issues.")
            
        torch.manual_seed(-1)
        model_id = "qwen/Qwen2.5-VL-3B-Instruct"
        self.model_checkpoint = os.path.join("models/prompt_generator", os.path.basename(model_id))
        
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

        # Use more conservative image size limits to reduce memory usage
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint, 
            min_pixels=153600,  # Reduced: 480x320
            max_pixels=409600   # Reduced: 640x640
        )
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        # Clean memory before loading model
        torch.cuda.empty_cache()
        gc.collect()

        # Get available GPU memory dynamically
        free_mem = mm.get_free_memory(self.device)
        # Set a reasonable limit (80% of available memory)
        memory_limit = int(free_mem * 0.8 / (1024 * 1024 * 1024))
        # Use at least 4GB but no more than available memory minus 2GB safety margin
        memory_limit = max(4, min(memory_limit, int(free_mem / (1024 * 1024 * 1024)) - 2))
        
        logging.info(f"Loading model with {memory_limit}GiB memory limit (available: {free_mem/(1024*1024*1024):.2f}GiB)")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
            max_memory={0: f"{memory_limit}GiB"},
            low_cpu_mem_usage=True,
        )
        
        # Mark model as loaded
        self._model_loaded = True

    def inference(
        self,
        system_prompt,
        temperature,
        max_new_tokens,
        source_path=None,
        image_path=None,
    ):
        try:
            # Ensure model is loaded
            if not self._model_loaded:
                self.load_model()
                
            # Clean memory before inference
            torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                if source_path:
                    messages = [
                        {
                            "role": "system",
                            "content":
                            "You are a detailed visual analysis system. Analyze the provided visual content and output the following information in structured JSON format:\n",

                            "frame_description": 
                            "Detailed description of all visible objects, locations, lighting, and activity","license_plates": "All visible car license plates exactly as they appear (or partially if obscured)\n",

                            "scene_sentiment": 
                            "Assessment of whether the environment appears peaceful, neutral, or dangerous, with justification\n",

                            "people_nearby": 
                            "Description of all people in focus, including estimated age range, clothing, behavior, and interactions\n",

                            "risk_analysis": "Any signs of risk, conflict, or abnormal activity"
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": source_path}, 
                                {"type": "text", "text": system_prompt},
                            ],
                        },
                    ]
                elif image_path:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{image_path}"},
                                {"type": "text", "text": system_prompt},
                            ],
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": system_prompt},
                            ],
                        }
                    ]

                # Preparation for inference
                system_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Process vision info in a memory-efficient way
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Clear memory after processing vision info
                torch.cuda.empty_cache()
                gc.collect()
                
                # Check memory before processor step
                free_mem = mm.get_free_memory(self.device) / (1024 * 1024 * 1024)
                logging.info(f"Memory before processor: {free_mem:.2f}GB")
                
                try:
                    # Process with smaller batch size if needed
                    max_frames = 8  # Limit number of video frames processed at once
                    inputs = self.processor(
                        text=[system_prompt],
                        images=image_inputs,
                        videos=video_inputs[:max_frames] if video_inputs and len(video_inputs) > max_frames else video_inputs,
                        padding=True,
                        return_tensors="pt",            
                    )
                except RuntimeError as e:
                    # If we still run out of memory, try again with even smaller batch
                    if "CUDA out of memory" in str(e):
                        logging.warning("CUDA OOM during processing, reducing batch size further")
                        torch.cuda.empty_cache()
                        gc.collect()
                        max_frames = 4  # Even smaller batch
                        inputs = self.processor(
                            text=[system_prompt],
                            images=image_inputs,
                            videos=video_inputs[:max_frames] if video_inputs and len(video_inputs) > max_frames else video_inputs,
                            padding=True,
                            return_tensors="pt",            
                        )
                    else:
                        raise

                # Move inputs to device efficiently
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                # Clear memory before generation
                torch.cuda.empty_cache()
                gc.collect()

                # Check memory before generation
                free_mem = mm.get_free_memory(self.device) / (1024 * 1024 * 1024)
                logging.info(f"Memory before generation: {free_mem:.2f}GB")
                
                try:
                    # Use more memory-efficient generation settings
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=min(max_new_tokens, 256),  # Limit max tokens if needed 
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        use_cache=True,
                        repetition_penalty=1.2  # Helps with memory efficiency
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        # Last resort - try CPU fallback
                        logging.warning("CUDA OOM during generation, attempting CPU fallback")
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Move inputs to CPU
                        cpu_inputs = {}
                        for key in inputs:
                            if isinstance(inputs[key], torch.Tensor):
                                cpu_inputs[key] = inputs[key].cpu()
                            else:
                                cpu_inputs[key] = inputs[key]
                        
                        # Move model to CPU temporarily
                        self.model = self.model.cpu()
                        
                        # Generate on CPU with minimal parameters
                        generated_ids = self.model.generate(
                            **cpu_inputs, 
                            max_new_tokens=128,  # Much shorter on CPU
                            temperature=0,       # No sampling on CPU
                            do_sample=False,
                            use_cache=True
                        )
                        
                        # Move model back to GPU after generation
                        self.model = self.model.to(self.device)
                    else:
                        raise
                
                # Store input_ids before deleting inputs
                input_ids = inputs.input_ids.clone()
                
                # Free memory from inputs after generation
                del inputs
                torch.cuda.empty_cache()
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
                # Free memory from input_ids
                del input_ids
                
                # Free more memory
                del generated_ids
                torch.cuda.empty_cache()
                
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # Clear everything from memory
                del generated_ids_trimmed
                torch.cuda.empty_cache()
                gc.collect()
                
                return result
                
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            return f'Error: {str(e)}'

# Initialize the model manager but don't load model yet
model_manager = Qwen2_VQA()

# MARK: Gemma

def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )

class Gemma3ModelLoader:
    def __init__(self):
        self._loaded_models = {}
        
    def load_model(self, model_id, load_local_model, *args, **kwargs):
        # If we've already loaded this model, return it
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
            
        device = mm.get_torch_device()
        
        # Clean memory before loading model
        torch.cuda.empty_cache()
        gc.collect()
        
        if load_local_model:
            from huggingface_hub import snapshot_download
            
            model_checkpoint = os.path.join(
                "models/", os.path.basename(model_id)
            )

            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
            )
            model_path = model_checkpoint
        else:
            # Use the provided model_id directly
            model_path = model_id
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
            
        # Load the correct model architecture based on the model_id
        model = (
            Gemma3ForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto",
                quantization_config=quantization_config,
                max_memory={0: "10GiB"},
                low_cpu_mem_usage=True
            )
            .eval()
        )
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Cache the loaded model and processor
        self._loaded_models[model_id] = (model, processor)
        
        return (model, processor)

class ApplyGemma3:
    def apply_gemma3(
        self, model, processor, prompt, max_new_tokens, image=None, video=None
    ):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }
        ]

        if video is not None:
            # Assume 'video' is a path to the video file
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"file://{video}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        elif image is not None:
            image_pil = tensor2pil(image)  # Convert tensor to PIL image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            )

        # Process the messages
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return (decoded,)

# if __name__ == "__main__":
#     qwen_vqa = Qwen2_VQA()


#     # Example parameters
#     text = "Describe the scene in the image."
#     model = "Qwen2.5-VL-3B-Instruct"
#     keep_model_loaded = False
#     temperature = 0.4
#     max_new_tokens = 1024
#     min_pixels = 200704      # 256 * 784
#     max_pixels = 1003520     # 1280 * 784
#     seed = -1
#     quantization = "4bit"  # or "8bit"

#     # source_path = "X:\\stadprin-active\\SP Technical\\video-analysis-mvp\\adp-video-pipeline\\Cape_Town_Metro_officers_and_car.jpg"
#     source_path = None
#     video_path = 'X:\\stadprin-active\\SP Technical\\video-analysis-mvp\\adp-video-pipeline\\test_video_short.mp4'
#     image = None  # torch tensor with shape [1, H, W, 3] in range [0, 1]
#     attention = "eager"

#     qwen_vqa.inference(
#         text=text,
#         model=model,
#         keep_model_loaded=keep_model_loaded,
#         temperature=temperature,
#         max_new_tokens=max_new_tokens,
#         min_pixels=min_pixels,
#         max_pixels=max_pixels,
#         seed=seed,
#         quantization=quantization,
#         source_path=source_path,
#         image=image,
#         attention=attention,
#     )

# if __name__ == "__main__":
#     # Initialize the model loader and application classes
#     model_loader = Gemma3ModelLoader()
#     apply_gemma3_instance = ApplyGemma3()

#     # Define the model ID and whether to load a local model
#     model_id = "google/gemma-3-1b-it"
#     load_local_model = True  # Set to True if loading a local model

#     # Load the model and processor
#     model, processor = model_loader.load_model(model_id, load_local_model)

#     # Define the prompt and video path
#     prompt = "Describe the events in this video."
#     video_path = "X:\\stadprin-active\\SP Technical\\video-analysis-mvp\\adp-video-pipeline\\Cape_Town_Metro_officers_and_car.jpg"

#     # Ensure the video file exists
#     if not os.path.isfile(video_path):
#         raise FileNotFoundError(f"Video file not found at {video_path}")

#     # Define the maximum number of new tokens to generate
#     max_new_tokens = 512  # Adjust as needed

#     # Apply the model to the video input
#     result = apply_gemma3_instance.apply_gemma3(
#         model=model,
#         processor=processor,
#         prompt=prompt,
#         max_new_tokens=max_new_tokens,
#         video=video_path,
#     )

#     # Output the result
#     print("Generated Output:")
#     print(result[0])
