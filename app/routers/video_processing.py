import os
from typing import Optional
from pydantic import BaseModel
import torch
import numpy as np
import cv2

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class BaseProcessor(BaseModel):
    processor: str
    model: Optional[str] = "Qwen2.5-VL-3B-Instruct"
    system_prompt: Optional[str] = 'Perform a detailed analysis of the given data'
    keep_model_loaded: Optional[bool] = False
    temperature: Optional[float] = 0.5
    max_new_tokens: Optional[int] = 512
    min_pixels: Optional[int] = 200704
    max_pixels: Optional[int] = 1003520 
    seed: Optional[int] = -1
    quantization: Optional[str] = "8bit"
    source_path: Optional[str] = None
    image_path: Optional[str] = None
    attention: Optional[str] = "eager"

router = APIRouter()

router = APIRouter(
    prefix="/process",
    tags=["process"],
    responses={404: {"description": "Not found"}},
)
@router.post("/process/", tags=["process"])
async def process_video(request_body: BaseProcessor):
    qwen_vqa = Qwen2_VQA()

    generated_data = qwen_vqa.inference(
        text=request_body.system_prompt,
        model=request_body.model,
        keep_model_loaded=request_body.keep_model_loaded,
        temperature=request_body.temperature,
        max_new_tokens=request_body.max_new_tokens,
        min_pixels=request_body.min_pixels,
        max_pixels=request_body.max_pixels,
        seed=request_body.seed,
        quantization=request_body.quantization,
        source_path=request_body.source_path,
        image_path=request_body.image_path,
        attention=request_body.attention,
    )

    print(generated_data)
    return [{"results":generated_data }]

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
        print(mm.get_torch_device())

    def inference(
        self,
        text,
        model,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
        seed,
        quantization,
        source_path=None,
        image_path=None,
        attention="eager",
    ):
        if seed != -1:
            torch.manual_seed(seed)
        
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            "models/prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
        )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
            use_fast=False
        )

        with torch.no_grad():
            print(source_path)
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
                            {"type": "text", "text": text},
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
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                use_fast=True,
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",            
            )
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature
            )

            if not keep_model_loaded:
                del self.processor  # release processor memory
                del self.model  # release model memory
                self.processor = None  # set processor to None
                self.model = None  # set model to None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release GPU memory
                    torch.cuda.ipc_collect()

            # split_result = str(result).split("\n")
            # joined_result_str = "\n".join(split_result)

            return result

# MARK: Gemma

def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )

class Gemma3ModelLoader:
    def load_model(self, model_id, load_local_model, *args, **kwargs):
        device = mm.get_torch_device()
        
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
            
        # Load the correct model architecture based on the model_id
        model = (
            Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
            .eval()
            .to(device)
        )
        processor = AutoProcessor.from_pretrained(model_path)
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
