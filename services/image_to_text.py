import os
import torch
import numpy as np
import gc
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

# Comfy
import comfy.model_management as mm

# Qwen
from qwen_vl_utils import process_vision_info

# Set PyTorch memory management options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def tensor2pil(image):
    """Convert tensor to PIL image"""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


class BaseVLModel:
    """Base class for vision-language models"""
    def __init__(self):
        self.device = mm.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.processor = None
        self.model = None
        self.model_checkpoint = None

    def _get_quantization_config(self, quantization):
        """Get quantization config based on specified bit precision"""
        if quantization == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None

    def _download_model(self, model_id, model_base_path="models/prompt_generator"):
        """Download model from Hugging Face if not available locally"""
        model_checkpoint = os.path.join(model_base_path, os.path.basename(model_id))
        
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
            )
        
        return model_checkpoint

    def _prepare_image(self, image, seed):
        """Prepare image for processing"""
        if image is not None:
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = f"/var/tmp/temp_image_{seed}.png"
            pil_image.save(temp_path)
            return temp_path
        return None

    def _unload_model(self):
        """Unload model and free up memory"""
        if self.processor:
            del self.processor
            self.processor = None
        
        if self.model:
            del self.model
            self.model = None
        
        # Force Python garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Additional memory cleanup for CUDA
            with torch.cuda.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


class QwenVLModel(BaseVLModel):
    """Qwen VL model implementation"""
    def load_model(self, model_id, min_pixels, max_pixels, quantization=None, attention="eager"):
        """Load Qwen VL model and processor"""
        self.model_checkpoint = self._download_model(model_id)
        
        # Clean memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Initialize processor if not already done
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                use_fast=True  # Force fast tokenizer to avoid warning
            )
        
        # Initialize model if not already done
        if self.model is None:
            quantization_config = self._get_quantization_config(quantization)
            
            # If not already using quantization and we're getting low on CUDA memory,
            # automatically enable 8-bit quantization
            if quantization is None and torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB free
                    print("Low CUDA memory detected, automatically enabling 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
                # Enable memory efficient attention if available
                use_flash_attention_2="flash_attention_2" in attention,
                # Use low CPU memory usage
                low_cpu_mem_usage=True,
            )

    def generate(self, text, messages, max_new_tokens, temperature):
        """Generate text from input messages"""
        with torch.no_grad():
            # Prepare inputs
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )
            
            return result


class Gemma3VLModel(BaseVLModel):
    """Gemma 3 VL model implementation"""
    def load_model(self, model_id, quantization=None, attention="eager"):
        """Load Gemma 3 model and processor"""
        self.model_checkpoint = self._download_model(model_id, "models")
        
        # Clean memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Initialize model and processor
        quantization_config = self._get_quantization_config(quantization)
        
        # If not already using quantization and we're getting low on CUDA memory,
        # automatically enable 8-bit quantization
        if quantization is None and torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB free
                print("Low CUDA memory detected, automatically enabling 8-bit quantization")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                use_fast=True  # Force fast tokenizer to avoid warning
            )
        
        if self.model is None:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
                # Use low CPU memory usage
                low_cpu_mem_usage=True,
            )

    def generate(self, text, messages, max_new_tokens, temperature):
        """Generate text from input messages"""
        with torch.inference_mode():
            # Process messages
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Set safe generation parameters
            # If temperature is 0 or very small, use greedy decoding instead of sampling
            do_sample = False
            if temperature > 0.01:  # Only use sampling with meaningful temperature values
                do_sample = True
            else:
                temperature = 1.0  # Set to 1.0 when not using temperature to avoid CUDA errors
            
            # Generate output
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.95,  # Add top_p sampling for stability
            )
            generation = generation[0][input_len:]
            
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return [decoded]


class ImageToTextService:
    """Unified service to handle image/video to text generation with multiple models"""
    def __init__(self):
        self.qwen_model = QwenVLModel()
        self.gemma_model = Gemma3VLModel()

    def _extract_frames(self, video_path):
        """Extract key frames from a video as a set of images
        Returns a list of image paths instead of a video path"""
        # Only process video if it exists
        if not video_path or not os.path.exists(video_path):
            return None
            
        try:
            import cv2
            from pathlib import Path
            
            # Create output directory for frames
            video_filename = Path(video_path).stem
            frames_dir = f"/var/tmp/{video_filename}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}.")
                return None
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract only a small number of key frames to avoid CUDA OOM
            max_dimension = 384  # Reduced max dimension
            max_frames = 4       # Very limited number of frames
            
            # Calculate step for frame extraction
            frame_step = max(1, frame_count // max_frames)
            
            # Calculate new dimensions for memory efficiency
            new_width, new_height = width, height
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
            
            print(f"Extracting frames from video: {video_path}")
            print(f"Original: {width}x{height}, {frame_count} frames")
            print(f"Target: {new_width}x{new_height}, up to {max_frames} frames")
            
            # Extract key frames
            frame_paths = []
            frame_idx = 0
            
            while True and len(frame_paths) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_step == 0:
                    # Resize frame
                    if new_width != width or new_height != height:
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Save frame as JPEG (more compatible with models than PNG)
                    frame_path = os.path.join(frames_dir, f"frame_{len(frame_paths):02d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_paths.append(frame_path)
                
                frame_idx += 1
            
            # Release video capture
            cap.release()
            
            if len(frame_paths) > 0:
                print(f"Extracted {len(frame_paths)} key frames to {frames_dir}")
                return frame_paths
            else:
                print("No frames were extracted")
                return None
                
        except Exception as e:
            import traceback
            print(f"Error extracting frames: {str(e)}")
            traceback.print_exc()
            return None
            
    def _process_video(self, video_path):
        """Process video to reduce memory usage or downsample if needed
        Returns the processed video path or the original path"""
        # Only process video if it exists
        if not video_path or not os.path.exists(video_path):
            return video_path
            
        try:
            import cv2
            from pathlib import Path
            
            # Create output path for processed video
            video_filename = Path(video_path).stem
            processed_path = f"/var/tmp/{video_filename}_processed.mp4"
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Using original video.")
                return video_path
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Use very aggressive downsampling to avoid CUDA OOM
            max_dimension = 256  # Drastically reduced from 640 to 256
            max_frames = 5       # Drastically reduced from 300 to 5
            target_fps = 1       # Only 1 FPS to minimize memory usage
            
            # Calculate new dimensions
            new_width, new_height = width, height
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
            
            # Always process videos for memory efficiency
            # Calculate frame step to extract limited frames
            frame_step = max(1, int(frame_count / max_frames))
            
            print(f"Processing video: {video_path}")
            print(f"Original: {width}x{height}, {frame_count} frames at {fps} FPS")
            print(f"Target: {new_width}x{new_height}, {min(max_frames, frame_count)} frames at {target_fps} FPS")
            
            # Set up video writer with reduced dimensions and fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_path, fourcc, target_fps, (new_width, new_height))
            
            # Process only selected frames
            frame_idx = 0
            frames_written = 0
            
            while frames_written < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_step == 0:
                    # Resize frame
                    if new_width != width or new_height != height:
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Write frame
                    out.write(frame)
                    frames_written += 1
                
                frame_idx += 1
                
                # Check if we've processed enough frames
                if frames_written >= max_frames:
                    break
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"Created heavily optimized video: {processed_path}")
            print(f"Final: {new_width}x{new_height}, {frames_written} frames at {target_fps} FPS")
            
            return processed_path
                
        except Exception as e:
            import traceback
            print(f"Error processing video: {str(e)}")
            traceback.print_exc()
            print("Using original video")
            return video_path

    def _prepare_messages(self, text, source_path=None, video_path=None, temp_path=None, model_type="qwen"):
        """Prepare messages for the model based on input type"""
        system_content = """You are a detailed visual analysis system. Analyze the provided visual content and output the following information in structured JSON format:
{
  "frame_description": "Detailed description of all visible objects, locations, lighting, and activity",
  "license_plates": "All visible car license plates exactly as they appear (or partially if obscured)",
  "scene_sentiment": "Assessment of whether the environment appears peaceful, neutral, or dangerous, with justification",
  "people_nearby": "Description of all people in focus, including estimated age range, clothing, behavior, and interactions",
  "risk_analysis": "Any signs of risk, conflict, or abnormal activity"
}

Important: Provide a complete and accurate analysis, focusing on objective details rather than subjective interpretations. Maintain JSON structure in your response.
"""
        
        # Use the user-provided text if available, otherwise use a default prompt
        if not text or text.strip() == "":
            default_prompt = "Perform a detailed visual analysis and output the results in JSON format."
            text = default_prompt
        
        if video_path:
            # Try to extract frames first (preferred method to avoid CUDA OOM)
            frame_paths = self._extract_frames(video_path)
            
            if frame_paths and len(frame_paths) > 0:
                # Use extracted frames as multiple images instead of video
                content_items = [{"type": "text", "text": text}]
                
                # Add each frame as a separate image
                for frame_path in frame_paths:
                    content_items.insert(0, {"type": "image", "image": f"file://{frame_path}"})
                
                # Add frame numbering information to help the model
                frame_info = "The images shown are key frames from a video. Analyze each frame and provide a comprehensive analysis."
                
                messages = [
                    {"role": "system", "content": f"{system_content}\n\n{frame_info}"},
                    {"role": "user", "content": content_items},
                ]
            else:
                # Fall back to video processing if frame extraction failed
                processed_video_path = self._process_video(video_path)
                
                messages = [
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": processed_video_path},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
        elif source_path:
            image_specific_prompt = "Analyze this image and provide the information in the specified JSON format."
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{source_path}"},
                        {"type": "text", "text": text if text else image_specific_prompt},
                    ],
                },
            ]
        elif temp_path:
            image_specific_prompt = "Analyze this image and provide the information in the specified JSON format."
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_path}"},
                        {"type": "text", "text": text if text else image_specific_prompt},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ]
        
        return messages

    def infer(
        self,
        text,
        model,
        keep_model_loaded=False,
        temperature=0.0,  # Default to greedy decoding for stability
        max_new_tokens=512,  # Reduced from 1024 to 512 for memory efficiency
        min_pixels=64512,  # Reduced from 200704 to 64512 (256*252) for memory efficiency
        max_pixels=250880,  # Reduced from 1003520 to 250880 (320*784) for memory efficiency
        seed=-1,
        quantization="8bit",  # Default to 8-bit quantization for videos
        video_path=None,
        source_path=None,
        image=None,
        attention="eager",
    ):
        """Main inference method supporting both Qwen and Gemma models"""
        if seed != -1:
            torch.manual_seed(seed)
        
        # Clean up memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Determine model type and set up model ID
        if model.startswith("gemma-"):
            model_type = "gemma"
            model_id = f"google/{model}"
            active_model = self.gemma_model
        else:
            model_type = "qwen"
            model_id = f"qwen/{model}"
            active_model = self.qwen_model
        
        # Prepare image if available
        temp_path = active_model._prepare_image(image, seed) if image is not None else None
        
        try:
            # Auto-detect if we should use quantization based on available memory
            auto_quantization = quantization
            if auto_quantization is None and torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_memory_gb = free_memory / (1024**3)  # Convert to GB
                
                # If less than 7GB available and working with video, use 8-bit quantization
                if free_memory_gb < 7.0 and (video_path is not None):
                    print(f"Available GPU memory: {free_memory_gb:.2f}GB - Automatically enabling 8-bit quantization")
                    auto_quantization = "8bit"
            
            # Load the appropriate model
            if model_type == "qwen":
                active_model.load_model(
                    model_id=model_id,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    quantization=auto_quantization,
                    attention=attention
                )
            else:  # gemma
                active_model.load_model(
                    model_id=model_id,
                    quantization=auto_quantization,
                    attention=attention
                )
            
            # Prepare messages
            messages = self._prepare_messages(
                text=text,
                source_path=source_path,
                video_path=video_path,
                temp_path=temp_path,
                model_type=model_type
            )
            
            # Generate response
            result = active_model.generate(
                text=text,
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Format the result
            split_result = str(result).split("\n")
            joined_result_str = "\n".join(split_result)
            print(joined_result_str)
            
            # Unload model if not keeping it loaded
            if not keep_model_loaded:
                active_model._unload_model()
                
            # Clean temporary files
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {e}")
            
            # Clean up extracted frame files if they exist
            if video_path:
                frame_paths = self._extract_frames(video_path)
                if frame_paths:
                    for frame_path in frame_paths:
                        if os.path.exists(frame_path):
                            try:
                                os.remove(frame_path)
                            except:
                                pass
                    
                    # Try to remove the frames directory
                    try:
                        import shutil
                        from pathlib import Path
                        frames_dir = str(Path(frame_paths[0]).parent)
                        shutil.rmtree(frames_dir, ignore_errors=True)
                    except:
                        pass
                
            return joined_result_str
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            # Ensure model is unloaded in case of error
            if not keep_model_loaded:
                active_model._unload_model()
            
            # Clean temporary files even on error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            raise


if __name__ == "__main__":
    # Create service instance
    image_to_text_service = ImageToTextService()
    
    import argparse
    
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description="Image/Video to Text Processing Service")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-3B-Instruct", 
                        help="Model to use: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, gemma-3-4b-it, etc.")
    parser.add_argument("--text", type=str, default="Analyze this visual content and provide detailed information in JSON format.", 
                        help="Prompt text to send to the model")
    parser.add_argument("--image", type=str, default=None, 
                        help="Path to input image file")
    parser.add_argument("--video", type=str, default=None, 
                        help="Path to input video file")
    parser.add_argument("--temperature", type=float, default=0.5, 
                        help="Temperature for sampling (0.0 = greedy)")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--quantize", type=str, default="8bit", choices=["none", "4bit", "8bit"], 
                        help="Quantization to use (none, 4bit, 8bit)")
    parser.add_argument("--keep_loaded", action="store_true", 
                        help="Keep model loaded in memory after inference")
    parser.add_argument("--attention", type=str, default="eager", 
                        choices=["eager", "sdpa", "flash_attention_2"], 
                        help="Attention implementation to use")
    
    args = parser.parse_args()
    
    # Set parameters from command line arguments
    text = args.text
    model = args.model
    keep_model_loaded = args.keep_loaded
    temperature = args.temperature
    max_new_tokens = args.max_tokens
    min_pixels = 64512   # 256 * 252 (reduced for memory efficiency)
    max_pixels = 250880  # 320 * 784 (reduced for memory efficiency)
    seed = -1
    quantization = None if args.quantize == "none" else args.quantize
    video_path = args.video
    source_path = args.image
    image = None  # For tensor input, not used in CLI mode
    attention = args.attention
    
    # Print configuration
    print(f"Model: {model}")
    print(f"Input: {'Video' if video_path else 'Image' if source_path else 'Text only'}")
    if video_path:
        print(f"Video path: {video_path}")
    if source_path:
        print(f"Image path: {source_path}")
    print(f"Prompt: {text}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Quantization: {quantization}")
    print(f"Attention: {attention}")
    print("-" * 40)
    
    # Run inference
    try:
        result = image_to_text_service.infer(
            text=text,
            model=model,
            keep_model_loaded=keep_model_loaded,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            seed=seed,
            quantization=quantization,
            video_path=video_path,
            source_path=source_path,
            image=image,
            attention=attention,
        )
        
        print("\nResult:")
        print("-" * 40)
        print(result)
        
    except Exception as e:
        import traceback
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()