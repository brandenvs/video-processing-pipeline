import csv
import datetime
import json
import os
import atexit
import shutil
import time
import uuid
import subprocess
import torch
import gc
import asyncio
import urllib
import tempfile
import functools
from pathlib import Path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig # type: ignore
from qwen_vl_utils import process_vision_info

from fastapi import APIRouter
from app.routers import model_management as mm

from typing import Any, Optional
from pydantic import BaseModel

from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FieldDefinition(BaseModel):
    label: str
    field_type: str
    description: str
    required: bool = False

class BaseProcessor(BaseModel):
  system_prompt: Optional[str] = "Identify key data and fillout the given system schema"
  max_tokens: Optional[int] = 512
  model_id: Optional[str] = "qwen/Qwen2.5-VL-3B-Instruct"
  source_path: str
  input_schema: Any

router = APIRouter()

router = APIRouter(
  prefix="/process",
  tags=["process"],
  responses={404: {"description": "Not found"}},
)

executor = ThreadPoolExecutor(max_workers=4)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def cleanup_executor():
  executor.shutdown(wait=True)

atexit.register(cleanup_executor)

def check_memory(device=mm.get_torch_device):
  total_mem = mm.get_total_memory(device) / (1024 * 1024 * 1024)
  print(f"GPU has {total_mem} GBs")

  free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
  print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
  return (free_mem_gb, total_mem)

@router.post("/video/")
async def process_video(request_body: BaseProcessor):
  print('request_body: ', request_body)
  loop = asyncio.get_running_loop()
  torch.cuda.empty_cache()
  gc.collect()
  check_memory()

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  print(request_body)
  filename = f'/home/ubuntu/adp-video-pipeline/source_data/{uuid.uuid4()}video.mp4'
  result = urllib.request.urlretrieve(request_body.source_path, filename)

  if (result):
    scene_detection = functools.partial(batch_scene_detect, filename)
    scene_detection_response = await loop.run_in_executor(executor, scene_detection)
  
  infer_obj = {
    'model_id': request_body.model_id,
    'system_prompt': request_body.system_prompt,
    'max_tokens': request_body.max_tokens,
    'input_schema': request_body.input_schema,
    'segments': scene_detection_response['segments'],
    'stats_scene': scene_detection_response['stats_scene']
  }
  print('Inference Object', infer_obj)
  process_video = functools.partial(model_manager.inference_helper, **infer_obj)
  processed_video_response, finalStructuredOutput = await loop.run_in_executor(executor, process_video)
  
  # Cleanup for next video ...
  [os.remove(os.path.join('segments', f)) for f in os.listdir('segments') if os.path.isfile(os.path.join('segments', f))]
  [os.remove(os.path.join('audio', f)) for f in os.listdir('audio') if os.path.isfile(os.path.join('audio', f))]
 
  structured_outputs= {
    item['sequence_no']: item 
    for item in processed_video_response 
    if item is not None
  }

  return {
    "status": "success",
    "scene_detection_response": scene_detection_response,
    "structured_output": structured_outputs,
    "finalStructuredOutput": finalStructuredOutput
  }
 
def get_mean_content_val(stats_file: str) -> float:
  content_vals = []
  
  with open(f'segments/{stats_file}', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      content_vals.append(float(row['content_val']))
  
  return sum(content_vals) / len(content_vals)

def get_content_val(stats_file: str) -> dict:
  content_vals = []
  
  with open(stats_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      content_vals.append(float(row['content_val']))
  
  content_vals.sort()
  n = len(content_vals)
  
  mean_val = sum(content_vals) / n
  median_val = content_vals[n // 2] if n % 2 == 1 else (content_vals[n // 2 - 1] + content_vals[n // 2]) / 2
  variance = sum((x - mean_val) ** 2 for x in content_vals) / n
  
  return {
    'count': n,
    'min': min(content_vals),
    'max': max(content_vals),
    'mean': mean_val,
    'median': median_val,
    'std': variance ** 0.5,
    'p25': content_vals[n // 4],
    'p75': content_vals[3 * n // 4],
    'p90': content_vals[int(0.9 * n)]
  }

def batch_scene_detect(video):
  stats_scene = []
    
  # TODO Black box - splits
  response = subprocess.run([
      'scenedetect', '--config', 'scenedetect.cfg',
      '-i', video,
      '--output', 'segments',
      'detect-content', '-t', '30',
      'split-video', '--filename', '$SCENE_NUMBER', '--copy'
  ], check=True, capture_output=True, text=True)
  segments = [f for f in os.listdir('segments') if f.endswith('.mp4')]

  for segment in segments:
    stats = f'{segment}_stats.csv'
    scene = f'{segment}_scene.csv'
    stats_obj = {
        'stats': stats,
        'scene': scene
    }
    stats_scene.append(stats_obj)        

    # TODO Black box - Gets stats on segment
    response = subprocess.run([
      'scenedetect', '--config', 'scenedetect.cfg',
      '-i', f'segments/{segment}',
      '--output', 'segments',
      '--stats', stats,
      'detect-content', '-t', '30',
      'list-scenes', '-f', scene,
      'split-video', '--filename', '$SCENE_NUMBER', '--copy'
    ], check=True, capture_output=True, text=True)
    print(response.stdout)
    
    mean_content_val = get_mean_content_val(stats)

    content_threshold = 15
    if mean_content_val < content_threshold:
      _ = segments.pop(segments.index(segment))

  response = {  
    'segments': segments,
    'stats_scene': stats_scene,
    "mean_content_val": round(mean_content_val, 3)
  }
  return response

class Qwen2_VQA:
  def __init__(self):
    self.model_checkpoint = None
    self.processor = None
    self.model = None
    self.device = mm.get_torch_device()
    self.dtype = None
    self._model_loaded = False

  def process_generated_response(self, generated_response: str, sequence_no: int):
    print(f'GENERATED RESPONSE FOR BATCH: {sequence_no}', generated_response)
    if generated_response.startswith("```json"):
      try:
        lines = generated_response.strip().splitlines()
        json_content = "\n".join(lines[1:-1])

        json_response = json.loads(json_content)
        spread_response = {**json_response, **{"sequence_no": sequence_no}}
        return spread_response

      except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {
          "error": f"JSON parsing error: {str(e)}",
          "sequence_no": sequence_no
        } 

    try:
      json_response = json.loads(generated_response)
      spread_response = {**json_response, **{"sequence_no": sequence_no}}
      return spread_response
    except json.JSONDecodeError as e:
      print(f"JSON decode error: {e}")
      return {
        "error": f"JSON parsing error: {str(e)}",
        "sequence_no": sequence_no,
        "raw_sample": generated_response
      }

  def load_model(self, model_id: str):
    if self._model_loaded:
      return

    torch.cuda.empty_cache()
    print('Total Garbage Collected', gc.collect())
    check_memory()
    torch.manual_seed(42)

    self.model_checkpoint = os.path.join(
        "models/prompt_generator", os.path.basename(model_id)
    )

    if not os.path.exists(self.model_checkpoint):
      from huggingface_hub import snapshot_download

      snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

    # MARK: Precision
    if mm.should_use_fp16(self.device):
      self.dtype = torch.float16
    elif mm.should_use_fp16(self.device):
      self.dtype = torch.float16
    else:
      torch.float32

    # print(f'>>> Selected DType: {self.dtype}')

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
      # device_name = torch.cuda.get_device_name(device_index)
      capability = torch.cuda.get_device_capability(device_index)
      # print(f'>>> CUDA Device: {device_name}')
      # print(f'>>> Compute Capability: {capability[0]}.{capability[1]}')

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
      attn_implementation="flash_attention_2",
      device_map=self.device,
      quantization_config=quantization_config,
    )

    self.processor = AutoProcessor.from_pretrained(
      self.model_checkpoint,
      min_pixels = 256*28*28,
      max_pixels = 1280*28*28,
    )

    self._model_loaded = True

  def process_final_schema(self, input_schema, responses):
    data = [item for item in responses if item is not None]
    input_data = json.dumps(data, indent=2)
    
    messages = [
      {
          "role": "system",
          "content": f"""You are an expert data analyst specializing in video analysis aggregation. Your task is to synthesize multiple video sequence analyses into a single, coherent final report.

ANALYSIS APPROACH:
1. Review all sequence data for patterns and consistency
2. Resolve conflicts by prioritizing most detailed/confident observations
3. Aggregate temporal information across sequences
4. Generate comprehensive summary maintaining data integrity

OUTPUT REQUIREMENTS:
- Must conform exactly to this JSON schema: {input_schema}
- Use "Yes"/"No" for boolean fields, never "Not applicable" 
- Consolidate contradictory evidence logically
- Prioritize positive detections over negative ones when evidence exists
- Include confidence indicators where multiple sequences conflict

CONFLICT RESOLUTION RULES:
- If any sequence detects weapons/suspects/plates, mark as detected
- Combine witness and civilian counts from all sequences
- Use most detailed suspect descriptions available
- Aggregate all unique license plate observations"""
      },
      {
          "role": "user", 
          "content": f"""Analyze and synthesize the following video sequence data into the final schema:

SEQUENCE DATA ({len(data)} sequences analyzed):
{input_data}

Requirements:
- Synthesize information across all {len(data)} video sequences
- Resolve any conflicting observations logically
- Generate a comprehensive final analysis
- Return ONLY valid JSON matching the required schema

Generate the final consolidated analysis:"""
      }
  ]
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
    inputs =  inputs.to(self.device)

    print('>>> Inference: Generation of the output')
    try:
      outputs = self.model.generate(**inputs, max_new_tokens=2000)

      generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
      ]

      generated_response = self.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
      )
      print('>> PRE PROCESSED generated_response ', generated_response)
      generated_response = self.process_generated_response(generated_response[0], 999)
      print('>>> POST PROCESSED generated_response ', generated_response)
      print(json.dumps(generated_response, indent=2))

    except Exception as ex:
      print(ex)
    return generated_response
      
  def inference_helper(self, model_id: str, system_prompt: str, max_tokens: int, input_schema: str, segments: list, stats_scene: list):
    responses = []
        
    if not self._model_loaded:
      print('>>> Loading model into memory')
      self.load_model(model_id)

    segments.sort(key=lambda x: int(x[:3]), reverse=False)
    stats_scene.sort(key=lambda x: int(x['stats'].split('.')[0]))     # segments.sort(reverse=True)

    for seq, segment in enumerate(segments):
      segment_path = os.path.join('segments', segment)

        # TODO Black box (strips audio) / no alternative really ...
      with tempfile.TemporaryDirectory(dir='tmp') as temp_dir:
        temp_path = os.path.join(temp_dir, segment)
        
        subprocess.run([
          'ffmpeg', '-y', '-i', segment_path,
          '-r', '5', '-c:v', 'libx264', '-crf', '23',
          '-an',
          temp_path,
          '-vn', '-c:a', 'aac',
          f'audio/{segment}.aac'
        ], capture_output=True, text=True)                
        shutil.move(temp_path, segment_path)

      with open(f'segments/{stats_scene[seq]["scene"]}', 'r') as f:
        reader = csv.DictReader(f)
        # NOTE THIS DOES NOT WORK !!!
        for idx, row in enumerate(reader):
          if idx > 0:
            for val in row.values():
              if isinstance(val, list):
                batch_length = float(val[-1])
      responses.append(self.inference(segment_path, system_prompt, max_tokens, input_schema, seq, batch_length))
    generated_response = self.process_final_schema(input_schema, responses)
    return (responses, generated_response)

  def inference(self, segment_path: str, system_prompt: str, max_tokens: int, input_schema: str, seq: int, batch_length: str):
    start_time = time.time() # timer
    
    fields_info = {}
    for field_name, field_def in input_schema.items():
      fields_info[field_name] = {
        "label": field_def['label'],
        "value": "[Generated Value Based on the video analysis]",
      }
    schema_str = json.dumps(fields_info, indent=2)

    messages = [
      {
        "role": "system",
        "content": f"You are an expert visual analysis system. Analyze the video and generate structured output using the following JSON schema: {schema_str}",
      },
      {
        "role": "user",
        "content": [
          {"type": "video", "video": segment_path},
          {"type": "text", "text": system_prompt},
        ],
      },
    ]

    print('>>> Preparation for inference')
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
    inputs =  inputs.to(self.device)

    print('>>> Inference: Generation of the output')
    try:
      outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

      generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
      ]

      generated_response = self.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
      )
      generated_response = self.process_generated_response(generated_response[0], seq)

      finished_in = time.time() - start_time
      print(json.dumps(generated_response, indent=2))
      response = {
        **generated_response,
        "batch_length": batch_length,
        "finished_in": round(finished_in, 3)
      }
      return response
    except Exception as ex:
      print(ex)


model_manager = Qwen2_VQA()
