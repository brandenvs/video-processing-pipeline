# import asyncio
# import atexit
# from io import BytesIO
# import json
# import os
# from pathlib import Path
# from typing import Optional
# from urllib.request import urlopen
# from app.routers.database_service import Db_helper
# import librosa
# from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
# from app.routers import model_management as mm
# import torch
# import gc
# from moviepy import VideoFileClip
# from pydub import AudioSegment
# from pathlib import Path
# import math
# import functools

# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
# # model = Qwen2AudioForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
# # )

# # conversation = [
# #     {"role": "system", "content": "You are a helpful assistant."},
# #     {
# #         "role": "user",
# #         "content": [
# #             {
# #                 "type": "audio",
# #                 "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
# #             },
# #             {"type": "text", "text": "What's that sound?"},
# #         ],
# #     },
# #     {"role": "assistant", "content": "It is the sound of glass shattering."},
# #     {
# #         "role": "user",
# #         "content": [
# #             {"type": "text", "text": "What can you do when you hear that?"},
# #         ],
# #     },
# #     {
# #         "role": "assistant",
# #         "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property.",
# #     },
# #     {
# #         "role": "user",
# #         "content": [
# #             {
# #                 "type": "audio",
# #                 "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
# #             },
# #             {"type": "text", "text": "What does the person say?"},
# #         ],
# #     },
# # ]
# # text = processor.apply_chat_template(
# #     conversation, add_generation_prompt=True, tokenize=False
# # )
# # audios = []
# # for message in conversation:
# #     if isinstance(message["content"], list):
# #         for ele in message["content"]:
# #             if ele["type"] == "audio":
# #                 audios.append(
# #                     librosa.load(
# #                         BytesIO(urlopen(ele["audio_url"]).read()),
# #                         sr=processor.feature_extractor.sampling_rate,
# #                     )[0]
# #                 )

# # inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
# # inputs.input_ids = inputs.input_ids.to("cuda")

# # generate_ids = model.generate(**inputs, max_length=256)
# # generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

# # response = processor.batch_decode(
# #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# # )[0]


# class Data(BaseModel):
#     system_prompt: Optional[str] = (
#         "Identify key data and fillout the given system schema"
#     )
#     source_path: Optional[str] = None


# executor = ThreadPoolExecutor(max_workers=4)


# def cleanup_executor():
#     executor.shutdown(wait=True)


# atexit.register(cleanup_executor)


# async def process_video(audio_file):
#     loop = asyncio.get_running_loop()
#     torch.cuda.empty_cache()
#     gc.collect()
#     check_memory()

#     data = Data()
#     data.source_path = audio_file
#     inference_task = functools.partial(model_manager.inference, **data.model_dump())
#     results = await loop.run_in_executor(executor, inference_task)

#     for analysis_data in results:
#         analysis_ids = []

#         # Store in database
#         db_helper = Db_helper()
#         analysis_id = db_helper.audio_analysis(analysis_data)

#         if analysis_id:
#             analysis_ids.append(analysis_id)

#     return {
#         "status": "success",
#         "analysis_ids": analysis_ids,
#         "results": results,
#     }


# def check_memory(device=mm.get_torch_device()):
#     print("Device Loaded: ", device)

#     total_mem = mm.get_total_memory() / (1024 * 1024 * 1024)
#     print(f"GPU has {total_mem} GBs")

#     free_mem_gb = mm.get_free_memory(device) / (1024 * 1024 * 1024)
#     print(f"GPU memory checked: {free_mem_gb:.2f}GB available.")
#     return (free_mem_gb, total_mem)


# class Qwen2_Audio:
#     def __init__(self):
#         self.model_checkpoint
#         self.processor = None
#         self.model = None
#         self.device = mm.get_torch_device()
#         self.dtype = None
#         self._model_loaded = False

#     def process_generated_response(self, generated_response: str, sequence_no: int):
#         if generated_response.startswith("```json"):
#             try:
#                 lines = generated_response.strip().splitlines()
#                 json_content = "\n".join(lines[1:-1])

#                 json_response = json.loads(json_content)
#                 spread_response = {**json_response, **{"sequence_no": sequence_no}}
#                 return spread_response

#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {e}")
#                 return e

#         return generated_response

#     def load_model(self):
#         if self._model_loaded:
#             return

#         torch.cuda.empty_cache()
#         print(gc.collect())
#         check_memory()
#         torch.manual_seed(42)

#         model_id = "Qwen/Qwen2-Audio-7B-Instruct"
#         self.model_checkpoint = os.path.join(
#             "models/prompt_generator", os.path.basename(model_id)
#         )

#         if not os.path.exists(self.model_checkpoint):
#             from huggingface_hub import snapshot_download

#             snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint)

#         self.dtype = (
#             torch.float16
#             if mm.should_use_fp16(self.device)
#             else torch.bfloat16 if mm.should_use_bf16(self.device) else torch.float32
#         )

#         model = Qwen2AudioForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2-Audio-7B-Instruct", device_map=self.device
#         )
#         self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
#         self._model_loaded = True

#     def inference(
#         self,
#         system_prompt,
#         source_path=None,
#     ):
#         audio = AudioSegment.from_file(source_path)

#         results = []

#         batch_duration = 2
#         batch_duration_ms = batch_duration * 1000
#         audio_duration_ms = len(audio)

#         # Audio batching
#         batches = []
#         for start_ms in range(0, audio_duration_ms, batch_duration_ms):
#             end_ms = min(start_ms + batch_duration_ms, audio_duration_ms)
#             batch = audio[start_ms:end_ms]
#             batches.append(batch)

#         data_dir = Path("./audio")
#         data_dir.mkdir(parents=True, exist_ok=True)

#         # Export each batch
#         for idx, batch in enumerate(batches):
#             batch_filename = data_dir / f"batch_{idx+1}.wav"
#             batch.export(batch_filename, format="wav")

#         if not self._model_loaded:
#             self.load_model()

#         sequence_no = 0
#         for filename in os.listdir(data_dir):
#             file_path = os.path.join(data_dir, filename)  # Audio file path
#             messages = [
#                 {
#                     "role": "assistant",
#                     "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property.",
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "audio",
#                             "audio_url": file_path,
#                         },
#                         {"type": "text", "text": system_prompt},
#                     ],
#                 },
#             ]

#             # Preparation for inference
#             system_prompts = self.processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#             check_memory(self.device)

#             audios = []
#             for message in messages:
#                 if isinstance(message["content"], list):
#                     for ele in message["content"]:
#                         if ele["type"] == "audio":
#                             audios.append(
#                                 librosa.load(
#                                     BytesIO(urlopen(ele["audio_url"]).read()),
#                                     sr=processor.feature_extractor.sampling_rate,
#                                 )[0]
#                             )

#             inputs = processor(
#                 text=system_prompts, audios=audios, return_tensors="pt", padding=True
#             )
#             inputs.input_ids = inputs.input_ids.to("cuda")

#             generate_ids = model.generate(**inputs, max_length=256)
#             generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

#             generated_response = processor.batch_decode(
#                 generate_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False,
#             )

#             sequence_no += 1
#             result = self.process_generated_response(generated_response[0], sequence_no)
#             results.append(result)
#         return results


# model_manager = Qwen2_Audio()
