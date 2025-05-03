# ADP Video Pipeline

## TODOs

### Document pipeline

- [ ] Integrate technical document processing within the pipeline ([Qwen3-1.7B-GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF))
  - [ ] Database integration(use database_service.py)

### Audio pipeline

- [ ] Integrate audio to text within the pipeline ([Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct))
  - [ ] Database integration(use database_service.py)

### Video pipeline

- [x] Integrate video to text within pipeline ([Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct))
  - [x] Db writes with generated data

---

- [ ] Vectorizers with generated data.
- [ ] RAG on Db vectors.
