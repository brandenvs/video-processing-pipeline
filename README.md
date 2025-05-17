# ADP Video Pipeline

## TODOs

### Document pipeline

- [x] Integrate technical document processing within the pipeline ([Qwen3-1.7B-GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF))
  - [x] Database integration(use database_service.py and add create table to database_setup.py)

### Audio pipeline

- [ ] Integrate audio to text within the pipeline ([Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct))
  - [ ] Database integration(use database_service.py)
- [ ] Build vectorisers for the table.

### Video pipeline

- [x] Integrate video to text within pipeline ([Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct))
  - [x] Database integration(use database_service.py)
- [ ] Build vectorisers for the table.

---

- [ ] RAG on Db vectors. (reseaching(2025/05/09) - Dylan )


- [ ] Build a frontend dashboard. {
            
- [ ] Set up Connection point, upload point for customer
