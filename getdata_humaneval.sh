CUDA_VISIBLE_DEVICES="0" accelerate launch  get_data.py \
  --model /data2/zhouyz/M3Bench/models/ckpts/Qwen/Qwen2.5-Coder-1.5B-Instruct/ \
  --max_length_generation 1024 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 200 \
  --batch_size 10 \
  --allow_code_execution