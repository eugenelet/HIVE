# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
#     --num_processes=8 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mme,gqa,textvqa \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./logs/

python3 -m accelerate.commands.launch \
    --gpu_ids 6,7 \
    --num_processes=2 \
    --main_process_port 29500 \
    -m lmms_eval \
    --model llava_paidge \
    --model_args "model_base=./checkpoints/Qwen_Qwen2-1.5B-Instruct/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8,\
model_path=./checkpoints_out/Stage3/Stage3_llava1.5_qwen2-1.5B_siglipL384_LoRA_llava/checkpoint-last,\
conv_template=qwen_2,\
attn_implementation=sdpa" \
    --tasks mme,gqa,textvqa \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

# model_base=./checkpoints/Qwen_Qwen2-1.5B-Instruct/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8,\