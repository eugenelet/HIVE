#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=ablate
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=56

module purge
# module load hpc-x
module load hpcx
module load nvidia-hpc
module load nvhpc-hpcx-cuda12
module load anaconda
module load slurm

# conda init bash
source /cm/shared/apps/anaconda/2024.02/etc/profile.d/conda.sh

# Active the conda environment
conda activate llava_next_mod

export OMP_NUM_THREADS=16
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export HF_TOKEN=""
export CUTLASS_PATH="/home/user/jeffrey/cutlass"
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=""
export ACCELERATE_DYNAMO_USE_DYNAMIC=true
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export WANDB_MODE=offline

export TMPDIR="/dev/shm/torchrun_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

LLM_VERSION="facebook/MobileLLM-600M"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
if [[ "$LLM_VERSION" == *MobileLLM* ]]; then
    LLM_MODEL_NAME_OR_PATH="./checkpoints/${LLM_VERSION_CLEAN}"
else
    LLM_MODEL_NAME_OR_PATH="$LLM_VERSION"
fi
VISION_MODEL_VERSION="google/siglip-large-patch16-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Detect number of nodes and GPUs
export NNODES=${SLURM_NNODES:-1}
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS=$((NUM_GPUS_PER_NODE * NNODES))

# Master node and port setup
export RANK=${SLURM_PROCID:-0}
export WORLD_SIZE=$((NNODES * NUM_GPUS_PER_NODE))

# Get master node
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

NUM_WORKERS=$(( (SLURM_CPUS_PER_TASK - 4 * NUM_GPUS_PER_NODE) / NUM_GPUS_PER_NODE ))

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Total nodes: $NNODES"
echo "Total GPUs: $NUM_GPUS"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "World size: $WORLD_SIZE"

echo "Number of workers: $NUM_WORKERS"

wandb login

KEYWORD="SA_florence_siglip"

## Grad_norm for PD3M

STAGE2_GRAD_NORM=1.0
STAGE3_GRAD_NORM=1.0


# STAGE2_GRAD_NORM=0.6
# STAGE3_GRAD_NORM_ARRAY=(0.6 0.55 0.45 0.5 0.6)
# STAGE3_GRAD_NORM=0.8

STAGE1=1
STAGE2=1
STAGE3=1
PRETRAIN=1
FINETUNE=1
EVAL=1
CLASSIFICATION=0

set -e


### Stage 1

export WANDB_PROJECT="LLaVA_next_pretrain_siglip"
PROMPT_VERSION=plain

STAGE1_RUN_NAME="Stage1_${KEYWORD}"
echo "STAGE1_RUN_NAME: ${STAGE1_RUN_NAME}"

if [ ${STAGE1} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_mem.py \
        --deepspeed scripts/zero1.json \
        --mm_vision_select_feature patch \
        --model_name_or_path ${LLM_MODEL_NAME_OR_PATH} \
        --cache_dir ./checkpoints/${LLM_VERSION_CLEAN} \
        --version ${PROMPT_VERSION} \
        --data_path "./playground/data/CC3M/CC3M_florence_filtered.json" \
        --image_folder ./playground/data/CC3M/images.lmdb \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_tunable_parts="mm_mlp_adapter" \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir ./checkpoints_out/Stage1/${STAGE1_RUN_NAME} \
        --num_train_epochs 0.2 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $((8 / NUM_GPUS)) \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_only_model True \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 1e-5 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine_with_min_lr" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_persistent_workers True \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name $STAGE1_RUN_NAME \
        --torch_compile True \
        --torch_compile_backend inductor \
        --attn_implementation sdpa \
        --optim adamw_torch_fused \
        --seed 40

else
    echo "Stage 1 is skipped."
fi

### Stage 2

export WANDB_PROJECT="LLaVA_next_finetune_siglip"
PROMPT_VERSION=plain

STAGE2_RUN_NAME="Stage2_${KEYWORD}"
echo "STAGE2_RUN_NAME: ${STAGE2_RUN_NAME}"

if [ ${STAGE2} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_mem.py \
        --deepspeed scripts/zero1.json \
        --mm_vision_select_feature patch \
        --pretrain_other_parameters ./checkpoints_out/Stage1/${STAGE1_RUN_NAME}/checkpoint-last/other_parameters.bin \
        --model_name_or_path ${LLM_MODEL_NAME_OR_PATH} \
        --cache_dir ./checkpoints/${LLM_VERSION_CLEAN} \
        --version ${PROMPT_VERSION} \
        --data_path "./playground/data/CC3M/CC3M_florence_filtered.json" \
        --image_folder ./playground/data/CC3M/images.lmdb \
        --mm_tunable_parts="mm_mlp_adapter,first_layer_language_model" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        --pretrain_mm_mlp_adapter ./checkpoints_out/Stage1/${STAGE1_RUN_NAME}/checkpoint-last/mm_projector.bin \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio square \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./checkpoints_out/Stage2/${STAGE2_RUN_NAME} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $((32 / NUM_GPUS)) \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_only_model True \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 1e-5 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine_with_min_lr" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_persistent_workers True \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name $STAGE2_RUN_NAME \
        --torch_compile True \
        --torch_compile_backend inductor \
        --attn_implementation sdpa \
        --optim adamw_torch_fused \
        --adam_beta2 0.95 \
        --max_grad_norm ${STAGE2_GRAD_NORM} \
        --seed 41 

else
    echo "Stage 2 is skipped."
fi

### Stage 3

export WANDB_PROJECT="LLaVA_next_finetune_siglip"
PROMPT_VERSION=plain

STAGE3_RUN_NAME="Stage3_${KEYWORD}"
echo "STAGE3_RUN_NAME: ${STAGE3_RUN_NAME}"

if [ ${STAGE3} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_mem.py \
        --deepspeed scripts/zero1.json \
        --mm_vision_select_feature patch \
        --pretrain_other_parameters ./checkpoints_out/Stage2/${STAGE2_RUN_NAME}/checkpoint-last/other_parameters.bin \
        --model_name_or_path ${LLM_MODEL_NAME_OR_PATH} \
        --cache_dir ./checkpoints/${LLM_VERSION_CLEAN} \
        --version ${PROMPT_VERSION} \
        --data_path "./playground/data/CC3M/CC3M_florence_filtered.json" \
        --image_folder ./playground/data/CC3M/images.lmdb \
        --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,first_layer_language_model" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        --pretrain_mm_mlp_adapter ./checkpoints_out/Stage2/${STAGE2_RUN_NAME}/checkpoint-last/mm_projector.bin \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio square \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./checkpoints_out/Stage3/${STAGE3_RUN_NAME} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $((64 / NUM_GPUS)) \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_only_model True \
        --save_total_limit 1 \
        --learning_rate 5e-6 \
        --weight_decay 1e-5 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_persistent_workers True \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name $STAGE3_RUN_NAME \
        --torch_compile True \
        --torch_compile_backend inductor \
        --attn_implementation sdpa \
        --optim adamw_torch_fused \
        --adam_beta2 0.95 \
        --max_grad_norm ${STAGE3_GRAD_NORM} \
        --seed 42

else
    echo "Stage 3 is skipped."
fi

### Stage 4

export WANDB_PROJECT="LLaVA_next_pretrain"
PROMPT_VERSION=plain

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
PRETRAIN_RUN_NAME="Pretrain_Qwen2_${KEYWORD}"
echo "PRETRAIN_RUN_NAME: ${PRETRAIN_RUN_NAME}"

if [ ${PRETRAIN} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_mem.py \
        --deepspeed scripts/zero1.json \
        --model_name_or_path ${LLM_VERSION} \
        --cache_dir ./checkpoints/${LLM_VERSION_CLEAN} \
        --version ${PROMPT_VERSION} \
        --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
        --image_folder ./playground/data/LLaVA-Pretrain/images.lmdb \
        --vision_tower ${VISION_MODEL_VERSION} \
        --pretrain_vision_tower ./checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \
        --mm_tunable_parts="mm_mlp_adapter" \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --mm_vision_select_feature patch \
        --bf16 True \
        --output_dir ./checkpoints_out/pretrain/${PRETRAIN_RUN_NAME} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $((4 / NUM_GPUS)) \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_only_model True \
        --save_total_limit 1 \
        --save_only_model True \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_persistent_workers True \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name $PRETRAIN_RUN_NAME \
        --torch_compile True \
        --torch_compile_backend inductor \
        --attn_implementation flash_attention_2 \
        --optim adamw_torch_fused 

else
    echo "Pretrain is skipped."
fi

### Stage 5

export WANDB_PROJECT="LLaVA_next_finetune"
PROMPT_VERSION="qwen_2"

FINETUNE_RUN_NAME="Finetune_Qwen2_${KEYWORD}"
echo "FINETUNE_RUN_NAME: ${FINETUNE_RUN_NAME}"

if [ ${FINETUNE} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_mem.py \
        --deepspeed scripts/zero1.json \
        --model_name_or_path ${LLM_VERSION} \
        --cache_dir ./checkpoints/${LLM_VERSION_CLEAN} \
        --version ${PROMPT_VERSION} \
        --data_path ./playground/data/LLaVA-NeXT-Data/llava_next_data.json \
        --image_folder ./playground/data/LLaVA-NeXT-Data/images.lmdb \
        --pretrain_mm_mlp_adapter ./checkpoints_out/pretrain/${PRETRAIN_RUN_NAME}/checkpoint-last/mm_projector.bin \
        --mm_tunable_parts "mm_mlp_adapter,lm_head,mm_language_model" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --pretrain_vision_tower ./checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --mm_vision_select_feature patch \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "./checkpoints_out/finetune/${FINETUNE_RUN_NAME}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $((4 / NUM_GPUS)) \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_only_model True \
        --save_total_limit 1 \
        --save_only_model True \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers ${NUM_WORKERS} \
        --dataloader_persistent_workers True \
        --lazy_preprocess True \
        --report_to wandb \
        --dataloader_drop_last True \
        --run_name $FINETUNE_RUN_NAME \
        --torch_compile True \
        --torch_compile_backend inductor \
        --attn_implementation flash_attention_2 \
        --optim adamw_torch_fused 

else
    echo "Finetune is skipped."
fi

### evlaluation

if [ ${EVAL} -eq 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        -m lmms_eval \
        --model llava_paidge \
        --model_args "model_path=./checkpoints_out/finetune/${FINETUNE_RUN_NAME}/checkpoint-last,conv_template=${PROMPT_VERSION},attn_implementation=flash_attention_2" \
        --tasks mme,scienceqa,gqa,ok_vqa,textvqa_val,pope \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/

else
    echo "Evaluation is skipped."
fi

DATASET_NAME="ILSVRC/imagenet-1k"
DATASET="${DATASET_NAME#*/}"

if [ ${CLASSIFICATION} -eq 1 ]; then

    for i in {1..3}; do

        CLASSIFICATION_RUN_NAME="Classification_siglip_${DATASET}_${KEYWORD}_${i}"
        echo "CLASSIFICATION_RUN_NAME: ${CLASSIFICATION_RUN_NAME}"

        torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            llava_next/train/train_classification.py \
            --mm_vision_tower_parameters checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \
            --dataset_name ${DATASET_NAME} \
            --data_path ./playground/classification/ \
            --version ${PROMPT_VERSION} \
            --vision_tower ${VISION_MODEL_VERSION} \
            --bf16 True \
            --output_dir ./checkpoints_out/lit/${CLASSIFICATION_RUN_NAME} \
            --num_train_epochs 1 \
            --per_device_train_batch_size $((512 / NUM_GPUS)) \
            --per_device_eval_batch_size $((512 / NUM_GPUS)) \
            --gradient_accumulation_steps 1 \
            --eval_strategy "epoch" \
            --eval_steps 1 \
            --save_strategy "no" \
            --learning_rate 2e-4 \
            --weight_decay 0.01 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_strategy "no" \
            --tf32 True \
            --gradient_checkpointing False \
            --dataloader_num_workers ${NUM_WORKERS} \
            --lazy_preprocess False \
            --report_to wandb \
            --run_name $CLASSIFICATION_RUN_NAME \
            --torch_compile True \
            --torch_compile_backend inductor \
            --attn_implementation flash_attention_2 \
            --optim adamw_torch_fused \
            --max_grad_norm 3.0 \
            --ddp_find_unused_parameters False

    done

else
    echo "Classification is skipped."
fi