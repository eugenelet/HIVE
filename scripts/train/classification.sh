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

# KEYWORD="baseline_linear"
# STAGE3_RUN_NAME="Stage3_roadmap_ablation_600M_8layers_square_CC3M_alt_preLN_mlp_trainfullLLM_gradNorm1_siglip"
# KEYWORD="alt"
# STAGE3_RUN_NAME="Stage3_connection_ablation_600M_8layers_square_CC3M_florence_preLN_mlp_trainMMLayer_siglip"
# KEYWORD="synthetic_linear"
# STAGE3_RUN_NAME=Stage3_SA_ablation_600M_SA_square_CC3M_florence_mlp_trainfirstLLM_WD1e-5_siglip
# KEYWORD="SA_synthetic"
STAGE3_RUN_NAME=Stage3_SA_ablation_600M_SA_square_CC3M_alt_mlp_trainfullLLM_WD1e-5_siglip
KEYWORD="SA_alt"

# --mm_vision_tower_parameters checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \

set -e


DATASET_NAME="ILSVRC/imagenet-1k"
DATASET="${DATASET_NAME#*/}"

for i in {1..3}; do

    CLASSIFICATION_RUN_NAME="Classification_siglip_${DATASET}_${KEYWORD}_${i}"
    echo "CLASSIFICATION_RUN_NAME: ${CLASSIFICATION_RUN_NAME}"

    torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        llava_next/train/train_classification.py \
        --mm_vision_tower_parameters checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \
        --dataset_name ${DATASET_NAME} \
        --data_path ./playground/classification/ \
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
        --learning_rate 1e-3 \
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

DATASETS=("cifar10" "cifar100" "zh-plus/tiny-imagenet" "ilee0022/Caltech-256" "ethz/food101" "tanganke/stanford_cars" "timm/oxford-iiit-pet")

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET="${DATASET_NAME#*/}"

    for i in {1..3}; do
        CLASSIFICATION_RUN_NAME="Classification_siglip_${DATASET}_${KEYWORD}_${i}"
        echo "CLASSIFICATION_RUN_NAME: ${CLASSIFICATION_RUN_NAME}"

        torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            llava_next/train/train_classification.py \
            --mm_vision_tower_parameters checkpoints_out/Stage3/${STAGE3_RUN_NAME}/checkpoint-last/vision_tower.bin \
            --dataset_name ${DATASET_NAME} \
            --data_path ./playground/classification/ \
            --vision_tower ${VISION_MODEL_VERSION} \
            --bf16 True \
            --output_dir ./checkpoints_out/lit/${CLASSIFICATION_RUN_NAME} \
            --num_train_epochs 5 \
            --per_device_train_batch_size $((512 / NUM_GPUS)) \
            --per_device_eval_batch_size $((512 / NUM_GPUS)) \
            --gradient_accumulation_steps 1 \
            --eval_strategy "epoch" \
            --eval_steps 5 \
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
done