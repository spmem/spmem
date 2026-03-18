#!/bin/bash
set -x

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30

GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
PORT=29503
# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("8000")
WARMUP_STEPS=100
CHECKPOINT_STEPS=250
TRAIN_BATCH_SIZE=1
COND_FRAMES=5
# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_8gpu.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.

# training dataset parameters
DATA_ROOT="datasets/train_data_example"
MODEL_PATH="ckpt/spmem_ckpt"
OUTPUT_PATH="outputs_train"

CAPTION_COLUMN="datasets/train_data_example_config/prompts.txt"
VIDEO_COLUMN="datasets/train_data_example_config/videos.txt"
TRACKING_COLUMN="datasets/train_data_example_config/trackings.txt"


cross_attn_interval=14

VALIDATION_DIRS=(
    "examples/000000000011.0_005"
)

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_PATH}/spmem_train"
        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE training/cogvideox_image_to_video_sft_ref.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --trainable_modules transformer_blocks_copy initial_combine_linear combine_linears perceiver_cross_attention ref_patch_embed \  
          --is_train_cross \
          --cross_attn_interval $cross_attn_interval \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --tracking_column $TRACKING_COLUMN \
          --cond_frames $COND_FRAMES \
          --num_tracking_blocks 18 \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_dir "${VALIDATION_DIRS[@]}" \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 25 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --resume_from_checkpoint \"latest\" \
          --nccl_timeout 18000"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
