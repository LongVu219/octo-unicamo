# conda activate octo
# python convert_to_rlds_10k.py

# Fix cuDNN algorithm selection issue on H100 (compute capability 9.0)
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

python scripts/finetune.py \
    --config scripts/configs/finetune_bridge10k_config.py:full,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-small \
    --config.save_dir=checkpoints/bridge10k_finetune \
    --config.wandb.project=octo_bridge10k \
    --config.wandb.group=finetune
