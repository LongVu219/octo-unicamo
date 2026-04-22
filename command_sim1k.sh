# conda activate octo
#
# Before running training, build the TFDS dataset once (CPU only):
#   cd /prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo/sim1k_dataset
#   tfds build --data_dir=/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo/data
#   cd /prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo

# Fix cuDNN algorithm selection issue on H100 (compute capability 9.0)
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_CUDNN_DETERMINISTIC=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
# If this still fails, also clear the JAX compile cache so new flags actually apply:
#   rm -rf /prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/.jax_compilation_cache

# Disable NVLink SHARP (NVLS) multicast — Fabric Manager/NVSwitch on this node
# can't bind NVLS memory (CUDA error 401). NCCL falls back to standard collectives.
export NCCL_NVLS_ENABLE=0

python scripts/finetune.py \
    --config scripts/configs/finetune_sim1k_config.py:full,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-small \
    --config.save_dir=checkpoints/sim1k_finetune_debug \
    --config.wandb.project=octo_sim1k_debug \
    --config.wandb.group=finetune
