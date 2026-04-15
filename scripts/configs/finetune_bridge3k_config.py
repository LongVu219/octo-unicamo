"""
Config for training Octo-small from scratch on BridgeV2 3K subset.

Usage:
    conda activate octo
    cd /prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo

    python scripts/finetune.py \
        --config scripts/configs/finetune_bridge3k_config.py:full,language_conditioned \
        --config.pretrained_path=hf://rail-berkeley/octo-small \
        --config.save_dir=checkpoints/bridge3k_from_scratch \
        --config.wandb.project=octo_bridge3k \
        --config.wandb.group=from_scratch \
        --config.skip_merge_pretrained=True
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Our 3K dataset config
    FINETUNING_KWARGS = {
        "name": "bridge3k_dataset",
        "data_dir": "/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo/data",
        "image_obs_keys": {"primary": "image_0", "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "absolute_action_mask": [False, False, False, False, False, False, True],
        "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge3k_dataset_transform",
    }

    frozen_keys = None  # full training (from scratch)

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=64,               # smaller batch for 3K dataset
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_bridge3k", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        # Flag to skip loading pretrained weights (train from scratch)
        skip_merge_pretrained=True,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=3,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
        ],
    )
    config["frame_transform_threads"] = 16
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
