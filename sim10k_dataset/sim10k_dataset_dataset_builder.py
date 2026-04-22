"""
TFDS/RLDS dataset builder for the synthetic 10K subset (sim_5k_1 + sim_5k_2).

Mirrors bridge10k_dataset but points to the synthetic spoon directories.
Reads parquet + individual mp4 per episode, producing a single RLDS dataset
compatible with Octo's data pipeline.
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = "Synthetic 10K episode subset (sim_5k_1 + sim_5k_2) for Octo training."

_BASE_DIR = "/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/data/bridgeV2"
_DEFAULT_DATA_DIRS = [
    os.path.join(_BASE_DIR, "sim_5k_1"),
    os.path.join(_BASE_DIR, "sim_5k_2"),
]


class Builder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for synthetic 10K subset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial synthetic 10K subset (5K+5K)."}

    def __init__(self, *args, data_source_dirs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source_dirs = data_source_dirs or _DEFAULT_DATA_DIRS

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_0": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=tf.uint8,
                            encoding_format="jpeg",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,), dtype=tf.float32,
                            doc="7x robot state (6 joint/EEF + gripper)",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=tf.float32,
                        doc="7x action (6 joint/EEF delta + gripper)",
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Natural language task description.",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=tf.float32, doc="Reward (always 0 for this dataset)."
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=tf.float32, doc="Discount (always 1)."
                    ),
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_index": tfds.features.Scalar(
                        dtype=tf.int32, doc="Original synthetic episode index."
                    ),
                    "file_path": tfds.features.Text(
                        doc="Source video file path."
                    ),
                }),
            }),
            description=_DESCRIPTION,
        )

    def _split_generators(self, dl_manager):
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yield (key, episode) pairs from all data source directories."""
        global_key = 0

        for data_dir in self._data_source_dirs:
            parquet_path = os.path.join(data_dir, "data", "episodes_0000.parquet")
            df = pd.read_parquet(parquet_path)
            video_dir = os.path.join(data_dir, "videos", "observation.images.primary")

            for ep_idx in sorted(df["episode_index"].unique()):
                ep_frames = df[df["episode_index"] == ep_idx].sort_values("frame_index")

                video_path = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
                if not os.path.isfile(video_path):
                    print(f"[WARN] Missing video: {video_path}, skipping")
                    continue

                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    frames.append(frame)
                cap.release()

                if len(frames) == 0:
                    print(f"[WARN] Empty video for episode {ep_idx}, skipping")
                    continue

                n_steps = min(len(ep_frames), len(frames))

                steps = []
                for i, (_, row) in enumerate(ep_frames.iloc[:n_steps].iterrows()):
                    state = np.array(row["observation.state"], dtype=np.float32)
                    action = np.array(row["action"], dtype=np.float32)
                    lang = str(row.get("language_instruction", ""))

                    steps.append({
                        "observation": {
                            "image_0": frames[i],
                            "state": state,
                        },
                        "action": action,
                        "language_instruction": lang,
                        "reward": 0.0,
                        "discount": 1.0,
                        "is_first": i == 0,
                        "is_last": i == n_steps - 1,
                        "is_terminal": False,
                    })

                episode = {
                    "steps": steps,
                    "episode_metadata": {
                        "episode_index": int(ep_idx),
                        "file_path": video_path,
                    },
                }

                yield global_key, episode
                global_key += 1
