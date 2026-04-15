"""
TFDS/RLDS dataset builder for the BridgeV2 5K subset.

Reads from our extracted real_5k_1 format (parquet + individual mp4 per episode)
and produces an RLDS dataset compatible with Octo's data pipeline.
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = "BridgeV2 5K episode subset for Octo training from scratch."

_DEFAULT_DATA_DIR = "/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/data/bridgeV2/real_5k_1"


class Builder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for BridgeV2 5K subset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial 5K subset."}

    def __init__(self, *args, data_source_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source_dir = data_source_dir or _DEFAULT_DATA_DIR

    def _info(self) -> tfds.core.DatasetInfo:
        """Define the RLDS dataset schema matching original bridge_dataset."""
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
                        dtype=tf.int32, doc="Original BridgeV2 episode index."
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
        """Yield (key, episode) pairs."""
        data_dir = self._data_source_dir

        # Load frame-level data
        parquet_path = os.path.join(data_dir, "data", "episodes_0000.parquet")
        df = pd.read_parquet(parquet_path)

        video_dir = os.path.join(data_dir, "videos", "observation.images.primary")

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_frames = df[df["episode_index"] == ep_idx].sort_values("frame_index")

            # Read video frames
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

            n_parquet = len(ep_frames)
            n_video = len(frames)

            if n_video == 0:
                print(f"[WARN] Empty video for episode {ep_idx}, skipping")
                continue

            n_steps = min(n_parquet, n_video)

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

            yield int(ep_idx), episode
