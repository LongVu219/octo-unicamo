"""Upload sim10k finetune checkpoints (steps 2000, 4000) to HF."""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "cpp219/simpler_finetune_sim_only"
SRC = "/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo/checkpoints/sim10k_finetune_debug/octo_sim10k_debug/finetune/experiment_20260422_080316"

api = HfApi()
create_repo(REPO_ID, repo_type="model", exist_ok=True)

# Upload supporting files needed to load the model.
for name in ["config.json", "dataset_statistics.json", "example_batch.msgpack", "finetune_config.json"]:
    path = os.path.join(SRC, name)
    if os.path.isfile(path):
        print(f"Uploading {name}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=REPO_ID,
            repo_type="model",
        )

# Upload the two checkpoint step folders.
for step in ["2000", "4000"]:
    folder = os.path.join(SRC, step)
    print(f"Uploading step {step}/ ...")
    api.upload_folder(
        folder_path=folder,
        path_in_repo=step,
        repo_id=REPO_ID,
        repo_type="model",
    )

print(f"Done: https://huggingface.co/{REPO_ID}")
