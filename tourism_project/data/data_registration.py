import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Sagarikak17"
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"

api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(repo_id=DATASET_REPO, repo_type="dataset", token=os.environ["HF_TOKEN"], exist_ok=True)

api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
print("Dataset uploaded to:", f"https://huggingface.co/datasets/{DATASET_REPO}")
