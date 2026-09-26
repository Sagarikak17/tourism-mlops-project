import pandas as pd, os
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split

HF_USERNAME = "Sagarikak17"   # <-- change this
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
api = HfApi(token=os.environ["HF_TOKEN"])

path = hf_hub_download(repo_id=DATASET_REPO, filename="tourism.csv", repo_type="dataset", token=os.environ["HF_TOKEN"])
df = pd.read_csv(path)

df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

df = df.dropna(subset=["ProdTaken"])
num_cols = df.select_dtypes(include="number").columns.drop("ProdTaken")
cat_cols = df.select_dtypes(include="object").columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
os.makedirs("tourism_project/data", exist_ok=True)
train.to_csv("tourism_project/data/train.csv", index=False)
test.to_csv("tourism_project/data/test.csv", index=False)

for f in ["train.csv", "test.csv"]:
    api.upload_file(path_or_fileobj=f"tourism_project/data/{f}", path_in_repo=f, repo_id=DATASET_REPO, repo_type="dataset")
print("train/test uploaded.")
