import pandas as pd, os, mlflow, mlflow.sklearn, joblib
from huggingface_hub import hf_hub_download, HfApi, create_repo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HF_USERNAME = "Sagarikak17"   # <-- change this
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO = f"{HF_USERNAME}/tourism-model"

train_path = hf_hub_download(DATASET_REPO, "train.csv", repo_type="dataset", token=os.environ["HF_TOKEN"])
test_path = hf_hub_download(DATASET_REPO, "test.csv", repo_type="dataset", token=os.environ["HF_TOKEN"])
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X_train, y_train = train.drop(columns=["ProdTaken"]), train["ProdTaken"]
X_test, y_test = test.drop(columns=["ProdTaken"]), test["ProdTaken"]

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

mlflow.set_experiment("tourism-package-prediction")
best_score, best_pipeline, best_name = -1, None, None

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")
        print(name, metrics)
        if metrics["f1"] > best_score:
            best_score, best_pipeline, best_name = metrics["f1"], pipe, name

print(f"Best model: {best_name} (F1={best_score:.3f})")

os.makedirs("tourism_project/model_building", exist_ok=True)
model_path = "tourism_project/model_building/best_model.joblib"
joblib.dump(best_pipeline, model_path)

api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(repo_id=MODEL_REPO, repo_type="model", token=os.environ["HF_TOKEN"], exist_ok=True)
api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib", repo_id=MODEL_REPO, repo_type="model")
print("Model registered at:", f"https://huggingface.co/{MODEL_REPO}")
