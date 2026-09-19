import os
import json
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# LOAD PARAMETERS
# ==========================================

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

rf_params = params["random_forest"]
test_size = params["test_size"]


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/cleaned_kidney_disease.csv")

df.columns = df.columns.str.strip()

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ==========================================
# TARGET
# ==========================================

target = "classification"

df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)

df[target] = df[target].replace({
    "ckd": 1,
    "ckd\t": 1,
    "notckd": 0,
    "not ckd": 0
})

df = df[df[target].isin([0, 1])]

X = df.drop(columns=[target])
y = df[target].astype(int)


# ==========================================
# FEATURE TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# ==========================================
# PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    random_state=rf_params["random_state"]
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=rf_params["random_state"],
    stratify=y
)


# ==========================================
# TRAIN
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# PREDICT
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n================================")
print("DVC RANDOM FOREST EXPERIMENT")
print("================================")

print("Trees:", rf_params["n_estimators"])
print("Max Depth:", rf_params["max_depth"])
print("Random State:", rf_params["random_state"])

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    "models/reproducible_kidney_model.pkl"
)


# ==========================================
# SAVE METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("\nModel saved:")
print("models/reproducible_kidney_model.pkl")

print("\nMetrics saved:")
print("metrics.json")