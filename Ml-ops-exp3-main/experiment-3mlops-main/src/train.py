import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/cleaned_kidney_disease.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()


# ==========================================
# 3. REMOVE ID COLUMN
# ==========================================

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ==========================================
# 4. TARGET COLUMN
# ==========================================

target = "classification"

df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Convert target labels
df[target] = df[target].replace({
    "ckd": 1,
    "ckd\t": 1,
    "notckd": 0,
    "not ckd": 0
})


# Remove invalid target values
df = df[df[target].isin([0, 1])]


# ==========================================
# 5. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[target])
y = df[target].astype(int)


# ==========================================
# 6. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))


# ==========================================
# 7. PREPROCESSING
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
# 8. RANDOM FOREST V1
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


# ==========================================
# 9. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ==========================================
# 10. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 11. TRAIN MODEL
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 12. PREDICTION
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# 13. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("RANDOM FOREST MODEL V1")
print("================================")
print("Number of trees:", 100)
print("Maximum depth:", 5)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    pipeline,
    "models/random_forest_kidney_v1.pkl"
)

print("\nModel saved successfully:")
print("models/random_forest_kidney_v1.pkl")