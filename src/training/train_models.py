"""
train_models.py

Multilingual Phishing Detection Model Training
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = BASE_DIR / "datasets" / "processed" / "cleaned_multilingual.csv"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATASET)

print(df.head())

print("\nDataset Shape:", df.shape)

# =====================================================
# Clean Dataset
# =====================================================

print("\nCleaning Dataset...")

# Keep only required columns
df = df[["clean_text", "label"]]

# Remove missing values
df = df.dropna(subset=["clean_text", "label"])

# Convert to string
df["clean_text"] = df["clean_text"].astype(str)

# Remove empty rows
df = df[df["clean_text"].str.strip() != ""]

# Remove duplicates
df = df.drop_duplicates(subset=["clean_text"])

df = df.reset_index(drop=True)

print("\nDataset Shape After Cleaning:", df.shape)

print("\nMissing Values")

print(df.isnull().sum())

# =====================================================
# Features
# =====================================================

X = df["clean_text"]

encoder = LabelEncoder()

y = encoder.fit_transform(df["label"])

# Save Label Encoder
joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.pkl"
)

# =====================================================
# TF-IDF
# =====================================================

print("\nVectorizing Text...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

# =====================================================
# Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

}

results = []

best_accuracy = 0

best_model = None

best_name = ""

best_report = ""

# =====================================================
# Train Models
# =====================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({

        "Model": name,

        "Accuracy": round(accuracy,4),

        "Precision": round(precision,4),

        "Recall": round(recall,4),

        "F1 Score": round(f1,4)

    })

    report = classification_report(
        y_test,
        prediction,
        target_names=encoder.classes_
    )

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name

        best_report = report

# =====================================================
# Save Best Model
# =====================================================

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    RESULT_DIR / "model_results.csv",
    index=False
)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(best_report)

# =====================================================
# Final Summary
# =====================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"\nBest Model : {best_name}")
print(f"Best Accuracy : {best_accuracy:.4f}")

print("\nModel Comparison")

print(results_df)

print("\nSaved Models")

for file in MODEL_DIR.iterdir():
    print("✔", file.name)

print("\nSaved Results")

for file in RESULT_DIR.iterdir():
    print("✔", file.name)