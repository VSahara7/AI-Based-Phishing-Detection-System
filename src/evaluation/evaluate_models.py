import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report,
    accuracy_score
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

# ======================================================
# Create Results Folder
# ======================================================

RESULTS = "/Users/user/Documents/Phishing/results"
os.makedirs(RESULTS, exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(
    "/Users/user/Documents/Phishing/datasets/processed/cleaned_multilingual.csv"
)

df = df.dropna(subset=["clean_text", "label"])

# Convert labels into numeric format

label_map = {
    "legitimate": 0,
    "phishing": 1
}

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_map)
)

print(df["label"].unique())

X = df["clean_text"].astype(str)
y = df["label"]

print(df.head())

print("\nDataset Shape:", df.shape)
print(df["label"].unique())
print(df["label"].dtype)
# ======================================================
# Load TF-IDF Vectorizer
# ======================================================

print("\nLoading TF-IDF Vectorizer...")

vectorizer = joblib.load(
    "/Users/user/Documents/Phishing/models/tfidf_vectorizer.pkl"
)

X = vectorizer.transform(X)

print("✓ Vectorizer Loaded")

# ======================================================
# Train Test Split
# ======================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# ======================================================
# Load Best Model
# ======================================================

print("\nLoading Best Model...")

model = joblib.load(
    "/Users/user/Documents/Phishing/models/best_model.pkl"
)

print(f"Model Loaded : {type(model).__name__}")

# ======================================================
# 1. Cross Validation
# ======================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

cv_results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "Accuracy": scores
})

print(cv_results)

print("\nAverage Accuracy :", round(scores.mean(), 4))
print("Standard Deviation:", round(scores.std(), 4))

cv_results.loc[len(cv_results)] = [
    "Average",
    round(scores.mean(), 4)
]

cv_results.to_csv(
    os.path.join(
        RESULTS,
        "cross_validation_results.csv"
    ),
    index=False
)

print("\n✓ Saved cross_validation_results.csv")
# ======================================================
# 2. Model Evaluation
# ======================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy : {accuracy:.4f}")

# ======================================================
# Classification Report
# ======================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    )
)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(
        RESULTS,
        "classification_report.csv"
    )
)

print("✓ Saved classification_report.csv")

# ======================================================
# Prediction Results
# ======================================================

prediction_results = pd.DataFrame({
    "Actual_Label": y_test.map({
        0: "Legitimate",
        1: "Phishing"
    }).values,

    "Predicted_Label": pd.Series(y_pred).map({
        0: "Legitimate",
        1: "Phishing"
    }).values
})

prediction_results.to_csv(
    os.path.join(
        RESULTS,
        "prediction_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("✓ Saved prediction_results.csv")

# ======================================================
# Confusion Matrix
# ======================================================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULTS,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Saved confusion_matrix.png")

# ======================================================
# ROC Curve
# ======================================================

print("\nGenerating ROC Curve...")

# y_test already contains numeric labels (0 and 1)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(
    y_test,
    y_prob
)

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="red"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.grid(True)

plt.legend()

plt.savefig(
    os.path.join(
        RESULTS,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ AUC Score : {auc:.4f}")
print("✓ Saved roc_curve.png")
# ======================================================
# 4. Learning Curve
# ======================================================

print("\n" + "=" * 60)
print("LEARNING CURVE")
print("=" * 60)

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="s",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")

plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULTS,
        "learning_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Saved learning_curve.png")

# ======================================================
# Evaluation Summary
# ======================================================

print("\nSaving Evaluation Summary...")

summary = f"""
============================================================
AI-Based Phishing Detection System
Evaluation Summary
============================================================

Dataset Information
-------------------
Dataset File           : cleaned_multilingual.csv
Total Samples          : {len(df)}

Training Samples       : {X_train.shape[0]}
Testing Samples        : {X_test.shape[0]}

Machine Learning Model
----------------------
Model                  : {type(model).__name__}

Feature Extraction
------------------
Technique              : TF-IDF Vectorizer

Cross Validation
----------------
Fold 1                 : {scores[0]:.4f}
Fold 2                 : {scores[1]:.4f}
Fold 3                 : {scores[2]:.4f}
Fold 4                 : {scores[3]:.4f}
Fold 5                 : {scores[4]:.4f}

Mean Accuracy          : {scores.mean():.4f}
Std Deviation          : {scores.std():.4f}

Testing Performance
-------------------
Accuracy               : {accuracy:.4f}

ROC Analysis
------------
AUC Score              : {auc:.4f}

Generated Files
---------------
classification_report.csv
cross_validation_results.csv
prediction_results.csv
confusion_matrix.png
roc_curve.png
learning_curve.png

============================================================
Evaluation Completed Successfully
============================================================
"""

with open(
    os.path.join(
        RESULTS,
        "evaluation_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

print("✓ Saved evaluation_summary.txt")

print("\n" + "=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print("\nGenerated Files")

print("----------------------------")
print("✓ classification_report.csv")
print("✓ cross_validation_results.csv")
print("✓ prediction_results.csv")
print("✓ evaluation_summary.txt")
print("✓ confusion_matrix.png")
print("✓ roc_curve.png")
print("✓ learning_curve.png")
print("----------------------------")