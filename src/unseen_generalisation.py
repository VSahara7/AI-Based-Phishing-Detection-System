import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "results"

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from difflib import SequenceMatcher


# =========================================================
# 1. LOAD UNSEEN DATASET
# =========================================================

unseen = pd.read_csv("datasets/testing/unseen_messages.csv")

print("Columns in unseen_messages.csv:")
print(unseen.columns.tolist())

print("\nFirst 5 rows:")
print(unseen.head())


# =========================================================
# 2. LOAD MODEL AND PREPROCESSING OBJECTS
# =========================================================

model = joblib.load("models/random_forest.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# =========================================================
# 3. TRANSFORM UNSEEN MESSAGES
# =========================================================

X_unseen = tfidf.transform(
    unseen["text"].astype(str)
)

y_true = label_encoder.transform(
    unseen["expected_label"].astype(str)
)

y_pred = model.predict(X_unseen)


# Decode predictions
unseen["actual"] = label_encoder.inverse_transform(y_true)
unseen["predicted"] = label_encoder.inverse_transform(y_pred)

unseen["correct"] = (
    unseen["actual"] == unseen["predicted"]
)


# =========================================================
# 4. OVERALL PERFORMANCE
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n==============================")
print("UNSEEN GENERALISATION RESULTS")
print("==============================")

print(
    f"Overall Accuracy: "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# =========================================================
# 5. AUTOMATIC SCRIPT / LANGUAGE DETECTION
# =========================================================

def detect_language(text):

    text = str(text)

    # Check for Devanagari Unicode characters
    devanagari_count = sum(
        0x0900 <= ord(char) <= 0x097F
        for char in text
    )

    if devanagari_count > 0:
        return "Nepali Unicode"

    # Remaining non-Devanagari messages are treated as
    # English or Romanized Nepali based on common indicators.

    romanized_words = [
        "tapai",
        "tapaiko",
        "hajur",
        "khata",
        "bank",
        "bistarai",
        "garnuhos",
        "garnu",
        "verify",
        "paisaa",
        "paisa",
        "mobile",
        "sewa",
        "khalti",
        "ime",
        "bikash",
        "hunuhuncha",
        "hunchha"
    ]

    words = text.lower().split()

    if any(
        word.strip(".,!?") in romanized_words
        for word in words
    ):
        return "Romanized Nepali"

    return "English"


unseen["language"] = unseen["text"].apply(
    detect_language
)


# =========================================================
# 6. PERFORMANCE BY LANGUAGE
# =========================================================

print("\n==============================")
print("PERFORMANCE BY LANGUAGE")
print("==============================")

for language in unseen["language"].unique():

    subset = unseen[
        unseen["language"] == language
    ]

    language_accuracy = accuracy_score(
        subset["actual"],
        subset["predicted"]
    )

    correct = (
        subset["actual"] ==
        subset["predicted"]
    ).sum()

    total = len(subset)

    print(
        f"{language}: "
        f"{language_accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )


# =========================================================
# 7. DEVANAGARI DIAGNOSTIC
# =========================================================

dev = unseen[
    unseen["language"] == "Nepali Unicode"
]

if len(dev) > 0:

    print("\n==============================")
    print("DEVANAGARI DIAGNOSTIC")
    print("==============================")

    dev_accuracy = accuracy_score(
        dev["actual"],
        dev["predicted"]
    )

    print(
        f"Devanagari Accuracy: "
        f"{dev_accuracy * 100:.2f}%"
    )

    print("\nDevanagari Confusion Matrix:")

    print(
        confusion_matrix(
            dev["actual"],
            dev["predicted"],
            labels=label_encoder.classes_
        )
    )

    print("\nDevanagari Predictions:")

    print(
        dev[
            [
                "text",
                "actual",
                "predicted",
                "correct"
            ]
        ].to_string(index=False)
    )


# =========================================================
# 8. INCORRECT PREDICTIONS
# =========================================================

errors = unseen[
    ~unseen["correct"]
]

print("\n==============================")
print("INCORRECT UNSEEN PREDICTIONS")
print("==============================")

if len(errors) == 0:

    print("No incorrect predictions.")

else:

    print(
        errors[
            [
                "language",
                "text",
                "actual",
                "predicted"
            ]
        ].to_string(index=False)
    )


# =========================================================
# 9. DUPLICATE / TEMPLATE DIAGNOSTIC
# =========================================================

def normalise_text(text):

    return " ".join(
        str(text).lower().split()
    )


unseen["normalised_text"] = (
    unseen["text"].apply(normalise_text)
)

duplicate_count = (
    unseen["normalised_text"]
    .duplicated()
    .sum()
)

print("\n==============================")
print("TEMPLATE / DUPLICATE DIAGNOSTIC")
print("==============================")

print(
    f"Exact duplicate messages: "
    f"{duplicate_count}"
)


# =========================================================
# 10. DEVANAGARI TEMPLATE SIMILARITY
# =========================================================

if len(dev) > 1:

    similarities = []

    dev_texts = dev["text"].tolist()

    for i in range(len(dev_texts)):

        for j in range(i + 1, len(dev_texts)):

            similarity = SequenceMatcher(
                None,
                dev_texts[i],
                dev_texts[j]
            ).ratio()

            similarities.append(similarity)

    average_similarity = np.mean(
        similarities
    )

    print(
        f"Average Devanagari "
        f"pairwise similarity: "
        f"{average_similarity:.3f}"
    )


# =========================================================
# 11. SAVE RESULTS
# =========================================================

unseen.to_csv(
    "results/unseen_multilingual_diagnostics.csv",
    index=False
)

print("\n==============================")
print("DIAGNOSTIC COMPLETE")
print("==============================")

print(
    "Results saved to:"
)

print(
    "results/unseen_multilingual_diagnostics.csv"
)

# Unseen evaluation results
results = {
    "Metric": [
        "Overall Accuracy",
        "Legitimate Precision",
        "Legitimate Recall",
        "Legitimate F1",
        "Phishing Precision",
        "Phishing Recall",
        "Phishing F1"
    ],
    "Score": [
        90,
        83,
        100,
        91,
        100,
        80,
        89
    ]
}

df = pd.DataFrame(results)

# Create graph
plt.figure(figsize=(10, 6))

bars = plt.bar(
    df["Metric"],
    df["Score"],
    edgecolor="black",
    linewidth=1
)

# Add percentage values
for bar, value in zip(bars, df["Score"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.title(
    "Random Forest Performance on Unseen Messages",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Performance (%)")
plt.ylim(0, 110)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

# Save figure
plt.savefig(
    OUT_DIR / "unseen_random_forest_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()