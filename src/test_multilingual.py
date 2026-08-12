import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = "/Users/user/Documents/Phishing"

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
TEST_DATA = os.path.join(BASE_DIR, "datasets", "testing", "unseen_messages.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# Load Model
# =====================================================

print("="*60)
print("Loading Model...")
print("="*60)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("✓ Model Loaded")
print("✓ Vectorizer Loaded")

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(TEST_DATA)

print("\nDataset Loaded Successfully")
print(df.head())

X = df["text"].astype(str)
label_map = {
    "legitimate": 0,
    "phishing": 1
}

y_true = df["expected_label"].map(label_map)

# =====================================================
# Vectorize
# =====================================================

X_vectorized = vectorizer.transform(X)

# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_vectorized)

# =====================================================
# Save Results
# =====================================================

results = df.copy()
results["Predicted"] = y_pred
results["Correct"] = results["expected_label"] == results["Predicted"]

output_file = os.path.join(
    RESULTS_DIR,
    "multilingual_test_results.csv"
)

results.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# Accuracy
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

print("\nAccuracy:", round(accuracy*100,2), "%")

print("\nClassification Report\n")

print(classification_report(y_true, y_pred))

print("\nResults saved to:")
print(output_file)

print("="*60)
print("Testing Completed Successfully")
print("="*60)