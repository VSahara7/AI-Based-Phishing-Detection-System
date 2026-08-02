import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -----------------------------------------
# Load Dataset
# -----------------------------------------
df = pd.read_csv("datasets/cleaned_messages.csv")

print("=" * 50)
print("Dataset Loaded")
print("=" * 50)
print(df.head())

# -----------------------------------------
# Features and Labels
# -----------------------------------------
X = df["clean_text"]

# Convert labels to numeric
y = df["label"].replace({
    "legitimate": 0,
    "phishing": 1
})

# -----------------------------------------
# TF-IDF
# -----------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:")
print(X_tfidf.shape)

# -----------------------------------------
# Train Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# -----------------------------------------
# Create models folder
# -----------------------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------------------
# Save TF-IDF Vectorizer
# -----------------------------------------
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# -----------------------------------------
# Save Train/Test Data
# -----------------------------------------
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nFiles Saved Successfully!")

print("""
Saved Files
-----------
✔ tfidf_vectorizer.pkl
✔ X_train.pkl
✔ X_test.pkl
✔ y_train.pkl
✔ y_test.pkl
""")