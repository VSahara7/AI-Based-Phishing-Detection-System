"""
preprocess.py

Multilingual preprocessing for:

- English
- Nepali Unicode
- Romanized Nepali
"""

import re
from pathlib import Path

import pandas as pd

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------
# Download Resources
# ------------------------------

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    lemmatizer = WordNetLemmatizer()

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT = BASE_DIR / "datasets" / "processed" / "merged_multilingual.csv"

OUTPUT = BASE_DIR / "datasets" / "processed" / "cleaned_multilingual.csv"

# ------------------------------
# Load Dataset
# ------------------------------

df = pd.read_csv(INPUT)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print(df.head())

# ------------------------------
# Language Detection
# ------------------------------

def is_nepali(text):

    return bool(re.search(r'[\u0900-\u097F]', str(text)))

# ------------------------------
# English Cleaning
# ------------------------------

def clean_english(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = []

    for word in text.split():

        if word not in stop_words:

            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# ------------------------------
# Nepali Cleaning
# ------------------------------

def clean_nepali(text):

    text = str(text)

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ------------------------------
# Apply Cleaning
# ------------------------------

cleaned = []

for message in df["text"]:

    if is_nepali(message):

        cleaned.append(clean_nepali(message))

    else:

        cleaned.append(clean_english(message))

df["clean_text"] = cleaned

# ------------------------------
# Save
# ------------------------------

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(df.head())

print(f"\nSaved to:\n{OUTPUT}")