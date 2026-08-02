import pandas as pd
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------------------
# Load Dataset
# ----------------------------------------
df = pd.read_csv("datasets/merged_messages.csv")

print("="*50)
print("Dataset Loaded")
print("="*50)
print(df.head())

stop_words = ENGLISH_STOP_WORDS

# ----------------------------------------
# Cleaning Function
# ----------------------------------------
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# ----------------------------------------
# Apply Cleaning
# ----------------------------------------
print("\nCleaning dataset...")

df["clean_text"] = df["text"].apply(clean_text)

# Remove empty rows
df = df[df["clean_text"] != ""]

# Save
df.to_csv("datasets/cleaned_messages.csv", index=False)

print("\nCleaning Completed!")

print("\nFinal Shape:")
print(df.shape)

print("\nSample:")
print(df[["clean_text", "label"]].head())

print("\nSaved:")
print("datasets/cleaned_messages.csv")