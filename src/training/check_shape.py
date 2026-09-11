import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # adjust if you run this from a different folder
DATASET = BASE_DIR / "datasets" / "processed" / "cleaned_multilingual.csv"

df = pd.read_csv(DATASET)
print("Dataset Shape (before cleaning):", df.shape)

df = df[["clean_text", "label"]]
df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)
df = df[df["clean_text"].str.strip() != ""]
df = df.drop_duplicates(subset=["clean_text"])
df = df.reset_index(drop=True)

print("Dataset Shape After Cleaning:", df.shape)