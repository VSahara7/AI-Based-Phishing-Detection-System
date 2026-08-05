"""
merge_all.py

Merge all English, Nepali Unicode and Romanized Nepali
datasets into one multilingual dataset.
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Project Path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW = BASE_DIR / "datasets" / "raw"
GENERATED = BASE_DIR / "datasets" / "generated"
PROCESSED = BASE_DIR / "datasets" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

email = pd.read_csv(RAW / "phishing_email.csv")
sms = pd.read_csv(RAW / "spam.csv")
nepali = pd.read_csv(GENERATED / "nepali_unicode.csv")
romanized = pd.read_csv(GENERATED / "romanized_nepali.csv")

# --------------------------------------------------
# Prepare English Email
# --------------------------------------------------

email = email[["text_combined", "label"]]

email = email.rename(
    columns={
        "text_combined": "text"
    }
)

email["label"] = email["label"].replace(
    {
        0: "legitimate",
        1: "phishing"
    }
)

# --------------------------------------------------
# Prepare SMS
# --------------------------------------------------

sms = sms[["text", "label"]]

sms["label"] = sms["label"].replace(
    {
        "ham": "legitimate",
        "spam": "phishing"
    }
)

# --------------------------------------------------
# Merge
# --------------------------------------------------

merged = pd.concat(
    [
        email,
        sms,
        nepali,
        romanized
    ],
    ignore_index=True
)

print(f"Rows before cleaning : {len(merged)}")

# --------------------------------------------------
# Remove missing values
# --------------------------------------------------

merged.dropna(inplace=True)

# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------

merged.drop_duplicates(
    subset=["text"],
    inplace=True
)

# --------------------------------------------------
# Shuffle
# --------------------------------------------------

merged = merged.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# --------------------------------------------------
# Save
# --------------------------------------------------

output = PROCESSED / "merged_multilingual.csv"

merged.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)

print("MERGED SUCCESSFULLY")

print("=" * 60)

print(f"Final Dataset Size : {len(merged)}")

print("\nClass Distribution")

print(merged["label"].value_counts())

print("\nLanguage Summary")

print(f"English Email : {len(email)}")
print(f"English SMS   : {len(sms)}")
print(f"Nepali        : {len(nepali)}")
print(f"Romanized     : {len(romanized)}")

print(f"\nSaved to:\n{output}")

print("\nSample")

print(merged.sample(10))