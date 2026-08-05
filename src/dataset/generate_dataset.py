import random
from pathlib import Path
import pandas as pd

from vocabulary import (
    BANKS,
    WALLETS,
    TELECOM,
    DELIVERY,
    GOVERNMENT,
    AMOUNTS,
    LINKS
)

from templates import (
    BANKING_PHISHING,
    WALLET_PHISHING,
    LOTTERY_PHISHING,
    DELIVERY_PHISHING,
    TELECOM_PHISHING,
    LEGITIMATE_MESSAGES
)

PHISHING_TEMPLATES = (
    BANKING_PHISHING +
    WALLET_PHISHING +
    LOTTERY_PHISHING +
    DELIVERY_PHISHING +
    TELECOM_PHISHING
)
# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

random.seed(42)

# -------------------------------------------------
# Output Folder
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "datasets" / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Message Generators
# -------------------------------------------------

def generate_phishing():

    template = random.choice(PHISHING_TEMPLATES)

    return template.format(
        bank=random.choice(BANKS),
        wallet=random.choice(WALLETS),
        telecom=random.choice(TELECOM),
        delivery=random.choice(DELIVERY),
        government=random.choice(GOVERNMENT),
        amount=random.choice(AMOUNTS),
        link=random.choice(LINKS)
    )


def generate_legitimate():

    return random.choice(LEGITIMATE_TEMPLATES)

# -------------------------------------------------
# Build Dataset
# -------------------------------------------------

dataset = []

print("Generating phishing messages...")

for _ in range(5000):

    dataset.append({
        "text": generate_phishing(),
        "label": "phishing"
    })

print("Generating legitimate messages...")

for _ in range(5000):

    dataset.append({
        "text": generate_legitimate(),
        "label": "legitimate"
    })

# -------------------------------------------------
# Shuffle
# -------------------------------------------------

random.shuffle(dataset)

df = pd.DataFrame(dataset)

# -------------------------------------------------
# Remove duplicates
# -------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["text"])

df = df.reset_index(drop=True)

after = len(df)

# -------------------------------------------------
# Save
# -------------------------------------------------

output_file = OUTPUT_DIR / "nepali_unicode.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n" + "=" * 60)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"Messages Generated : {before}")
print(f"Duplicates Removed : {before-after}")
print(f"Final Dataset Size : {after}")

print("\nClass Distribution")

print(df["label"].value_counts())

print(f"\nSaved to:\n{output_file}")

print("\nSample Messages")

print(df.sample(10))