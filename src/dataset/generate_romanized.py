import random
from pathlib import Path
import pandas as pd

from romanized import (
    BANKS,
    WALLETS,
    TELECOM,
    DELIVERY,
    AMOUNTS,
    LINKS
)

from romanized_templates import (
    PHISHING,
    LEGITIMATE
)

random.seed(42)


def generate_phishing():

    return random.choice(PHISHING).format(
        bank=random.choice(BANKS),
        wallet=random.choice(WALLETS),
        telecom=random.choice(TELECOM),
        delivery=random.choice(DELIVERY),
        amount=random.choice(AMOUNTS),
        link=random.choice(LINKS)
    )


def generate_legitimate():

    return random.choice(LEGITIMATE)


dataset = []

for _ in range(5000):

    dataset.append({
        "text": generate_phishing(),
        "label": "phishing"
    })

for _ in range(5000):

    dataset.append({
        "text": generate_legitimate(),
        "label": "legitimate"
    })

random.shuffle(dataset)

df = pd.DataFrame(dataset)

before = len(df)

df = df.drop_duplicates(subset="text")

after = len(df)

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT = BASE_DIR / "datasets" / "generated"

OUTPUT.mkdir(parents=True, exist_ok=True)

file = OUTPUT / "romanized_nepali.csv"

df.to_csv(file, index=False, encoding="utf-8-sig")

print("=" * 60)
print("Romanized Nepali Dataset Generated")
print("=" * 60)

print(f"Generated : {before}")
print(f"Duplicates Removed : {before-after}")
print(f"Final Size : {len(df)}")

print(df["label"].value_counts())

print(f"\nSaved to:\n{file}")