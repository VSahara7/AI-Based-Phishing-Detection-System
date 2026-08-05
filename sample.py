import pandas as pd

df = pd.read_csv(
    "datasets/processed/merged_multilingual.csv"
)

sample = df.sample(
    5000,
    random_state=42
)

sample.to_csv(
    "datasets/processed/merged_multilingual_sample.csv",
    index=False
)