import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
# Load merged dataset
df = pd.read_csv("datasets/merged_messages.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("="*50)
print("DATASET INFORMATION")
print("="*50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nClass Distribution:")
print(df["label"].value_counts())

# -----------------------------
# Message Length
# -----------------------------
df["message_length"] = df["text"].astype(str).apply(len)

print("\nMessage Length Statistics:")
print(df["message_length"].describe())

# -----------------------------
# Bar Chart
# -----------------------------
plt.figure(figsize=(6,4))
df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/class_distribution.png")
plt.show()

# -----------------------------
# Message Length Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["message_length"], bins=40)
plt.title("Message Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/message_length.png")
plt.show()

# -----------------------------
# Word Clouds
# -----------------------------
phishing_text = " ".join(
    df[df["label"]=="phishing"]["text"].astype(str)
)

legitimate_text = " ".join(
    df[df["label"]=="legitimate"]["text"].astype(str)
)

wc1 = WordCloud(width=1000,height=500,
                background_color="white").generate(phishing_text)

plt.figure(figsize=(12,6))
plt.imshow(wc1)
plt.axis("off")
plt.title("Phishing Messages Word Cloud")
plt.savefig("outputs/figures/phishing_wordcloud.png")
plt.show()

wc2 = WordCloud(width=1000,height=500,
                background_color="white").generate(legitimate_text)

plt.figure(figsize=(12,6))
plt.imshow(wc2)
plt.axis("off")
plt.title("Legitimate Messages Word Cloud")
plt.savefig("outputs/figures/legitimate_wordcloud.png")
plt.show()

print("\nEDA Completed Successfully.")