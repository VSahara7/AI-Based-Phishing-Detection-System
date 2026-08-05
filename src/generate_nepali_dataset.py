import random
import pandas as pd

# -------------------------
# Nepali Organizations
# -------------------------

banks = [
    "Nabil Bank",
    "NIC Asia Bank",
    "Global IME Bank",
    "Kumari Bank",
    "Everest Bank",
    "Prabhu Bank",
    "Siddhartha Bank",
    "Sanima Bank",
    "Himalayan Bank",
    "Machhapuchchhre Bank"
]

wallets = [
    "eSewa",
    "Khalti",
    "IME Pay",
    "FonePay"
]

telecom = [
    "Ncell",
    "Nepal Telecom"
]

delivery = [
    "Daraz",
    "Sastodeal",
    "Pathao"
]

amounts = [
    "रु 5,000",
    "रु 10,000",
    "रु 25,000",
    "रु 50,000",
    "रु 100,000",
    "रु 250,000"
]

links = [
    "https://verify-now.com",
    "https://secure-login.net",
    "https://claim-now.net",
    "https://account-update.org",
    "https://bank-security.info"
]
phishing_templates = [

    "तपाईंको {bank} खाता बन्द हुन लागेको छ। यहाँ क्लिक गर्नुहोस् {link}",

    "तपाईंले {amount} जित्नुभएको छ। पुरस्कार लिन {link}",

    "{wallet} KYC अपडेट गर्नुहोस्। {link}",

    "तपाईंको OTP पुष्टि गर्नुहोस्। {link}",

    "{bank} सुरक्षा कारणले तपाईंको खाता रोकिएको छ। {link}",

    "{delivery} बाट आएको पार्सल रोकिएको छ। {link}",

    "{telecom} नम्बर बन्द हुन लागेको छ। Verify गर्नुहोस् {link}",

    "Congratulations! You won {amount}. Click {link}",

    "Update your bank account immediately. {link}",

    "Your account has been suspended. Verify now {link}"

]
legitimate_templates = [

    "भोलि बिहान १० बजे बैठक छ।",

    "आज बेलुका भेटौं।",

    "तपाईंको बिजुली बिल तयार भएको छ।",

    "कक्षा बिहान ९ बजे सुरु हुनेछ।",

    "तपाईंको अपोइन्टमेन्ट पुष्टि भएको छ।",

    "आजको मौसम राम्रो छ।",

    "Your meeting is tomorrow at 10 AM.",

    "Dinner at 7 PM?",

    "Happy Birthday! Have a wonderful day.",

    "Your order has been delivered."

]
def generate_phishing():

    msg = random.choice(phishing_templates)

    return msg.format(
        bank=random.choice(banks),
        wallet=random.choice(wallets),
        telecom=random.choice(telecom),
        delivery=random.choice(delivery),
        amount=random.choice(amounts),
        link=random.choice(links)
    )


def generate_legitimate():

    return random.choice(legitimate_templates)

phishing = []
legitimate = []

for _ in range(2500):

    phishing.append({
        "text": generate_phishing(),
        "label": "phishing"
    })

for _ in range(2500):

    legitimate.append({
        "text": generate_legitimate(),
        "label": "legitimate"
    })

dataset = phishing + legitimate

random.shuffle(dataset)

df = pd.DataFrame(dataset)

from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Create output directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(exist_ok=True)

# Build dataset
dataset = []

for _ in range(2500):
    dataset.append({
        "text": generate_phishing(),
        "label": "phishing"
    })

for _ in range(2500):
    dataset.append({
        "text": generate_legitimate(),
        "label": "legitimate"
    })

# Shuffle
random.shuffle(dataset)

# Create DataFrame
df = pd.DataFrame(dataset)

# Remove duplicates
before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
after = len(df)

# Save
output_file = DATASET_DIR / "nepali_unicode.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("=" * 60)
print("Nepali Dataset Generated Successfully")
print("=" * 60)
print(f"Total Messages : {len(df)}")
print(f"Duplicates Removed : {before - after}")
print("\nClass Distribution:")
print(df["label"].value_counts())
print(f"\nSaved to: {output_file}")