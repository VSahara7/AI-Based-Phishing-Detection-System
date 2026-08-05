"""
predict.py

Load the trained model and predict
whether a message is phishing or legitimate.
"""

from pathlib import Path
import joblib

# -----------------------------------------
# Paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"

# -----------------------------------------
# Load Files
# -----------------------------------------

model = joblib.load(MODEL_DIR / "best_model.pkl")

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)

# -----------------------------------------
# Prediction Function
# -----------------------------------------

def predict_message(message):

    message = str(message)

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    label = encoder.inverse_transform([prediction])[0]

    confidence = None

    # Random Forest supports probabilities
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(vector)[0]

        confidence = probability.max() * 100

    return {

        "prediction": label,

        "confidence": confidence

    }


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":

    while True:

        print()

        message = input("Enter Message: ")

        if message.lower() == "exit":
            break

        result = predict_message(message)

        print()

        print(result)