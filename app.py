from flask import Flask, render_template, request
from pathlib import Path
import joblib

app = Flask(__name__, template_folder="website")

# ------------------------------------------------
# Load Model
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")


def predict_message(message):

    vector = vectorizer.transform([str(message)])

    prediction = model.predict(vector)[0]

    label = encoder.inverse_transform([prediction])[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max() * 100

    if confidence is None:
        confidence = 95.0

    if confidence >= 90:
        risk = "High"
    elif confidence >= 70:
        risk = "Medium"
    else:
        risk = "Low"

    return label, round(confidence, 2), risk


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    label, confidence, risk = predict_message(message)

    if label == "phishing":

        title = "🚨 Phishing Detected"

        color = "#ff3b3b"

        advice = [
            "Do not click suspicious links.",
            "Do not share your OTP or password.",
            "Verify the sender through official channels."
        ]

    else:

        title = "✅ Legitimate Message"

        color = "#00c853"

        advice = [
            "No phishing indicators detected.",
            "Continue to stay alert.",
            "Always verify unexpected requests."
        ]

    return render_template(
        "result.html",
        message=message,
        prediction=label,
        confidence=confidence,
        risk=risk,
        title=title,
        color=color,
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)