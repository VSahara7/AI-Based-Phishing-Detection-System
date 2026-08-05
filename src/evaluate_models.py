import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -------------------------------------
# Load Test Data
# -------------------------------------

X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# -------------------------------------
# Load Models
# -------------------------------------

models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl")
}

results = []

os.makedirs("outputs/confusion_matrices", exist_ok=True)

print("="*60)
print("MODEL EVALUATION")
print("="*60)

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"]
    )

    disp.plot()

    plt.title(name)

    plt.savefig(
        f"outputs/confusion_matrices/{name.replace(' ','_')}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# -------------------------------------
# Save Results
# -------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\n")
print("="*60)
print(results_df)
print("="*60)

best_model = results_df.iloc[0]["Model"]

print(f"\nBest Model: {best_model}")