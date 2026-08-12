import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

# ======================================================
# Create output folders
# ======================================================

os.makedirs("/Users/user/Documents/Phishing/results", exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================

print("="*60)
print("Loading Dataset")
print("="*60)

df = pd.read_csv("datasets/processed/cleaned_multilingual.csv")

df = df.dropna(subset=["clean_text","label"])

X = df["clean_text"].astype(str)
y = df["label"].astype(str)

print(df.shape)

# ======================================================
# Load TF-IDF Vectorizer
# ======================================================

vectorizer = joblib.load("/Users/user/Documents/Phishing/models/tfidf_vectorizer.pkl")

X = vectorizer.transform(X)

# ======================================================
# Train-Test Split
# ======================================================

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================================
# Load Best Model
# ======================================================

model=joblib.load("/Users/user/Documents/Phishing/models/best_model.pkl")

# ======================================================
# Train
# ======================================================

model.fit(X_train,y_train)

# ======================================================
# Classification Report
# ======================================================

print("\nClassification Report\n")

y_pred=model.predict(X_test)

print(classification_report(y_test,y_pred))

# Save report

with open("results/classification_report.txt","w") as f:
    f.write(classification_report(y_test,y_pred))

# ======================================================
# Cross Validation
# ======================================================

print("\nRunning 5 Fold Cross Validation...\n")

scores=cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

for i,s in enumerate(scores):
    print(f"Fold {i+1}: {s:.4f}")

print()

print("Average Accuracy:",scores.mean())
print("Std:",scores.std())

cv_df=pd.DataFrame({
    "Fold":[1,2,3,4,5],
    "Accuracy":scores
})

cv_df.to_csv("/Users/user/Documents/Phishing/results/cross_validation_results.csv",index=False)

# ======================================================
# Confusion Matrix
# ======================================================

print("\nGenerating Confusion Matrix")

cm=confusion_matrix(y_test,y_pred)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig,ax=plt.subplots(figsize=(7,6))

disp.plot(ax=ax,cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("/Users/user/Documents/Phishing/results/confusion_matrix.png",dpi=300)

plt.close()

# ======================================================
# ROC Curve
# ======================================================

print("Generating ROC Curve")

positive="phishing"

y_binary=(y_test==positive).astype(int)

prob=model.predict_proba(X_test)[:,1]

auc=roc_auc_score(
    y_binary,
    prob
)

fpr,tpr,thresholds=roc_curve(
    y_binary,
    prob
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC={auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig("results/roc_curve.png",dpi=300)

plt.close()

print("AUC =",auc)

# ======================================================
# Learning Curve
# ======================================================

print("\nGenerating Learning Curve")

train_sizes,train_scores,test_scores=learning_curve(

    estimator=model,

    X=X_train,

    y=y_train,

    cv=5,

    scoring="accuracy",

    train_sizes=np.linspace(0.1,1.0,5),

    n_jobs=-1

)

train_mean=train_scores.mean(axis=1)

test_mean=test_scores.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(

    train_sizes,

    train_mean,

    marker="o",

    linewidth=2,

    label="Training Accuracy"

)

plt.plot(

    train_sizes,

    test_mean,

    marker="s",

    linewidth=2,

    label="Validation Accuracy"

)

plt.xlabel("Training Examples")

plt.ylabel("Accuracy")

plt.title("Learning Curve")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig("results/learning_curve.png",dpi=300)

plt.close()

# ======================================================
# Unseen Multilingual Testing
# ======================================================

print("\nTesting Unseen Dataset")

test_df=pd.read_csv("datasets/testing/unseen_messages.csv")

test_df["text"]=test_df["text"].fillna("").astype(str)

X_new=vectorizer.transform(test_df["text"])

prediction=model.predict(X_new)

test_df["prediction"]=prediction

test_df["correct"]=(
    test_df["expected_label"]==
    test_df["prediction"]
)

accuracy=test_df["correct"].mean()*100

print(test_df)

print()

print("Unseen Accuracy:",accuracy)

test_df.to_csv(
    "results/prediction_results.csv",
    index=False
)

# ======================================================
# Finished
# ======================================================

print()

print("="*60)

print("ALL EVALUATIONS COMPLETED")

print("="*60)

print()

print("Results saved in results/ folder")