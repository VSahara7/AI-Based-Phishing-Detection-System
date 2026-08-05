import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# Load TF-IDF Features
# ------------------------------------------

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("=" * 50)
print("Training Machine Learning Models")
print("=" * 50)

# ------------------------------------------
# Create models folder
# ------------------------------------------

os.makedirs("models", exist_ok=True)

# ------------------------------------------
# Logistic Regression
# ------------------------------------------

print("\nTraining Logistic Regression...")

lr = LogisticRegression(max_iter=1000)

print(type(y_train))
print(y_train[:10])
print(set(y_train))

lr.fit(X_train, y_train)

joblib.dump(lr, "models/logistic_regression.pkl")

print("✓ Logistic Regression Saved")

# ------------------------------------------
# Naive Bayes
# ------------------------------------------

print("\nTraining Multinomial Naive Bayes...")

nb = MultinomialNB()

nb.fit(X_train, y_train)

joblib.dump(nb, "models/naive_bayes.pkl")

print("✓ Naive Bayes Saved")

# ------------------------------------------
# Support Vector Machine
# ------------------------------------------

print("\nTraining Linear SVM...")

svm = LinearSVC()

svm.fit(X_train, y_train)

joblib.dump(svm, "models/svm.pkl")

print("✓ SVM Saved")

# ------------------------------------------
# Random Forest
# ------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

joblib.dump(rf, "models/random_forest.pkl")

print("✓ Random Forest Saved")

print("\n")
print("=" * 50)
print("All Models Trained Successfully!")
print("=" * 50)