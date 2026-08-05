# AI-Based Phishing and Social Engineering Scam Detection System Using Machine Learning and NLP

## Overview

This project presents an AI-based phishing and social engineering scam detection system capable of identifying suspicious messages using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes message content and classifies it as either:

- ✅ Legitimate
- 🚨 Phishing

The project supports multilingual phishing detection including:

- English
- Nepali Unicode
- Romanized Nepali

The aim is to develop a practical cybersecurity solution that helps users identify phishing attempts commonly distributed through emails, SMS, and online messaging platforms.

---

# Features

## AI-Based Detection

The system uses machine learning algorithms to classify suspicious messages.

Implemented models:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest


## Multilingual Support

The system supports:

English:

Nepali Unicode:

Romanized Nepali:



## Web Application

The system provides a user-friendly interface where users can:

- Enter suspicious messages
- Receive instant predictions
- View confidence scores
- Check risk level
- Receive safety recommendations


---



---

# Dataset

The system combines multiple datasets:

## English Dataset

Sources:

- Phishing Email Dataset
- SMS Spam Dataset


## Nepali Dataset

A synthetic multilingual dataset was created containing:

- Nepali banking scams
- Wallet scams
- OTP scams
- QR payment scams
- Delivery scams
- Telecom scams
- Job scams
- Government impersonation scams


## Dataset Statistics

Final dataset: merged_multilingual.csv


---

# Model Evaluation

The following models are evaluated:

| Model | Accuracy |
|------|----------|
| Random Forest | Best Performing |
| SVM | High Performance |
| Logistic Regression | Good Baseline |
| Naive Bayes | Baseline Model |


Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score


---


# Installation

Clone repository:

```bash
git clone https://github.com/VSahara7/AI-Based-Phishing-Detection-System.git

Running the Application

Start Flask server:

python app.py

Open browser:

http://127.0.0.1:5000