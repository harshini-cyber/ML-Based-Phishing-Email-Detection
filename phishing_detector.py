import pandas as pd

# Load CSV
df = pd.read_csv("emails.csv")

# Show column names
print("Columns found:")
print(df.columns)

# Show first rows
print("\nFirst 5 rows:")
print(df.head())
import re
import joblib
import matplotlib.pyplot as plt
from urllib.parse import urlparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load Dataset
df = pd.read_csv("emails.csv")

# Custom Feature Extractor
class URLFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        phishing_keywords = [
            "verify",
            "login",
            "password",
            "bank",
            "urgent",
            "click",
            "update",
            "account"
        ]

        features = []

        for text in X:

            urls = re.findall(r'https?://\S+|www\.\S+', str(text))

            url_count = len(urls)

            keyword_count = sum(
                text.lower().count(word)
                for word in phishing_keywords
            )

            suspicious_url = 0

            for url in urls:
                domain = urlparse(url).netloc

                if "-" in domain:
                    suspicious_url = 1

            features.append([
                url_count,
                keyword_count,
                suspicious_url
            ])

        return features

# Feature Engineering
features = FeatureUnion([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
    ),
    (
        "url_features",
        URLFeatures()
    )
])

# Model Pipeline
model = Pipeline([
    ("features", features),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")

# Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

disp.plot()

plt.title("Phishing Detection Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Save Model
joblib.dump(model, "phishing_model.pkl")

print("\nModel saved as phishing_model.pkl")
print("Confusion matrix saved as confusion_matrix.png")
sample_email = [
    "Urgent! Verify your bank account now by clicking the link."
]

prediction = model.predict(sample_email)

if prediction[0] == 1:
    print("\nResult: Phishing Email")
else:
    print("\nResult: Safe Email")