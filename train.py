import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_reviews.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def normalize_labels(labels):
    fake_labels = {"CG", "FAKE", "1", "SUSPICIOUS"}
    real_labels = {"OR", "REAL", "0", "GENUINE"}
    normalized = []

    for label in labels:
        value = str(label).strip().upper()
        if value in fake_labels:
            normalized.append(1)
        elif value in real_labels:
            normalized.append(0)
        else:
            raise ValueError(f"Unrecognized review label: {label!r}")

    return pd.Series(normalized, index=labels.index, dtype="int64")


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Training CSV must contain 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    if data.empty:
        raise ValueError("Training CSV has no non-empty labeled reviews.")
    if "source" in data.columns:
        print("Dataset sources:")
        print(data["source"].value_counts().to_string())

    targets = normalize_labels(data["label"])
    class_counts = targets.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Training requires at least two reviews in each class.")

    train_text, test_text, train_targets, test_targets = train_test_split(
        data["text"],
        targets,
        test_size=0.3,
        random_state=42,
        stratify=targets,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(train_features, train_targets)
    predictions = model.predict(test_features)

    print(f"Reviews: {len(data)} (genuine={int((targets == 0).sum())}, suspicious={int((targets == 1).sum())})")
    print(f"Holdout accuracy: {accuracy_score(test_targets, predictions):.2%}")
    print("Confusion matrix (rows=actual, columns=predicted; labels 0=genuine, 1=suspicious):")
    print(confusion_matrix(test_targets, predictions, labels=[0, 1]))
    print(classification_report(
        test_targets,
        predictions,
        labels=[0, 1],
        target_names=["genuine", "suspicious"],
        zero_division=0,
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"Saved model artifacts to {MODEL_DIR}")
    if "source" in data.columns and data["source"].eq("synthetic_demo").all():
        print(
            "WARNING: All bundled examples are synthetic demonstrations. "
            "Holdout metrics do not measure performance on real customer reviews."
        )
    if len(data) < 200:
        print("WARNING: This is a very small dataset. Holdout metrics are not evidence of production accuracy.")


if __name__ == "__main__":
    main()
