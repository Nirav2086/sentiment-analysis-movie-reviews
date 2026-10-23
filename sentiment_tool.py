"""Core logic for the Sentiment Analysis Tool for Movie Reviews.

Classifies text (movie reviews, tweets) as negative, neutral or positive
using TF-IDF features and Logistic Regression.
"""
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]
MODEL_PATH = Path("models/sentiment_model.joblib")


def to_3class(label: int) -> int:
    """Map SST-5 labels (0-4) to 3 classes: 0 negative, 1 neutral, 2 positive."""
    if label in (0, 1):
        return 0
    if label == 2:
        return 1
    return 2


def clean_text(text: str) -> str:
    """Light cleaning. Stopwords are kept on purpose so negation ("not good") survives."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"@\w+", " ", text)                   # @mentions
    text = re.sub(r"#", "", text)                       # hashtag symbol
    text = re.sub(r"<.*?>", " ", text)                  # HTML tags
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)         # keep letters, digits, ! ? '
    return re.sub(r"\s+", " ", text).strip()


def load_data() -> dict:
    """Load SST-5 from Hugging Face and return train/validation/test DataFrames."""
    from datasets import load_dataset  # imported here so the rest works without it

    ds = load_dataset("SetFit/sst5")
    frames = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        df["sentiment"] = df["label"].map(to_3class)
        df["clean"] = df["text"].apply(clean_text)
        frames[split] = df
    return frames


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])


def save_model(model, path: Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS}, path)


def load_model(path: Path = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"]


def predict_sentiment(texts, model) -> pd.DataFrame:
    """Classify one string or a list of strings. Returns label, confidence and class probabilities."""
    if isinstance(texts, str):
        texts = [texts]
    probs = model.predict_proba([clean_text(t) for t in texts])
    preds = probs.argmax(axis=1)
    out = pd.DataFrame({
        "text": texts,
        "sentiment": [LABELS[p] for p in preds],
        "confidence": probs.max(axis=1).round(3),
    })
    for i, label in enumerate(LABELS):
        out[f"p_{label}"] = probs[:, i].round(3)
    return out
