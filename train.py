# train.py
import os
import re
import string
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "data/raw/Reviews.csv"
MODEL_DIR = "models"

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------
# SETUP
# -----------------------
os.makedirs(MODEL_DIR, exist_ok=True)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -----------------------
# TEXT CLEANING
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# -----------------------
# SENTIMENT MAPPING
# -----------------------
def map_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

# -----------------------
# MAIN TRAINING PIPELINE
# -----------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning data...")
    df = df.dropna(subset=["Score", "Text"])
    df = df.drop_duplicates(subset=["Score", "Text"])

    df["raw_text"] = (
        df["Summary"].fillna("") + " " + df["Text"].fillna("")
    ).str.strip()

    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["Sentiment"] = df["Score"].apply(map_sentiment)

    # Encode target
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["Sentiment"])

    X = df["clean_text"]
    y = df["sentiment_encoded"]

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=5
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_tfidf, y_train)

    print("Saving artifacts...")
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    with open(os.path.join(MODEL_DIR, "logistic_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Training complete. Models saved to /models")

# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    main()
