# predict.py
import re
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")

MODEL_DIR = "models"

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
# LOAD ARTIFACTS
# -----------------------
# Option 1: All files directly in models/ folder
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(f"{MODEL_DIR}/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# OR Option 2: Files in models/data/ subfolder
MODEL_DIR = "models"
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_sentiment(text):
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)
    sentiment = label_encoder.inverse_transform(pred)[0]
    return sentiment
