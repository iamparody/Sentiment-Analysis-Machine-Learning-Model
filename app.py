# app.py
from flask import Flask, request, jsonify
from predict import predict_sentiment

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Amazon Reviews Sentiment API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not payload or "text" not in payload:
        return jsonify({
            "error": "Request body must contain a 'text' field"
        }), 400

    text = payload["text"]
    sentiment = predict_sentiment(text)

    return jsonify({
        "input_text": text,
        "predicted_sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
