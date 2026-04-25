import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
sentiment_pipeline = None
model_load_error = None


def get_sentiment_pipeline():
    global sentiment_pipeline, model_load_error

    if sentiment_pipeline is not None:
        return sentiment_pipeline

    if model_load_error is not None:
        return None

    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    except Exception as exc:
        model_load_error = (
            "The sentiment model is not available locally. "
            "Connect to the internet once so Hugging Face can download it, "
            "then restart the app. "
            f"Original error: {exc}"
        )
        return None

    return sentiment_pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 512:
        text = text[:512]

    current_pipeline = get_sentiment_pipeline()
    if current_pipeline is None:
        return jsonify(
            {
                "error": "Sentiment model could not be loaded.",
                "details": model_load_error,
            }
        ), 503

    result = current_pipeline(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    sentiment_map = {
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative",
    }

    return jsonify(
        {
            "sentiment": sentiment_map.get(label, label),
            "raw_label": label,
            "confidence": score,
            "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
