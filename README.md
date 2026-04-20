# SentimentAI – Flask Sentiment Analysis App

A simple web-based sentiment analysis tool that classifies user input as Positive, Negative, or Neutral using NLP techniques.

## 🚀 Features
- Real-time sentiment prediction from user input
- Clean Flask-based web interface
- NLP preprocessing for improved accuracy
- Lightweight and easy to run locally

## 🛠️ Tech Stack
- Python
- Flask
- NLP (Hugging Face / TextBlob / custom model)
- HTML/CSS

## 📊 Result
Achieved ~85% accuracy on test data, improving prediction reliability through preprocessing and model tuning.

## Files

- `app.py` - the main Python file you run
- `templates/index.html` - the frontend page
- `requirements.txt` - the Python packages you install

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5001/`.
