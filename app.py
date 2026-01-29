from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load ML components
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

@app.route("/")
def home():
    return "Fake Job Posting Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "job_description" not in data:
        return jsonify({"error": "job_description is required"}), 400

    clean = preprocess_text(data["job_description"])
    vector = tfidf.transform([clean])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0].max()

    return jsonify({
        "prediction": "Fake Job" if pred == 1 else "Real Job",
        "confidence": round(float(prob), 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
