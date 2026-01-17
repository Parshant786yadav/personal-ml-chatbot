import pickle
import os
import string
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------- Load ML Model --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")

with open(model_path, "rb") as f:
    model, vectorizer, data = pickle.load(f)

# -------- Text Preprocessing --------
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

# -------- Chat API --------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"reply": "Please type something."})

    processed = preprocess(user_input)
    X = vectorizer.transform([processed])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)

    if max_prob < 0.6:
        return jsonify({
            "reply": "I can answer questions related to Parshant only. Please rephrase."
        })

    intent = model.classes_[probs.argmax()]

    for i in data["intents"]:
        if i["tag"] == intent:
            return jsonify({"reply": i["responses"][0]})

    return jsonify({"reply": "Sorry, I didn’t understand that."})

# -------- Run Server --------
if __name__ == "__main__":
    app.run(debug=True)
