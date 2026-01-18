import pickle
import os
import string
from rapidfuzz import fuzz
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------- Load ML Model --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")

with open(model_path, "rb") as f:
    model, vectorizer, data = pickle.load(f)

# -------- No Cache Response --------
def no_cache_response(payload):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# -------- Text Preprocessing --------
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # name normalization (VERY IMPORTANT)
    sentence = fuzzy_name_normalize(sentence)
    
    return sentence
def fuzzy_name_normalize(sentence):
    words = sentence.split()
    normalized = []

    for w in words:
        # fuzzy match with "parshant"
        if fuzz.ratio(w, "parshant") >= 80:
            normalized.append("parshant")
        else:
            normalized.append(w)

    return " ".join(normalized)

# -------- Chat API --------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if not user_input:
        return no_cache_response({"reply": "Please type something."})

    processed = preprocess(user_input)

    # -------- RULE 1: GREETING (NO ML) --------
    GREETINGS = {"hi", "hello", "hey", "hii", "hyy", "helo"}

    if processed.strip() in GREETINGS:
        return no_cache_response({
            "reply": "Hyy! I am Parshant's personal AI assistant. How can I help you?",
            "intent": "greeting-rule"
        })

    # -------- RULE 2: SHORT QUESTIONS --------
    if "parshant" in processed and "how" in processed:
        return no_cache_response({
            "reply": "Parshant is doing well and currently focusing on his studies and projects.",
            "intent": "rule_how_is"
        })

    # -------- ML STARTS HERE --------
    X = vectorizer.transform([processed])
    probs = model.predict_proba(X)[0]
    intent = model.classes_[probs.argmax()]
    max_prob = float(max(probs))

    # Debug (you can remove later)
    print("INPUT:", user_input)
    print("PROCESSED:", processed)
    print("INTENT:", intent)
    print("PROBABILITY:", max_prob)

    # Intent → response map
    intent_response_map = {
        i["tag"]: i["responses"][0] for i in data["intents"]
    }

    # -------- GREETING VIA ML (BACKUP) --------
    if intent == "greeting":
        return no_cache_response({
            "reply": intent_response_map["greeting"],
            "intent": intent
        })

    # -------- FALLBACK --------
    if max_prob < 0.2:
        return no_cache_response({
            "reply": "I can answer questions related to Parshant only. Please rephrase.",
            "intent": "fallback"
        })

    # -------- NORMAL INTENT RESPONSE --------
    if intent in intent_response_map:
        return no_cache_response({
            "reply": intent_response_map[intent],
            "intent": intent
        })

    return no_cache_response({"reply": "Sorry, I didn’t understand that."})

# -------- Run Server --------
if __name__ == "__main__":
    app.run(debug=True)
