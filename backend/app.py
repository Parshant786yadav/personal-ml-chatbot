import pickle
import os
import string
from flask import session
from rapidfuzz import fuzz
from flask import Flask, request, jsonify

app = Flask(__name__)
app.secret_key = "temporary-chatbot-session-key"

# -------- Simple memory (single-user demo) --------
last_suggested_intent = None

YES_WORDS = {"yes", "yeah", "y", "yep", "haan", "haa", "ok"}
NO_WORDS = {"no", "nah", "nope", "nahi"}

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

# -------- Fuzzy Name Normalization --------
def fuzzy_name_normalize(sentence):
    words = sentence.split()
    normalized = []

    for w in words:
        if fuzz.ratio(w, "parshant") >= 80:
            normalized.append("parshant")
        else:
            normalized.append(w)

    return " ".join(normalized)

# -------- Text Preprocessing --------
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = fuzzy_name_normalize(sentence)
    return sentence


def respond_with_history(user_msg, bot_reply, intent=None):
    history = session.get("history", [])
    history.append({
        "user": user_msg,
        "bot": bot_reply
    })
    session["history"] = history

    return no_cache_response({
        "reply": bot_reply,
        "intent": intent,
        "history": history   # optional: send to frontend
    })

# -------- Chat API --------
@app.route("/chat", methods=["POST"])
def chat():
    # Initialize history for this session
    if "history" not in session:
          session["history"] = []

    global last_suggested_intent

    user_input = request.json.get("message", "")
    if not user_input:
        return no_cache_response({"reply": "Please type something."})

    processed = preprocess(user_input)

    # -------- Build intent → response map EARLY --------
    intent_response_map = {
        i["tag"]: i["responses"][0] for i in data["intents"]
    }

    # -------- USER CONFIRMATION HANDLING --------
    if processed in YES_WORDS and last_suggested_intent:
        reply = intent_response_map[last_suggested_intent]
        last_suggested_intent = None
        return respond_with_history(
            user_input,
            reply,
            "confirmation-yes"
        )


    if processed in NO_WORDS and last_suggested_intent:
        last_suggested_intent = None
        return respond_with_history(
            user_input,
            "Okay 👍 No problem. What would you like to know about Parshant?",
            "confirmation-no"
        )


    # -------- PRIORITY RULE: QUALIFICATION --------
    if any(word in processed for word in [
        "qualification", "degree", "education", "course", "academic"
    ]):
        return respond_with_history(
            user_input,
            intent_response_map["qualification"],
            "qualification-rule"
        )


    # -------- RULE 1: GREETING --------
    GREETINGS = {"hi", "hello", "hey", "hii", "hyy", "helo"}

    if processed.strip() in GREETINGS:
        return respond_with_history(
            user_input,
            "Hyy! I am Parshant's personal AI assistant. How can I help you?",
            "greeting-rule"
        )


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

    print("INPUT:", user_input)
    print("PROCESSED:", processed)
    print("INTENT:", intent)
    print("PROBABILITY:", max_prob)

    # -------- SMART FALLBACK WITH SUGGESTION --------
    if max_prob < 0.2:
        last_suggested_intent = intent

        example_question = next(
            i["patterns"][0]
            for i in data["intents"]
            if i["tag"] == intent
        )

        return no_cache_response({
            "reply": f"I’m not fully sure 🤔 Did you mean: \"{example_question}\" ?",
            "intent": "clarification",
            "suggested_intent": intent
        })

    # -------- NORMAL RESPONSE --------
    if intent in intent_response_map:
        return respond_with_history(
    user_input,
    intent_response_map[intent],
    intent
)


    return no_cache_response({"reply": "Sorry, I didn’t understand that."})

# -------- Run Server --------
if __name__ == "__main__":
    app.run(debug=True)
