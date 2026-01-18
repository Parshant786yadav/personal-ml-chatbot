import pickle
import os
import string
from flask import session
from rapidfuzz import fuzz
from flask import Flask, request, jsonify

INTENT_KEYWORDS = {
    "skills": {"skill", "skills", "technologies", "tech"},
    "qualification": {"qualification", "degree", "education", "course"},
    "projects": {"project", "projects", "work"},
    "experience": {"experience", "internship", "job"},
    "parshant" : {"prashant", "prshnt", "psnt", "prasant"}
}

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

# -------- Keyword-based Intent Recommendation --------
STOPWORDS = {
    "tell", "me", "about", "who", "is", "are", "the", "a", "an", "please"
}
LAST_WORD_WINDOW = 3
def recommend_intent_by_keywords(processed, intents):
    words = processed.split()

    # Take last N meaningful words
    meaningful_words = [w for w in words if w not in STOPWORDS]
    tail_words = meaningful_words[-LAST_WORD_WINDOW:]

    best_intent = None
    best_score = 0

    for intent in intents:
        tag = intent["tag"]
        patterns = intent["patterns"]

        score = 0

        for p in patterns:
            p_words = p.lower().split()

            # 🔥 reverse matching (last word gets highest weight)
            for idx, w in enumerate(reversed(tail_words)):
                if w in p_words:
                    score += (LAST_WORD_WINDOW - idx) * 3

        if score > best_score:
            best_score = score
            best_intent = tag

    return best_intent if best_score > 0 else None

def last_words_match_any_intent(processed, intents, window=3):
    words = processed.split()

    # remove very common filler words
    STOPWORDS = {"tell", "me", "about", "who", "is", "are", "the", "a", "an"}
    meaningful = [w for w in words if w not in STOPWORDS]

    tail_words = meaningful[-window:]

    for intent in intents:
        for pattern in intent["patterns"]:
            p_words = pattern.lower().split()
            if any(w in p_words for w in tail_words):
                return intent["tag"]

    return None


# -------- Response with History --------
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
        "history": history
    })

# -------- Chat API --------
@app.route("/chat", methods=["POST"])
def chat():
    global last_suggested_intent

    # Initialize history for this session
    if "history" not in session:
        session["history"] = []

    user_input = request.json.get("message", "")
    if not user_input:
        return no_cache_response({"reply": "Please type something."})

    processed = preprocess(user_input)

    # -------- Build intent → response map --------
    intent_response_map = {
        i["tag"]: i["responses"][0] for i in data["intents"]
    }

    # -------- USER CONFIRMATION HANDLING --------
    if processed in YES_WORDS and last_suggested_intent:
        reply = intent_response_map[last_suggested_intent]
        last_suggested_intent = None
        return respond_with_history(user_input, reply, "confirmation-yes")

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
        return respond_with_history(
            user_input,
            "Parshant is doing well and currently focusing on his studies and projects.",
            "rule_how_is"
        )
    
    

    # -------- ML STARTS HERE --------
    X = vectorizer.transform([processed])
    probs = model.predict_proba(X)[0]
    intent = model.classes_[probs.argmax()]
    max_prob = float(max(probs))

    print("INPUT:", user_input)
    print("PROCESSED:", processed)
    print("INTENT:", intent)
    print("PROBABILITY:", max_prob)

    # -------- SMART FALLBACK WITH KEYWORD-AWARE SUGGESTION --------
    if max_prob < 0.2:
        # 🔹 Use keyword logic ONLY for suggestion
        suggested_intent = recommend_intent_by_keywords(
            processed, data["intents"]
        )

        # fallback to ML intent if keyword logic finds nothing
        if not suggested_intent:
            suggested_intent = intent

        last_suggested_intent = suggested_intent

        example_question = next(
            i["patterns"][0]
            for i in data["intents"]
            if i["tag"] == suggested_intent
        )

        return respond_with_history(
            user_input,
            f"I’m not fully sure 🤔 Did you mean: \"{example_question}\" ?",
            "clarification"
        )


    # -------- NORMAL RESPONSE --------
    if intent in intent_response_map:
        return respond_with_history(
            user_input,
            intent_response_map[intent],
            intent
        )

    return respond_with_history(
        user_input,
        "Sorry, I didn’t understand that.",
        "fallback"

# -------- Run Server --------
    )
if __name__ == "__main__":
    app.run(debug=True)
