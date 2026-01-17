import json
import os
import nltk
import numpy as np
import pickle
import string

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

stemmer = PorterStemmer()

# -------- Text Preprocessing --------
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(sentence)
    stemmed = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed)

# -------- Load Dataset (SAFE PATH) --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "intents.json")

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(preprocess(pattern))
        labels.append(intent["tag"])

# -------- Vectorization --------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# -------- Train Model --------
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# -------- Save Model --------
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "chatbot_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("✅ Model trained and saved successfully!")
print("📁 Model location:", model_path)