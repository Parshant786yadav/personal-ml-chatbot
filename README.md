# 🤖 Chat Me — Personal ML Chatbot

**Chat Me** is a custom **Machine Learning–based chatbot** trained on my own structured data.  
It is integrated directly into my personal website and answers questions about me in real time.

This project focuses on **data training, intent classification, and frontend–backend integration**.

---

## 🧠 Training & Data

- Chatbot is trained using a **custom intents dataset**
- Each intent contains:
  - Multiple question variations
  - A predefined response
- Covers topics like:
  - About me
  - Education & qualification
  - Skills & technologies
  - Projects
  - Experience
  - Current work

The model learns to understand **different ways of asking the same question**, not just exact matches.

---

## ✨ Smart Understanding

- Handles **spelling mistakes** and name variations
- Uses **fuzzy matching** before prediction
- Predicts intent using an ML classifier
- Applies confidence checks:
  - High confidence → direct answer
  - Low confidence → clarification suggestion

---

## 🔌 Frontend Integration

- Frontend sends user messages to the backend API
- Backend processes the message and returns a response
- Response is displayed instantly in the chat panel

The chatbot runs completely on **my own backend**, not third-party AI APIs.

---

## 🌐 Live API

- Backend exposed as a **REST API**
- Deployed and publicly accessible
- Frontend communicates using `fetch()` requests
- CORS enabled for secure cross-origin access

---

## 🛠 Tech Stack

- **Backend:** Python, Flask, Scikit-learn
- **ML:** Intent classification model
- **Text Processing:** NLTK, RapidFuzz
- **Deployment:** Github, Render

---

## 🎯 Why This Project?

- Shows how training data drives chatbot behavior
- Demonstrates ML + backend integration
- Highlights real-world frontend–backend communication
- Fully owned, explainable, and customizable system

---

✨ *A practical example of turning trained data into a live AI feature on a website.*
