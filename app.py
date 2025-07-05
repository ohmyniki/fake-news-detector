
import streamlit as st
import pandas as pd
import re
import string
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess input
def preprocess(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# UI
st.title("📰 Fake News Detector")

headline = st.text_input("Enter a news headline:")

if st.button("Check"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        processed = preprocess(headline)
        vec_input = vectorizer.transform([processed])
        result = model.predict(vec_input)[0]
        label = "🟢 Real News" if result == 1 else "🔴 Fake News"
        st.success(f"Prediction: {label}")
