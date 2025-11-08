import os
import streamlit as st
from transformers import pipeline

# Force Streamlit to use pandas instead of pyarrow
os.environ["STREAMLIT_PANDAS"] = "1"

# ✅ Use a stable Hugging Face model that always loads
model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Website title
st.title("📰 Fake News Detector")
st.write("Paste a headline or article below to check if it's fake or real.")

# Text input box
user_input = st.text_area("Enter news text:")

# Button to run detection
if st.button("Check"):
    if user_input.strip():
        result = model(user_input)[0]
        label = result['label']
        score = result['score']
        st.success(f"Prediction: {label} (confidence: {score:.2f})")
    else:
        st.warning("Please enter some text.")
