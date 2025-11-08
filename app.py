import os
import streamlit as st
from transformers import pipeline

# Force Streamlit to use pandas instead of pyarrow
os.environ["STREAMLIT_PANDAS"] = "1"

model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

st.title("📰 Fake News Detector")
st.write("Paste a headline or article below to check if it's fake or real.")

user_input = st.text_area("Enter news text:")

if st.button("Check"):
    if user_input.strip():
        result = model(user_input)[0]
        st.success(f"Prediction: {result['label']} (confidence: {result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
