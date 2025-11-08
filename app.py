import os
import streamlit as st
from transformers import pipeline

# Force Streamlit to use pandas instead of pyarrow
os.environ["STREAMLIT_PANDAS"] = "1"

# Load a stronger fake news detection model from Hugging Face
model = pipeline("text-classification", model="Pulk17/Fake-News-Detection")

# Map raw model labels to human-friendly text
label_map = {
    "LABEL_0": "Real News",
    "LABEL_1": "Fake News"
}

# Website title
st.title("📰 Fake News Detector")
st.write("Paste a headline or article below to check if it's fake or real.")

# Text input box
user_input = st.text_area("Enter news text:")

# Button to run detection
if st.button("Check"):
    if user_input.strip():
        result = model(user_input)[0]
        raw_label = result['label']
        label = label_map.get(raw_label, raw_label)
        score = result['score']

        # Add "Uncertain" category if confidence is low
        if score < 0.7:
            st.info(f"Prediction: Uncertain — please verify with trusted sources (confidence: {score:.2f})")
        else:
            st.success(f"Prediction: {label} (confidence: {score:.2f})")

        # Confidence bar chart
        st.subheader("Confidence Breakdown")
        st.bar_chart({
            "Confidence": {
                "Real News": result['score'] if raw_label == "LABEL_0" else 1 - result['score'],
                "Fake News": result['score'] if raw_label == "LABEL_1" else 1 - result['score']
            }
        })
    else:
        st.warning("Please enter some text.")

# Disclaimer section
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is experimental. No detector is 100% accurate. "
           "Always verify information with trusted sources such as BBC, Reuters, or official statements.")


