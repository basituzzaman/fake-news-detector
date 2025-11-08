import os
import requests
import streamlit as st
from transformers import pipeline

# Force Streamlit to use pandas instead of pyarrow
os.environ["STREAMLIT_PANDAS"] = "1"

# ✅ Primary model (verified fake-news detector)
try:
    model = pipeline("text-classification", model="Pulk17/Fake-News-Detection")
    MODEL_NAME = "Pulk17/Fake-News-Detection"
except Exception:
    st.warning("⚠️ Primary model failed to load. Falling back to backup model.")
    model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2")
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2"

# Map raw model labels to human-friendly text
label_map = {
    "LABEL_0": "Real News",
    "LABEL_1": "Fake News",
    "NEGATIVE": "Fake News",
    "POSITIVE": "Real News"
}

# Google Fact Check API endpoint
FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = "AIzaSyDmdUxpYeu7Wf-dGLnN48GpkuM2m8v6-LQ"  # <-- your key

def check_fact_with_google(query):
    """Query Google Fact Check Tools API for fact-check results."""
    params = {"query": query, "key": API_KEY}
    response = requests.get(FACTCHECK_API, params=params)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"]
    return None

# Website title
st.title("📰 Fake News Detector + Fact Check")
st.write(f"Currently using model: **{MODEL_NAME}**")

# Text input box
user_input = st.text_area("Enter news text:")

# Button to run detection
if st.button("Check"):
    if user_input.strip():
        # Run AI model
        result = model(user_input)[0]
        raw_label = result['label']
        label = label_map.get(raw_label, raw_label)
        score = result['score']

        # Show AI prediction
        if score < 0.7:
            st.info(f"AI Prediction: Uncertain — please verify (confidence: {score:.2f})")
        else:
            st.success(f"AI Prediction: {label} (confidence: {score:.2f})")

        # Confidence breakdown
        st.subheader("Confidence Breakdown")
        st.bar_chart({
            "Confidence": {
                "Real News": result['score'] if label == "Real News" else 1 - result['score'],
                "Fake News": result['score'] if label == "Fake News" else 1 - result['score']
            }
        })

        # Fact check results
        st.subheader("🔎 Fact Check Results")
        claims = check_fact_with_google(user_input)
        if claims:
            for c in claims:
                review = c.get("claimReview", [])
                if review:
                    publisher = review[0]['publisher']['name']
                    rating = review[0].get('textualRating', 'No rating provided')
                    url = review[0].get('url', '')
                    st.write(f"- Source: {publisher}")
                    st.write(f"  Verdict: {rating}")
                    if url:
                        st.write(f"  [Read more]({url})")
        else:
            st.info("No fact‑check results found for this claim.")
    else:
        st.warning("Please enter some text.")

# Disclaimer
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is experimental. AI predictions are not authoritative. "
           "Always verify information with trusted sources and fact‑checker verdicts.")








