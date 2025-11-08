import os
import requests
import streamlit as st
from transformers import pipeline

# Force Streamlit to use pandas instead of pyarrow
os.environ["STREAMLIT_PANDAS"] = "1"

# Load the fake news detection model
model = pipeline("text-classification", model="Pulk17/Fake-News-Detection")

# ✅ Correct label mapping
label_map = {
    "LABEL_0": "Fake News",
    "LABEL_1": "Real News"
}

# Google Fact Check API endpoint
FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = "AIzaSyDmdUxpYeu7Wf-dGLnN48GpkuM2m8v6-LQ"

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
st.title("RealityCheck — AI News Verifier")
st.write("Paste a headline or article below to check if it's fake or real.")

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

        # ✅ Fixed confidence breakdown
        st.subheader("Confidence Breakdown")
        st.bar_chart({
            "Confidence": {
                "Real News": result['score'] if raw_label == "LABEL_1" else 1 - result['score'],
                "Fake News": result['score'] if raw_label == "LABEL_0" else 1 - result['score']
            }
        })

        # Run fact check
        st.subheader("🔎 Fact Check Results")
        claims = check_fact_with_google(user_input)
        if claims:
            for c in claims:
                review = c.get("claimReview", [])
                if review:
                    st.write(f"- Source: {review[0]['publisher']['name']}")
                    st.write(f"  Rating: {review[0]['textualRating']}")
                    st.write(f"  URL: {review[0]['url']}")
        else:
            st.info("No fact‑check results found for this claim.")
    else:
        st.warning("Please enter some text.")

# Disclaimer section
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is experimental. No detector is 100% accurate. "
           "Always verify information with trusted sources such as BBC, Reuters, or official statements.")

