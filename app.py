import os
import requests
import streamlit as st
from transformers import pipeline

os.environ["STREAMLIT_PANDAS"] = "1"

# ✅ Use a stable model that works with pipeline
try:
    model = pipeline("text-classification", model="Pulk17/Fake-News-Detection")
    MODEL_NAME = "Pulk17/Fake-News-Detection"
except Exception:
    st.warning("⚠️ Primary model failed to load. Falling back to backup model.")
    model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2")
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2"

label_map = {
    "LABEL_0": "Real News",
    "LABEL_1": "Fake News",
    "NEGATIVE": "Fake News",
    "POSITIVE": "Real News"
}

FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = "AIzaSyDmdUxpYeu7Wf-dGLnN48GpkuM2m8v6-LQ"

def check_fact_with_google(query):
    params = {"query": query, "key": API_KEY}
    response = requests.get(FACTCHECK_API, params=params)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"]
    return None

st.title("📰 Fake News Detector + Fact Check")
st.write(f"Currently using model: **{MODEL_NAME}**")

user_input = st.text_area("Enter news text:")

if st.button("Check"):
    if user_input.strip():
        result = model(user_input)[0]
        raw_label = result['label']
        label = label_map.get(raw_label, raw_label)
        score = result['score']

        if score < 0.7:
            st.info(f"AI Prediction: Uncertain — please verify (confidence: {score:.2f})")
        else:
            st.success(f"AI Prediction: {label} (confidence: {score:.2f})")

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

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is experimental. AI predictions are not authoritative. "
           "Always verify information with trusted sources and fact‑checker verdicts.")







