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
API_KEY = "YOUR_GOOGLE_API_KEY"  # replace with your key

def check_fact_with_google(query):
    """Query Google Fact Check Tools API for fact-check results."""
    params = {"query": query, "key": API_KEY}
    response = requests.get(FACTCHECK_API, params=params)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"]
    return None

# --- Custom Header ---
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
    }
    </style>
    <div class="title">RealityCheck ✅</div>
    <div class="subtitle">AI‑powered Fake News Detection + Fact Verification</div>
    """,
    unsafe_allow_html=True
)

# --- Input ---
user_input = st.text_area("Enter news text:")

# --- Prediction + Fact Check ---
if st.button("Check"):
    if user_input.strip():
        # Run AI model
        result = model(user_input)[0]
        raw_label = result['label']
        label = label_map.get(raw_label, raw_label)
        score = result['score']

        # Verdict with icons
        if score < 0.7:
            st.warning(f"⚠️ Uncertain — please verify (confidence: {score:.2f})")
        elif label == "Real News":
            st.success(f"✅ RealityCheck Verdict: Real News (confidence: {score:.2f})")
        else:
            st.error(f"❌ RealityCheck Verdict: Fake News (confidence: {score:.2f})")

        # Confidence breakdown
        st.subheader("Confidence Breakdown")
        st.bar_chart({
            "Confidence": {
                "Real News": result['score'] if raw_label == "LABEL_1" else 1 - result['score'],
                "Fake News": result['score'] if raw_label == "LABEL_0" else 1 - result['score']
            }
        })

        # Fact check results
        st.subheader("🔎 Fact Check Results")
        claims = check_fact_with_google(user_input)
        if claims:
            for c in claims:
                review = c.get("claimReview", [])
                if review:
                    st.write(f"- Source: {review[0]['publisher']['name']}")
                    st.write(f"  Rating: {review[0]['textualRating']}")
                    st.write(f"  [Read more]({review[0]['url']})")
        else:
            st.info("No fact‑check results found for this claim.")
    else:
        st.warning("Please enter some text.")

# --- Disclaimer ---
st.markdown(
    """
    <div style="
        background-color:#ffffff; 
        padding:12px; 
        border-radius:6px; 
        border:2px solid #000000; 
        color:#000000;
        font-size:14px;
    ">
    ⚠️ <b>Disclaimer:</b> RealityCheck is experimental. No detector is 100% accurate. 
    Always verify information with trusted sources such as BBC, Reuters, or official statements.
    </div>
    """,
    unsafe_allow_html=True
)



