import os
import requests
import streamlit as st

# Hugging Face Inference API
def get_model_prediction(text):
    API_URL = "https://api-inference.huggingface.co/models/abhishek/fake-news-detection"
    headers = {"Authorization": "Bearer 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"}  # Replace with your token
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    result = response.json()

    # Handle errors and empty responses
    if isinstance(result, dict) and "error" in result:
        return "Error", 0.0
    if isinstance(result, list) and len(result) > 0:
        label = result[0].get("label", "Unknown")
        score = result[0].get("score", 0.0)
        return label, score
    return "Unknown", 0.0

# Google Fact Check API
FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACTCHECK_KEY = "AIzaSyDmdUxpYeu7Wf-dGLnN48GpkuM2m8v6-LQ"  # Replace with your key

# NewsAPI
NEWSAPI_KEY = "6024b58d5e4549dbaccd2d49cd473cea"  # Replace with your key

def check_fact_with_google(query):
    params = {"query": query, "key": FACTCHECK_KEY}
    response = requests.get(FACTCHECK_API, params=params)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data:
            return data["claims"]
    return None

def check_with_newsapi(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWSAPI_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# --- UI Header ---
st.markdown(
    """
    <style>
    .title { font-size: 40px; font-weight: bold; color: #2E86C1; text-align: center; }
    .subtitle { font-size: 18px; color: #555; text-align: center; }
    </style>
    <div class="title">RealityCheck ✅</div>
    <div class="subtitle">AI‑powered Fake News Detection + Fact Verification</div>
    """,
    unsafe_allow_html=True
)

# --- Input ---
user_input = st.text_area("Enter news text:")

# --- Prediction + Fact Check + NewsAPI ---
if st.button("Check"):
    if user_input.strip():
        label, score = get_model_prediction(user_input)

        # Fact check results
        claims = check_fact_with_google(user_input)

        # NewsAPI results
        articles = check_with_newsapi(user_input)

        # --- Unified Verdict Logic ---
        if label == "Fake News":
            if articles:
                st.warning(f"⚠️ AI flagged this as Fake News (confidence: {score:.2f}), "
                           f"but recent headlines suggest it may be real.")
            elif claims:
                st.warning(f"⚠️ AI flagged this as Fake News (confidence: {score:.2f}), "
                           f"but fact-check sources provide context below.")
            else:
                st.error(f"❌ RealityCheck Verdict: Fake News (confidence: {score:.2f})")
        elif label == "Real News":
            st.success(f"✅ RealityCheck Verdict: Real News (confidence: {score:.2f})")
        elif label in ["Unknown", "Error"]:
            st.info("🤔 RealityCheck couldn’t verify this claim. Try rephrasing or check back later.")
        else:
            st.info("🤔 Unable to determine verdict. Please try again.")

        # Fact Check Results
        st.subheader("🔎 Fact Check Results")
        if claims:
            for c in claims:
                review = c.get("claimReview", [])
                if review:
                    st.write(f"- Source: {review[0]['publisher']['name']}")
                    st.write(f"  Rating: {review[0]['textualRating']}")
                    st.write(f"  [Read more]({review[0]['url']})")
        else:
            st.info("No fact‑check results found for this claim.")

        # NewsAPI Results
        st.subheader("📰 Recent News Mentions")
        if articles:
            for article in articles:
                st.markdown(f"**[{article['title']}]({article['url']})**  \n"
                            f"*Source: {article['source']['name']} | Published: {article['publishedAt'][:10]}*")
        else:
            st.info("No recent news articles found for this claim.")
    else:
        st.warning("Please enter some text.")

# --- Disclaimer ---
st.markdown(
    """
    <div style="
        background-color:#000000;
        padding:14px;
        border-radius:6px;
        border:2px solid #000000;
        color:#ffffff;
        font-size:15px;
        text-align:center;
        margin-top:30px;
    ">
    ⚠️ <b>Disclaimer:</b><br>
    RealityCheck is experimental. AI predictions are based on limited training data and may misclassify current events.<br>
    Always verify information with trusted sources such as BBC, Reuters, or official statements.
    </div>
    """,
    unsafe_allow_html=True
)
