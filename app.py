import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords

# ---------------- NLTK SETUP ----------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("imdb_tfidf_vectorizer.pkl", "rb"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit header */
header {
    visibility: hidden;
}

/* Remove top padding */
.block-container {
    padding-top: 0.5rem !important;
}

/* 🔥 REMOVE EMPTY STREAMLIT BLOCK (THE GREY BAR) */
div[data-testid="stVerticalBlock"]:has(> div:empty) {
    display: none;
}

/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Main card */
.main {
    background-color: rgba(255, 255, 255, 0.06);
    padding: 3rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    margin-top: 0 !important;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff;
}

/* Subtitle */
p {
    text-align: center;
    color: #dddddd;
}

/* Text area */
textarea {
    border-radius: 14px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 30px;
    padding: 0.6rem 2.2rem;
    font-size: 16px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("## 🎬 IMDB Movie Review Sentiment Analysis")
st.markdown(
    "<p>Enter a movie review and instantly find out whether the sentiment is <b>Positive</b> or <b>Negative</b>.</p>",
    unsafe_allow_html=True
)

review = st.text_area(
    "✍️ Your Review",
    height=150,
    placeholder="Example: This movie was absolutely amazing..."
)

if st.button("✨ Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        cleaned_review = clean_text(review)
        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        st.markdown("<hr>", unsafe_allow_html=True)

        if prediction == 1:
            st.success("💚 **Positive Sentiment**")
        else:
            st.error("❤️‍🔥 **Negative Sentiment**")

st.markdown("</div>", unsafe_allow_html=True)
