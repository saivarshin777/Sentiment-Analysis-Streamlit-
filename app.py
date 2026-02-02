import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------- Load Model ----------------
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("imdb_tfidf_vectorizer.pkl", "rb"))

# ---------------- Text Cleaning ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 3rem;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}
h1 {
    text-align: center;
    color: #ffffff;
    font-weight: 700;
}
p {
    text-align: center;
    color: #dcdcdc;
    font-size: 16px;
}
textarea {
    border-radius: 12px !important;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 30px;
    padding: 0.6rem 2rem;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("## 🎬 IMDB Movie Review Sentiment Analysis")
st.markdown(
    "<p>Type a movie review below and instantly find out whether the sentiment is <b>Positive</b> or <b>Negative</b>.</p>",
    unsafe_allow_html=True
)

review = st.text_area("✍️ Your Review", height=150, placeholder="Example: I absolutely loved this movie...")

if st.button("✨ Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.markdown("<hr>", unsafe_allow_html=True)

        if prediction == 1:
            st.success("💚 **Positive Sentiment** — Looks like people would enjoy this movie!")
        else:
            st.error("❤️‍🔥 **Negative Sentiment** — This review expresses dissatisfaction.")

st.markdown("</div>", unsafe_allow_html=True)
