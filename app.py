import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load model & vectorizer
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("imdb_tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and find out whether it is **Positive or Negative**.")

review = st.text_area("✍️ Enter your review here:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")
