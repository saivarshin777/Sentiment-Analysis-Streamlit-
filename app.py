import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("imdb_tfidf_vectorizer.pkl
