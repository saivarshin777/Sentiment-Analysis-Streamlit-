import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle

nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("imdb_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("imdb_tfidf_vectorizer.pkl", "rb"))


