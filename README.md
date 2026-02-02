🎬 IMDB Sentiment Analysis Web App
A complete end-to-end NLP project that analyzes movie reviews and predicts their sentiment (Positive / Negative) using Machine Learning and a Streamlit web application.

🚀 Project Overview
This project builds an AI system that understands human language and classifies movie reviews based on sentiment.
It covers the entire NLP pipeline — from text preprocessing and model training to real-time web deployment.
Users can enter any movie review and instantly get a sentiment prediction.

🎯 Objective
Clean and preprocess raw text data
Convert text into numerical features using TF-IDF
Train a Logistic Regression model for sentiment classification
Deploy the trained model using Streamlit
Create a user-friendly web interface for real-time predictions

📊 Dataset
IMDB Movie Reviews Dataset
50,000 labeled reviews
Classes:
Positive
Negative

🧠 Technologies Used
Python
Natural Language Processing (NLP)
Scikit-learn
NLTK
Streamlit
Pandas & NumPy

⚙️ Machine Learning Pipeline
Data Preprocessing
Lowercasing text
Removing punctuation & special characters
Stopword removal
Lemmatization
Feature Extraction
TF-IDF Vectorization
Unigrams + Bigrams (ngram_range=(1,2))
Model Training
Logistic Regression
Class balancing to handle bias
Evaluation
Accuracy
Precision, Recall, F1-Score
Deployment
Streamlit web app
Publicly accessible link

🖥️ Web App Features
Clean & interactive UI
Real-time sentiment prediction
Emoji-based output for better user experience
Easy to use for non-technical users

📁 Project Structure
imdb-sentiment-streamlit/
│
├── app.py
├── requirements.txt
├── imdb_sentiment_model.pkl
├── imdb_tfidf_vectorizer.pkl
└── README.md

▶️ How to Run Locally
Clone the repository
git clone https://github.com/your-username/imdb-sentiment-streamlit.git
cd imdb-sentiment-streamlit

Install dependencies
pip install -r requirements.txt

Run the Streamlit app
streamlit run app.py

🌐 Live Deployment

🔗 Live App Link:
👉 (Add your Streamlit Cloud link here after deployment)

📈 Results
Achieved high accuracy on IMDB dataset
Successfully predicts sentiment for unseen reviews
Demonstrates a full real-world NLP application

🎓 Learning Outcomes
Hands-on experience with NLP preprocessing
Understanding of text vectorization techniques
Practical model deployment skills
Experience building AI-powered web applications

🙌 Acknowledgements
IMDB Dataset
Scikit-learn & NLTK documentation
Streamlit Community

👩‍💻 Author
Varshini
AIML Student | NLP Enthusiast