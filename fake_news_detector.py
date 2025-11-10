import pandas as pd
import numpy as np
import re
import string
import logging
from typing import Union, Optional, Tuple, Dict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources safely
def _setup_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

_setup_nltk_resources()

class DetectorConfig:
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    LOGISTIC_C = 0.1
    LOGISTIC_MAX_ITER = 1000
    RF_N_ESTIMATORS = 200

class FakeNewsDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.vectorizer = TfidfVectorizer(max_features=self.config.MAX_FEATURES, stop_words='english', ngram_range=self.config.NGRAM_RANGE)
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._is_trained = False
        logger.info("FakeNewsDetector initialized")

    def preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words]
        return ' '.join(words)

    def create_sample_data(self) -> pd.DataFrame:
        data = {
            'title': [
                'Breaking: New Study Shows Coffee Is Healthy',
                'Scientists Discover Cure for Common Cold',
                'President Makes Historic Climate Agreement',
                'Trump Wins Nobel Prize in Physics',
                'New COVID Vaccine Approved by FDA',
                'Aliens Land in Times Square',
                'Local Mayor Cuts Ribbon on New Bridge',
                'Doctors Warn About Dangerous Health Trend',
                'Celebrity Dies in Freak Accident',
                'Giant Meteor to Hit Earth Next Month',
                'New Study: Exercise Improves Mental Health',
                'Government Announces Tax Cut Program',
                'Flat Earth Proven by Scientists',
                'Moon Landing Was Fake',
                'New Tech Company Raises $100M',
                'Birds Are Actually Government Drones'
            ],
            'text': [
                'A recent study from Harvard shows that moderate coffee consumption improves health...',
                'Researchers claim to have found a cure for the common cold after years of research...',
                'World leaders gathered to sign a historic climate agreement addressing global warming...',
                'In a shocking turn of events, Donald Trump won the Nobel Prize in Physics...',
                'Health officials announced that the new COVID vaccine has been approved by FDA...',
                'Thousands of witnesses reported seeing UFOs land in Times Square yesterday...',
                'The mayor cut the ribbon on a newly renovated bridge in the city center...',
                'Medical professionals warn about a new dangerous health trend spreading online...',
                'A famous actor died in an unexpected accident while filming a movie...',
                'Astronomers claim a massive meteor is headed toward Earth next month...',
                'A comprehensive study shows that regular exercise improves mental health significantly...',
                'The government announced a new tax cut program for all citizens starting next year...',
                'A group of scientists have finally proven the Earth is flat through experiments...',
                'Evidence suggests the moon landing never actually happened and was filmed in studio...',
                'A promising new tech startup raised $100 million in Series A funding round...',
                'Recent investigations suggest birds are actually government surveillance drones...'
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
        }
        return pd.DataFrame(data)

    def prepare_and_train(self, df: pd.DataFrame, title_col='title', text_col='text', label_col='label', model_type='logistic') -> Dict[str, float]:
        df['content'] = df[title_col] + ' ' + df[text_col]
        X_train, X_test, y_train, y_test = train_test_split(
            df['content'], df[label_col], test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=df[label_col]
        )
        X_train_processed = X_train.apply(self.preprocess)
        X_test_processed = X_test.apply(self.preprocess)
        self.X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        self.X_test_tfidf = self.vectorizer.transform(X_test_processed)
        self.y_train, self.y_test = y_train, y_test

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(self.X_train_tfidf, self.y_train)
        self._is_trained = True
        logger.info("Training complete")

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        processed = self.preprocess(text)
        vec = self.vectorizer.transform([processed])
        pred = self.model.predict(vec)[0]
        prob = self.model.predict_proba(vec)[0]
        return {'label': 'FAKE' if pred == 1 else 'REAL', 'confidence': f"{max(prob)*100:.2f}%", 'probabilities': {'real': f"{prob[0]*100:.2f}%", 'fake': f"{prob[1]*100:.2f}%"}}

# Streamlit UI
def run_streamlit_app():
    st.title("📰 Fake News Detector")
    st.write("Enter a news headline or article to classify it as Real or Fake.")

    if 'detector' not in st.session_state:
        st.session_state.detector = FakeNewsDetector()
        df = st.session_state.detector.create_sample_data()
        st.session_state.detector.prepare_and_train(df)

    user_input = st.text_area("Enter News Text:", height=150)

    if st.button("Predict"):
        if user_input.strip():
            result = st.session_state.detector.predict(user_input)
            st.subheader(f"Prediction: {result['label']}")
            st.write(f"Confidence: {result['confidence']}")
            st.json(result['probabilities'])
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    run_streamlit_app()