# 📰 Fake News Detector

A Python-based application for detecting fake news using natural language processing (NLP) and machine learning (ML). Features a Streamlit web interface, sample data, and supports logistic regression and random forest classifiers.

## Features

- Preprocessing text data (tokenization, lemmatization, removal of punctuation and stopwords)
- TF-IDF vectorization
- Binary classification: Fake vs. Real news
- Supports Logistic Regression and Random Forest models
- Interactive web app interface (Streamlit)
- Model confidence and probability output
- Sample news data for quick demo

## Quick Demo

You can run the included Streamlit app to interactively classify news headlines or content.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/kartikey-86/fake-news-detector.git
    cd fake-news-detector
    ```

2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run fake_news_detector.py
    ```

4. **Use the Web Interface:**
   - Enter a news headline or article in the textbox.
   - Click "Predict" to see if it's Real or Fake with model confidence.

## File Overview

- `fake_news_detector.py`: Main codebase. Contains:
  - Data preprocessing
  - Model selection and training
  - Prediction function
  - Sample dataset
  - Streamlit web UI

## Model Details

- **Preprocessing:** Cleans data by lowercasing, removing punctuation, numbers, and stopwords, and lemmatizing words.
- **Vectorization:** Converts text using TF-IDF with up to 5000 features and bi-grams.
- **Classification:** Logistic Regression or Random Forest (default: Logistic Regression).

## Sample Data

Built-in diverse headlines with both real and fake news labels for instant demo use.


## License

MIT
