import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load("sentiment_analysis.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict sentiment
def predict(new_data):
    p = model.predict(new_data)
    return p

# Streamlit input
st.title("Sentiment Analysis Prediction")
userText = st.text_input("Enter your text:")

# Predict button
if st.button("Predict"):
    # Preprocess the input text using the vectorizer
    processed_text = vectorizer.transform([userText])
    
    # Predict sentiment
    prediction = predict(processed_text)
    
    # Map the prediction to sentiment
    sentiment = "positive" if prediction[0] == 1 else "negative"
    
    # Display the result
    st.write(f"The predicted sentiment is: {sentiment}")
