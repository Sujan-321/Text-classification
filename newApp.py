import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import streamlit as st


model = joblib.load("sentiment_analysis.pkl")

# predicting values
def predict(new_data):
    p = model.predict(new_data)
    return p


# taking the input as string from user
userText = st.text_input("Enter your text: ")


# input_data = np.array([userText])
input_data = {'userText': userText}


# converting the input_data into dataframe
input_data = pd.DataFrame(input_data)




# predict button 
if st.button("Predict"):
    pre = predict(input_data)
    # displaying the resutl
    st.write(f"The predicted flower species is : {pre[0]}")