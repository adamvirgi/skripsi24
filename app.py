import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
with open('svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    ss_train_test = pickle.load(f)

# Define the application title and header
st.title('SVM Classifier Web App')
st.header('Please fill in the details below to get the prediction')

# Get user input
gender = st.selectbox('Gender', ('Male', 'Female'))
age = st.number_input('Age in Years')
height = st.number_input('Height in cm')
weight = st.number_input('Weight in kg')

# Encode the gender
if gender == 'Male':
    gender_encoded = 1
else:
    gender_encoded = 0

# Create a user input array
user_input = np.array([[gender_encoded, age, height, weight]])

# Standardize the user input
user_input_scaled = ss_train_test.transform(user_input)

# Predict the outcome
prediction = svm_classifier.predict(user_input_scaled)[0]

# Print the prediction
if st.button('Predict'):
    if prediction == 1:
        st.write('The person is underweight')
    elif prediction == 2:
        st.write('The person is normal weight')
    elif prediction == 3:
        st.write('The person is overweight')
    else:
        st.write('The person is obese')
