import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
st.title('Passenger Survival Prediction')
st.subheader('Upload a CSV file to check survival status')

try:
    with open('model.pkl', 'rb') as file:
        load_model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file ('model.pkl') not found. Ensure the file is in the same directory.")
    st.stop()

# Load feature columns
try:
    df = pd.read_csv("dependent_feature.csv")
    column_list = df.columns.to_list()
except FileNotFoundError:
    st.error("Error: 'dependent_feature.csv' not found. Ensure the file is in the same directory.")
    column_list = []

# File uploader
up_file = st.file_uploader('Upload a CSV file', type=['csv'])

if up_file is not None:
    try:
        # Read uploaded CSV
        df1 = pd.read_csv(up_file)

        # Check if required columns exist in uploaded file
        missing_columns = [col for col in column_list if col not in df1.columns]
        if missing_columns:
            st.error(f"Missing columns in uploaded file: {missing_columns}")
            st.stop()

        # Reorder and fill missing columns with 0 (if any)
        df1 = df1.reindex(columns=column_list, fill_value=0)

        # Make prediction
        prediction = load_model.predict(df1)

        # Convert numerical predictions to text labels
        prediction_text = np.where(prediction == 1, 'Survived', 'Did Not Survive')

        # Display results
        st.subheader('Prediction Results:')
        df1['Survival Prediction'] = prediction_text
        st.write(df1[['Survival Prediction']])  # Show only predictions

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

