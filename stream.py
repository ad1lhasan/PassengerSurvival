import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open('model.pkl', 'rb') as file:
    load_model = pickle.load(file)

st.title('Passenger Survival')
st.subheader('Survival Status')

df = pd.read_csv("dependent_feature.csv")

column_list = df.columns.to_list()

up_file = st.file_uploader('Upload a csv file',type = ['csv'])

if up_file is not None:
    df=pd.read_csv(up_file)
    df=df.reindex(columns=column_list,fill_value=0)
    prediction = load_model.predict(df)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('Survived:')
    st.write(prediction_text)
