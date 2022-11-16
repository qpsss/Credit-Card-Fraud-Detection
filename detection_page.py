import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

logModel = data["model"]
X_test = data["X_test"]
y_test = data["y_test"]

def show_detection_page():
    st.title("Credit Card Fraud Detection")

    st.write("""### Select a transaction""")

    i = st.slider("#", 0, 20, 7)
    
    if i:
        trueResult = y_test.iloc[i]
        if trueResult == 1:
            trueResult = "FRUAD"
        elif trueResult == 0:
            trueResult = "NOT FRUAD"
        st.subheader(f'This transaction is {trueResult}')
        st.dataframe(X_test.iloc[i], use_container_width=True)
        
    ok = st.button("Detect")
    if ok:
        result = logModel.predict(X_test.iloc[i].values.reshape(1, -1))
        
        if result == 1:
            result = "FRUAD"
        elif result == 0:
            result = "NOT FRUAD"
        
        st.subheader(f'Detected: {result}')


    