import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Streamlit Page Setup
st.set_page_config(
    page_title="Salary Prediction App", 
    page_icon="💼", 
    layout="centered"
)

@st.cache_resource
def load_model():
    """Loads the linear_model.pkl file."""
    with open('linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load Model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'linear_model.pkl': {e}")
    st.stop()

# Title & Description
st.title("💼 Salary Estimator")
st.write("Predict estimated salary or outcome based on years of experience using your trained Linear Regression model.")

st.markdown("---")

# Input Widget
years_experience = st.number_input(
    label="Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.5,
    step=0.5
)

# Predict Button
if st.button("Predict Salary", use_container_width=True):
    # Format input as DataFrame matching feature name 'YearsExperience'
    input_data = pd.DataFrame([[years_experience]], columns=['YearsExperience'])
    
    try:
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.metric(
            label="Estimated Prediction", 
            value=f"${prediction:,.2f}"
        )
        st.success("Prediction generated successfully!")
        
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")
