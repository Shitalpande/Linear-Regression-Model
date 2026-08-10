import streamlit as st
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Years Experience Predictor", page_icon="📈", layout="centered")

@st.cache_resource
def load_model():
    """Load the trained Linear Regression model."""
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Header Section
st.title("📈 Salary / Outcome Predictor")
st.write("Enter your **Years of Experience** to get a prediction from the trained Linear Regression model.")

st.markdown("---")

# User Input
years_exp = st.number_input(
    label="Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.5,
    step=0.5
)

# Prediction Trigger
if st.button("Calculate Prediction", use_container_width=True):
    # Reshape input for single-sample prediction
    input_data = np.array([[years_exp]])
    
    try:
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Results")
        st.success(f"**Predicted Value:** {prediction:,.2f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
