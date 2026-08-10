import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Salary Analytics Portal", 
    page_icon="⚡", 
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Loads the linear_model.pkl file."""
    with open('linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'linear_model.pkl': {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">⚡ Salary & Career Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Linear Regression Model Deployment</div>', unsafe_allow_html=True)

# Create Tabs
tab1, tab2 = st.tabs(["🎯 Interactive Predictor", "📊 Model Insights"])

with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Input Parameters")
        st.write("Adjust experience level to calculate expected baseline compensation.")
        
        years_exp = st.slider(
            "Years of Experience",
            min_value=0.0,
            max_value=25.0,
            value=3.5,
            step=0.5,
            help="Select total relevant industry experience in years."
        )
        
        input_df = pd.DataFrame([[years_exp]], columns=['YearsExperience'])
        predicted_val = model.predict(input_df)[0]
        
    with col_result:
        st.subheader("Prediction")
        st.metric(
            label="Estimated Base Salary",
            value=f"${predicted_val:,.2f}",
            delta=f"+${model.coef_[0]:,.2f} / year"
        )
        st.info("💡 **Insight:** The model projects constant growth based on past linear regression fits.")

with tab2:
    st.subheader("Regression Curve Visualization")
    
    # Generate linear curve data using model parameters
    x_range = np.linspace(0, 20, 100)
    x_df = pd.DataFrame(x_range, columns=['YearsExperience'])
    y_range = model.predict(x_df)
    
    fig = go.Figure()
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range, 
        mode='lines', 
        name='Regression Line',
        line=dict(color='#2563EB', width=3)
    ))
    
    # Highlight current selected input point
    fig.add_trace(go.Scatter(
        x=[years_exp], 
        y=[predicted_val], 
        mode='markers', 
        name='Current Prediction',
        marker=dict(color='#EF4444', size=12, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Experience vs Salary Projection",
        xaxis_title="Years of Experience",
        yaxis_title="Salary ($)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model parameters summary
    col_a, col_b = st.columns(2)
    col_a.write(f"**Base Intercept:** ${model.intercept_:,.2f}")
    col_b.write(f"**Slope (Coefficient):** ${model.coef_[0]:,.2f} per year")
