import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Iris Flower AI Predictor",
    page_icon="🌸",
    layout="centered"
)

# Load Model
model = joblib.load("models/model.pkl")

# Custom CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #00c6ff, #0065ff);
    color: white;
}

/* Main Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

/* Card Design */
.card {
    background: rgba(255,255,255,0.08);
    padding: 35px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
}

/* Input Box Styling */
.stNumberInput input {
    background-color: #f8fafc !important;
    color: black !important;
    border-radius: 10px !important;
    height: 45px !important;
    font-size: 16px !important;
}

/* Labels */
label {
    color: white !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 52px;
    background: linear-gradient(to right, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    margin-top: 20px;
}

.stButton>button:hover {
    opacity: 0.9;
}

/* Result Box */
.result-box {
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: white;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">🌸 Iris Flower AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">      Machine Learning Classification System</div>',
    unsafe_allow_html=True
)

# Main Card

st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        format="%.2f"
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        format="%.2f"
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        format="%.2f"
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        format="%.2f"
    )

# Predict Button
if st.button("Predict Flower"):

    features = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(features)

    flower_names = {
        0: "🌼 Setosa",
        1: "🌺 Versicolor",
        2: "🌷 Virginica"
    }

    result = flower_names[int(prediction[0])]

    st.markdown(
        f'<div class="result-box">Prediction: {result}</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Streamlit • Scikit-learn • Machine Learning")