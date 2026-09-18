import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("electricity_bill.pkl")
poly = joblib.load("polynomial_features.pkl")

# Page title
st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict your electricity bill based on AC and Fan consumption "
    "using Polynomial Regression."
)

# AC Units
ac_units = st.number_input(
    "Enter AC Consumption (AC Units)",
    min_value=0.0,
    max_value=150.0,
    value=50.0,
    step=1.0
)

# Fan Units
fan_units = st.number_input(
    "Enter Fan Consumption (Fan Units)",
    min_value=0.0,
    max_value=150.0,
    value=50.0,
    step=1.0
)

# Prediction
if st.button("Predict Bill"):

    new_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })

    # Polynomial transformation
    new_data_poly = poly.transform(new_data)

    # Prediction
    prediction = model.predict(new_data_poly)

    st.success("Model predicted successfully!")

    st.metric(
        "Predicted Electricity Bill",
        f"₹{prediction[0]:,.2f}"
    )
