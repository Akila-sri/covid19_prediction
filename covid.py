import streamlit as st
import pandas as pd
import numpy as np
import pickle  # or use joblib
import matplotlib.pyplot as plt

# Load the trained model
model_filename = "covid_rf_model.pkl"  # Change filename if needed
with open(model_filename, "rb") as file:
    model = pickle.load(file)
    

# Streamlit App Title
st.title("📊 COVID-19 Cases Prediction App")

# Sidebar User Input
st.sidebar.header("User Input for Prediction")
days_since_start = st.sidebar.number_input("Days Since Start of Pandemic", min_value=0, max_value=1000, step=1)

if st.sidebar.button("Predict"):
    # Convert input into DataFrame (as expected by model)
    input_data = pd.DataFrame({'days_since_start': [days_since_start]})
    
    # Make Prediction
    prediction = model.predict(input_data)
    
    # Display Output
    st.subheader("📍 Prediction")
    st.write(f"Predicted COVID-19 Cases after {days_since_start} days: **{int(prediction[0]):,} cases**")

    # Generate Future Predictions
    future_days = np.arange(days_since_start, days_since_start + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)

    # Plot Predictions
    st.subheader("📈 Future COVID-19 Cases Trend")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(future_days, future_predictions, marker='o', linestyle='--', color='red', label="Predicted Cases")
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Total Cases")
    ax.set_title("Predicted COVID-19 Cases Trend")
    ax.legend()
    st.pyplot(fig)

st.sidebar.info("📌 Adjust the slider and click **Predict** to see future COVID-19 cases.")
