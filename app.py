import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️")

# Title
st.title("🌦️ Weather Prediction App")
st.write("Predict future temperatures using Linear Regression")

# Load dataset
df = pd.read_csv("weather.csv")

# Display dataset
st.subheader("Weather Dataset")
st.dataframe(df)

# Features and target
X = df[["Day"]]
y = df["Temperature"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction section
st.subheader("Temperature Prediction")

future_day = st.number_input(
    "Enter Day Number",
    min_value=1,
    value=10,
    step=1
)

if st.button("Predict"):
    prediction = model.predict([[future_day]])
    st.success(
        f"Predicted Temperature on Day {future_day}: {prediction[0]:.2f}°C"
    )

# Graph
st.subheader("Temperature Trend")

fig, ax = plt.subplots()

ax.scatter(df["Day"], df["Temperature"], label="Actual Data")
ax.plot(df["Day"], model.predict(X), label="Prediction Line")

ax.set_xlabel("Day")
ax.set_ylabel("Temperature")
ax.set_title("Weather Prediction")
ax.legend()

st.pyplot(fig)

