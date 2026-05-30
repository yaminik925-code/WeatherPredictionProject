import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("weather.csv")

# App title
st.title("🌦 Weather Prediction App")

# Show data
st.write("Weather Dataset")
st.dataframe(df)

# Train model
X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

# User input
future_day = st.slider("Select Future Day", 16, 30, 20)

# Prediction
prediction = model.predict([[future_day]])

st.subheader("Prediction Result")
st.write(f"Predicted Temperature on Day {future_day}: {prediction[0]:.2f} °C")

# Graph
fig, ax = plt.subplots()
ax.scatter(df["Day"], df["Temperature"])
ax.plot(df["Day"], model.predict(X))
ax.set_xlabel("Day")
ax.set_ylabel("Temperature")
st.pyplot(fig)

