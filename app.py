import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Title
st.title("🌤️ Weather Prediction Dashboard")
st.markdown("### Predict future temperatures using Machine Learning")

# Weather Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Current Temp", "34°C")

with col2:
    st.metric("💧 Humidity", "65%")

with col3:
    st.metric("🌬️ Wind Speed", "12 km/h")

st.divider()

# Load Dataset
df = pd.read_csv("weather.csv")

# Show Dataset
st.subheader("📊 Weather Dataset")
st.dataframe(df, use_container_width=True)

# Train Model
X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

st.divider()

# Prediction Section
st.subheader("🌡️ Temperature Prediction")

future_day = st.number_input(
    "Enter Day Number",
    min_value=1,
    value=10
)

if st.button("Predict Temperature"):
    prediction = model.predict([[future_day]])

    st.success(
        f"Predicted Temperature on Day {future_day}: {prediction[0]:.2f} °C"
    )

st.divider()

# Graph Section
st.subheader("📈 Temperature Trend")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Day"],
    df["Temperature"],
    label="Actual Data"
)

ax.plot(
    df["Day"],
    model.predict(X),
    label="Prediction Line"
)

ax.set_xlabel("Day")
ax.set_ylabel("Temperature")
ax.set_title("Weather Prediction")

ax.legend()

st.pyplot(fig)