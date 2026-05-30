import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================
# SETTINGS
# =========================
st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Replace with your API key temporarily
API_KEY = "4224fb43fa45442e182cd29adcafd4f0"

# =========================
# TITLE
# =========================
st.title("🌤️ Smart Weather Dashboard")
st.write("Live Weather + Temperature Prediction")

# =========================
# LIVE WEATHER
# =========================
st.header("🌍 Live Weather")

city = st.text_input("Enter City Name", "Chennai")

if st.button("Get Weather"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        condition = data["weather"][0]["main"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🌡️ Temperature", f"{temp} °C")

        with col2:
            st.metric("💧 Humidity", f"{humidity}%")

        with col3:
            st.metric("🌬️ Wind", f"{wind} m/s")

        with col4:
            st.metric("☁️ Condition", condition)

    else:
        st.error("City not found or API key not active yet.")

st.divider()

# =========================
# DATASET
# =========================
df = pd.read_csv("weather.csv")

st.subheader("📊 Weather Dataset")
st.dataframe(df, use_container_width=True)

# =========================
# ML MODEL
# =========================
X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

# =========================
# PREDICTION
# =========================
st.subheader("🤖 Temperature Prediction")

future_day = st.number_input(
    "Enter Future Day",
    min_value=1,
    value=25
)

if st.button("Predict Temperature"):
    prediction = model.predict([[future_day]])

    st.success(
        f"Predicted Temperature on Day {future_day}: {prediction[0]:.2f} °C"
    )

# =========================
# INTERACTIVE GRAPH
# =========================
st.subheader("📈 Interactive Temperature Trend")

fig = px.line(
    df,
    x="Day",
    y="Temperature",
    markers=True,
    title="Temperature Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)