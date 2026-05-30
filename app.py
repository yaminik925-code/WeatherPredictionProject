import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
st.set_page_config(
    page_title="Smart Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Your OpenWeather API Key
API_KEY = "4224fb43fa45442e182cd29adcafd4f0"

# =========================
# TITLE
# =========================
st.title("🌤️ Smart Weather Dashboard")
st.write("Live Weather + Temperature Prediction")

st.markdown("---")

# =========================
# LIVE WEATHER
# =========================
st.header("🌍 Live Weather")

city = st.text_input("Enter City Name", "Chennai")

if st.button("Get Weather"):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data["cod"] == 200:

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        condition = data["weather"][0]["main"]

        icon = data["weather"][0]["icon"]
        icon_url = f"https://openweathermap.org/img/wn/{icon}@4x.png"

        st.image(icon_url, width=120)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🌡 Temperature", f"{temp} °C")

        with col2:
            st.metric("💧 Humidity", f"{humidity}%")

        with col3:
            st.metric("💨 Wind", f"{wind} m/s")

        with col4:
            st.metric("☁ Condition", condition)

    else:
        st.error("City not found")

st.markdown("---")

# =========================
# DATASET
# =========================
st.header("📊 Weather Dataset")

df = pd.read_csv("weather.csv")

st.dataframe(df)

# =========================
# MACHINE LEARNING
# =========================
X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

st.markdown("---")

# =========================
# PREDICTION
# =========================
st.header("🔮 Temperature Prediction")

future_day = st.number_input(
    "Enter Day Number",
    min_value=1,
    value=20
)

if st.button("Predict Temperature"):

    prediction = model.predict([[future_day]])

    st.success(
        f"Predicted Temperature on Day {future_day}: {prediction[0]:.2f} °C"
    )

st.markdown("---")

# =========================
# MATPLOTLIB GRAPH
# =========================
st.header("📈 Prediction Graph")

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    df["Day"],
    df["Temperature"],
    color="blue",
    label="Actual Data"
)

ax.plot(
    df["Day"],
    model.predict(X),
    color="red",
    label="Prediction Line"
)

ax.set_xlabel("Day")
ax.set_ylabel("Temperature")
ax.legend()

st.pyplot(fig)

st.markdown("---")

# =========================
# PLOTLY GRAPH
# =========================
st.header("📉 Interactive Temperature Trend")

plot_fig = px.line(
    df,
    x="Day",
    y="Temperature",
    markers=True,
    title="Temperature Trend"
)

st.plotly_chart(
    plot_fig,
    use_container_width=True
)