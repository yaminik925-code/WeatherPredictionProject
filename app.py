import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

# =========================
# SETTINGS
# =========================

st.set_page_config(
    page_title="Smart Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# YOUR OPENWEATHER API KEY
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

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(weather_url)
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
            st.metric("🌡 Temperature", f"{temp:.2f} °C")

        with col2:
            st.metric("💧 Humidity", f"{humidity}%")

        with col3:
            st.metric("💨 Wind", f"{wind} m/s")

        with col4:
            st.metric("☁ Condition", condition)

        st.markdown("---")

        # =========================
        # 5 DAY FORECAST
        # =========================

        st.header("📅 5-Day Weather Forecast")

        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()

        if forecast_data["cod"] == "200":

            forecast_list = forecast_data["list"]

            dates = []
            temps = []

            for item in forecast_list[::8]:
                dates.append(item["dt_txt"].split(" ")[0])
                temps.append(item["main"]["temp"])

            forecast_df = pd.DataFrame({
                "Date": dates,
                "Temperature": temps
            })

            st.dataframe(forecast_df)

            forecast_fig = px.line(
                forecast_df,
                x="Date",
                y="Temperature",
                markers=True,
                title="5-Day Forecast"
            )

            st.plotly_chart(
                forecast_fig,
                use_container_width=True
            )

    else:
        st.error("City not found")

st.markdown("---")

# =========================
# DATASET
# =========================

st.header("📊 Weather Dataset")

df = pd.read_csv("weather.csv")

st.dataframe(df)

st.markdown("---")

# =========================
# MACHINE LEARNING
# =========================

X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

# =========================
# PREDICTION
# =========================

st.header("🔮 Temperature Prediction")

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

st.markdown("---")

# =========================
# PREDICTION GRAPH
# =========================

st.header("📈 Prediction Graph")

fig, ax = plt.subplots(figsize=(8, 4))

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
ax.legend()

st.pyplot(fig)

st.markdown("---")

# =========================
# INTERACTIVE GRAPH
# =========================

st.header("📉 Interactive Temperature Trend")

interactive_fig = px.line(
    df,
    x="Day",
    y="Temperature",
    markers=True,
    title="Temperature Trend"
)

st.plotly_chart(
    interactive_fig,
    use_container_width=True
)