import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict

BASE_URL = "http://connector:8003/predictor"

def fetch_weather_data(endpoint: str) -> List[float]:
    """
    Fetch weather predictions from the specified endpoint.

    Args:
        endpoint (str): The API endpoint to fetch data from.

    Returns:
        List[float]: A list of predicted temperatures.
    """
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json().get("predictions", [])
        else:
            st.error(f"Failed to fetch data from {endpoint}: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching data from {endpoint}: {e}")
        return []

def generate_time_labels(start_time: datetime, intervals: int, step_minutes: int) -> List[str]:
    """
    Generate a list of time labels based on the start time, intervals, and step size.

    Args:
        start_time (datetime): The starting datetime for labels.
        intervals (int): Number of time intervals.
        step_minutes (int): Step size in minutes between each interval.

    Returns:
        List[str]: A list of formatted time labels as strings.
    """
    times = [start_time + timedelta(minutes=i * step_minutes) for i in range(intervals)]
    return [time.strftime("%Y-%m-%d %H:%M") for time in times]

def plot_temperature_data(df: pd.DataFrame, forecast_type: str) -> None:
    """
    Plot temperature data as a bar chart.

    Args:
        df (pd.DataFrame): DataFrame containing 'Time' and 'Temperature' columns.
        forecast_type (str): The forecast type (e.g., "Next 24 Hours").
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["Time"], df["Temperature"], color="skyblue")
    ax.set_title(f"{forecast_type} Temperature")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    plt.xticks(rotation=90, fontsize=8)
    st.pyplot(fig)

st.title("Weather Prediction Dashboard")

forecast_config: Dict[str, tuple] = {
    "Next 24 Hours": ("/predict/next_24_hours", 24, 60),
    "Next 7 Days": ("/predict/next_7_days", 7, 1440),
}

current_time = datetime.now()
for forecast_type, (endpoint, intervals, step_minutes) in forecast_config.items():
    st.subheader(f"{forecast_type} Forecast")

    temperatures = fetch_weather_data(endpoint)
    
    if temperatures:
        time_labels = generate_time_labels(current_time, intervals, step_minutes)
        df = pd.DataFrame({
            "Time": time_labels,
            "Temperature": temperatures
        })

        st.write(f"Forecasted Temperatures for {forecast_type}:")
        st.dataframe(df)

        plot_temperature_data(df, forecast_type)
    else:
        st.warning(f"No data available for {forecast_type}. Please check the API.")
