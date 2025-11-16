"""
EV Charging Occupancy Predictor — Pattern Enhanced ⚡
------------------------------------------------------
✅ Realistic hourly patterns (Office, Home, Mall, Highway)
✅ Predicts percentage vacancy across all stations
✅ 24-hour occupancy trend with instant switching
✅ Weather auto-fetched, inputs saved in session_state
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore", message="X has feature names")

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_PATH = os.path.join(OUT_DIR, "ev_cleaned.csv")
META_PATH = os.path.join(OUT_DIR, "meta.pkl")
SCALER_X_PATH = os.path.join(OUT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(OUT_DIR, "scaler_y.pkl")
MODEL_PATH = os.path.join(OUT_DIR, "lstm_ev_model.keras")

SEQ_LEN = 48
API_KEY = "ee70548d21d4824ef8e3ac69c01c8a4f"  # your saved key
DEFAULT_CITY = "Noida"

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="EV Charging Occupancy Predictor", layout="wide")
st.title("⚡ EV Charging Occupancy Predictor")
st.caption("Predicts EV charging station vacancy percentage using LSTM + custom behavioral patterns.")

# =========================================================
# LOAD FUNCTIONS
# =========================================================
@st.cache_resource
def load_artifacts():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values(["station_id", "timestamp"])
    with open(META_PATH, "rb") as f: meta = pickle.load(f)
    with open(SCALER_X_PATH, "rb") as f: scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f: scaler_y = pickle.load(f)
    model = load_model(MODEL_PATH, compile=False)
    return df, meta, scaler_X, scaler_y, model

def get_hourly_forecast(city, api_key):
    """Fetch hourly temperature for the next few days."""
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        r = requests.get(url, params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        df = pd.DataFrame([(pd.to_datetime(p["dt_txt"]), p["main"]["temp"]) for p in data["list"]],
                          columns=["timestamp", "temp"]).set_index("timestamp")
        hourly = df.resample("1h").interpolate(method="time")
        return {t.replace(minute=0, second=0, microsecond=0): float(v) for t, v in hourly["temp"].items()}
    except Exception:
        return {}

# =========================================================
# LOAD ARTIFACTS
# =========================================================
df, meta, scaler_X, scaler_y, model = load_artifacts()
feature_cols = meta["feature_cols"]
station_to_index = meta["station_to_index"]

if "station_name" in df.columns:
    id_to_name = df.groupby("station_id")["station_name"].first().to_dict()
else:
    id_to_name = df.groupby("station_id")["location_type"].first().to_dict()

# =========================================================
# CUSTOM BEHAVIORAL PATTERN FUNCTION (Option 1)
# =========================================================
def adjust_pattern(station_name, hour, pred_occ):
    """Adjusts occupancy (%) based on known station behavior."""
    sname = station_name.lower()

    # Household / Residential
    if "house" in sname or "resid" in sname:
        if 21 <= hour or hour <= 8:   # 9 PM - 8 AM
            return min(100, pred_occ * 1.3)
        else:
            return max(0, pred_occ * 0.7)

    # Office / Workplace
    elif "office" in sname or "work" in sname:
        if 8 <= hour <= 19:
            return min(100, pred_occ * 1.25)
        else:
            return max(0, pred_occ * 0.6)

    # Mall / Market / Commercial
    elif "mall" in sname or "market" in sname or "plaza" in sname:
        if 10 <= hour <= 22:
            return min(100, pred_occ * 1.15)
        else:
            return max(0, pred_occ * 0.8)

    # Highway / Express
    elif "highway" in sname or "express" in sname:
        return pred_occ  # minimal adjustment

    # Default
    return pred_occ

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🔧 Prediction Settings")

city = st.sidebar.text_input("Weather City", DEFAULT_CITY)
car_model = st.sidebar.selectbox("Car Model", ["NEXON EV", "XEV 900", "WRANGLER", "MG ZS EV", "Other"])
car_battery_map = {"NEXON EV": 40.0, "XEV 900": 32.0, "WRANGLER": 75.0, "MG ZS EV": 50.0}
battery_kwh = st.sidebar.number_input("Battery Capacity (kWh)", value=car_battery_map.get(car_model, 50.0))
selected_date = st.sidebar.date_input("Date")
selected_time = st.sidebar.time_input("Time")
predict_flag = st.sidebar.button("🚀 Predict")

# =========================================================
# RUN PREDICTION & CACHE IN SESSION
# =========================================================
if predict_flag or "predictions" not in st.session_state:
    sel_dt = pd.to_datetime(datetime.combine(selected_date, selected_time))
    hour, month, weekday = sel_dt.hour, sel_dt.month, sel_dt.weekday()
    is_weekend = 1 if weekday in [5, 6] else 0
    hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)

    forecast = get_hourly_forecast(city, API_KEY)
    forecast_key = sel_dt.replace(minute=0, second=0, microsecond=0)

    results, station_forecasts = [], {}

    for sid, idx in station_to_index.items():
        sdf = df[df["station_id"] == sid].sort_values("timestamp")
        if len(sdf) < SEQ_LEN:
            continue

        seq = sdf[feature_cols].iloc[-SEQ_LEN:].copy()
        seq.iloc[-1, seq.columns.get_loc("hour")] = hour
        seq.iloc[-1, seq.columns.get_loc("hour_sin")] = hour_sin
        seq.iloc[-1, seq.columns.get_loc("hour_cos")] = hour_cos
        seq.iloc[-1, seq.columns.get_loc("month")] = month
        seq.iloc[-1, seq.columns.get_loc("is_weekend")] = is_weekend
        seq.iloc[-1, seq.columns.get_loc("avg_battery_capacity_kwh")] = battery_kwh
        seq.iloc[-1, seq.columns.get_loc("temperature_c")] = forecast.get(forecast_key, seq.iloc[-1]["temperature_c"])

        seq_scaled = scaler_X.transform(seq.values).reshape(1, SEQ_LEN, len(feature_cols))
        pred_scaled = model.predict([seq_scaled, np.array([[idx]])], verbose=0)[0][0]
        pred_occ = scaler_y.inverse_transform([[pred_scaled]])[0][0]

        # Apply manual adjustment
        pred_occ = adjust_pattern(id_to_name.get(sid, ""), hour, pred_occ)

        cap = int(sdf["station_capacity_spots"].iloc[-1])
        pred_vac = max(0.0, 100.0 - pred_occ)

        results.append({
            "station_id": sid,
            "station_name": id_to_name.get(sid, "Unknown"),
            "predicted_occupancy(%)": round(pred_occ, 2),
            "predicted_vacancy(%)": round(pred_vac, 2),
            "capacity": cap
        })

        # === Generate 24h Forecast ===
        cur_seq = seq.copy()
        last_time = sel_dt
        rows = []
        for h in range(1, 25):
            nxt = last_time + timedelta(hours=h)
            cur_seq.iloc[-1, cur_seq.columns.get_loc("hour")] = nxt.hour
            cur_seq.iloc[-1, cur_seq.columns.get_loc("hour_sin")] = np.sin(2*np.pi*nxt.hour/24)
            cur_seq.iloc[-1, cur_seq.columns.get_loc("hour_cos")] = np.cos(2*np.pi*nxt.hour/24)
            cur_seq.iloc[-1, cur_seq.columns.get_loc("is_weekend")] = 1 if nxt.weekday() >= 5 else 0
            cur_seq.iloc[-1, cur_seq.columns.get_loc("temperature_c")] = forecast.get(
                nxt.replace(minute=0, second=0, microsecond=0),
                cur_seq.iloc[-1]["temperature_c"]
            )
            seq_scaled = scaler_X.transform(cur_seq.values).reshape(1, SEQ_LEN, len(feature_cols))
            pred_scaled = model.predict([seq_scaled, np.array([[idx]])], verbose=0)[0][0]
            occ = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            occ = adjust_pattern(id_to_name.get(sid, ""), nxt.hour, occ)
            rows.append({"hour": nxt.hour, "occupancy(%)": occ})
            new_row = cur_seq.iloc[-1].copy()
            new_row["occupancy_rate"] = occ
            cur_seq = pd.concat([cur_seq.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

        station_forecasts[sid] = pd.DataFrame(rows)

    st.session_state["predictions"] = pd.DataFrame(results).sort_values("predicted_vacancy(%)", ascending=False)
    st.session_state["forecasts"] = station_forecasts
    st.session_state["sel_dt"] = sel_dt
    st.session_state["city"] = city

# =========================================================
# DISPLAY RESULTS
# =========================================================
res_df = st.session_state["predictions"]
station_forecasts = st.session_state["forecasts"]
sel_dt = st.session_state["sel_dt"]
city = st.session_state["city"]

res_df["label"] = res_df["station_id"].astype(str) + " — " + res_df["station_name"]

st.subheader(f"Predicted Vacancies — {city}, {sel_dt.strftime('%Y-%m-%d %H:%M')}")
st.dataframe(res_df[["label", "predicted_occupancy(%)", "predicted_vacancy(%)", "capacity"]], use_container_width=True)

# --- Vacancy Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(res_df["label"], res_df["predicted_vacancy(%)"], color="teal")
ax.set_xlabel("Vacancy (%)")
ax.set_ylabel("Station")
ax.set_title("Vacancy by Station (Next Hour)")
ax.invert_yaxis()
for i, val in enumerate(res_df["predicted_vacancy(%)"]):
    ax.text(val + 0.3, i, f"{val:.1f}%", va="center")
st.pyplot(fig)

# =========================================================
# 24-Hour Occupancy Trend
# =========================================================
st.markdown("---")
st.subheader("📊 Station 24-Hour Occupancy Trend")

station_labels = {sid: f"{sid} — {res_df.loc[res_df['station_id'] == sid, 'station_name'].values[0]}"
                   for sid in station_forecasts.keys()}
selected_station = st.selectbox("Select Station", list(station_labels.values()))

selected_sid = int(selected_station.split(" — ")[0])
forecast_df = station_forecasts[selected_sid]

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(forecast_df["hour"], forecast_df["occupancy(%)"], marker="o", color="royalblue")
ax2.set_title(f"24-Hour Occupancy — {selected_station}")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Occupancy (%)")
ax2.set_xticks(range(0, 24, 2))
ax2.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig2)
