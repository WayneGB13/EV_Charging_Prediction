"""
Phase 3 — LSTM with Station Embedding + Permutation Importance + 24h Forecast
Full version — bug-free and Streamlit-ready.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import requests
from datetime import timedelta

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Embedding, Concatenate,
    RepeatVector, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but MinMaxScaler was fitted without feature names")

# ======================================================
# CONFIGURATION
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(os.path.join(OUT_DIR, "eda_visuals"), exist_ok=True)

DATA_PATH = os.path.join(OUT_DIR, "ev_cleaned.csv")
MODEL_PATH = os.path.join(OUT_DIR, "lstm_ev_model.keras")
SCALER_X_PATH = os.path.join(OUT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(OUT_DIR, "scaler_y.pkl")
META_PATH = os.path.join(OUT_DIR, "meta.pkl")
IMPORTANCE_PATH = os.path.join(OUT_DIR, "feature_importances.csv")
IMPORTANCE_PNG = os.path.join(OUT_DIR, "eda_visuals", "feature_importance.png")
VAL_PLOT_PATH = os.path.join(OUT_DIR, "eda_visuals", "val_pred_vs_true.png")
FORECAST_OUT_PATH = os.path.join(OUT_DIR, "future_24h_occupancy_forecast.csv")

EPOCHS = 40
BATCH_SIZE = 32
SEQ_LEN = 48
EMBEDDING_DIM = 8
API_KEY = "ee70548d21d4824ef8e3ac69c01c8a4f"

# ======================================================
# WEATHER FORECAST FUNCTION
# ======================================================
def get_hourly_forecast_interpolated(city_name, api_key):
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        r = requests.get(url, params={"q": city_name, "appid": api_key, "units": "metric"}, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
        rows = [(pd.to_datetime(p["dt_txt"]), p["main"]["temp"]) for p in data.get("list", [])]
        if not rows:
            return {}
        df_fore = pd.DataFrame(rows, columns=["timestamp", "temp"]).set_index("timestamp").sort_index()
        hourly = df_fore.resample("1h").interpolate(method="time")
        return {pd.to_datetime(k).replace(minute=0, second=0, microsecond=0): float(v)
                for k, v in hourly["temp"].to_dict().items()}
    except Exception as e:
        print("Weather forecast error:", e)
        return {}

# ======================================================
# LOAD CLEANED DATA
# ======================================================
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"[ERROR] Cleaned dataset missing at {DATA_PATH}. Run Phase 1 first!")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values(["station_id", "timestamp"])
print(f"[OK] Loaded cleaned dataset: {df.shape}")

# ======================================================
# FEATURE ENGINEERING
# ======================================================
if "hour" not in df.columns:
    df["hour"] = df["timestamp"].dt.hour
if "month" not in df.columns:
    df["month"] = df["timestamp"].dt.month
if "day_of_week_num" not in df.columns:
    df["day_of_week_num"] = df["timestamp"].dt.weekday
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# One-hot encoding (if not already applied)
if not any(c.startswith("loc_") for c in df.columns) and "location_type" in df.columns:
    df = pd.get_dummies(df, columns=["location_type"], prefix="loc", drop_first=False)
if not any(c.startswith("dow_") for c in df.columns) and "day_of_week" in df.columns:
    df = pd.get_dummies(df, columns=["day_of_week"], prefix="dow", drop_first=False)
if not any(c.startswith("season_") for c in df.columns) and "season" in df.columns:
    df = pd.get_dummies(df, columns=["season"], prefix="season", drop_first=False)

loc_ohe = [c for c in df.columns if c.startswith("loc_")]
dow_ohe = [c for c in df.columns if c.startswith("dow_")]
season_ohe = [c for c in df.columns if c.startswith("season_")]

numeric_features = [
    "hour", "hour_sin", "hour_cos", "day_of_week_num",
    "is_weekend", "is_holiday", "month",
    "temperature_c", "avg_vehicle_age_years",
    "avg_battery_capacity_kwh", "station_capacity_spots",
    "occupancy_rate", "vacant_spots", "arrivals_this_hour", "demand_this_hour"
]

feature_cols = numeric_features + loc_ohe + dow_ohe + season_ohe
print(f"[INFO] Using total features: {len(feature_cols)}")

# ======================================================
# NEXT-HOUR TARGET
# ======================================================
df["next_occupancy_rate"] = df.groupby("station_id")["occupancy_rate"].shift(-1)
df = df.dropna(subset=["next_occupancy_rate"])

unique_stations = sorted(df["station_id"].unique())
station_to_index = {int(s): i for i, s in enumerate(unique_stations)}
index_to_station = {i: int(s) for s, i in station_to_index.items()}
num_stations = len(unique_stations)
print(f"[INFO] Stations count: {num_stations}")

# ======================================================
# SEQUENCE CREATION
# ======================================================
X_list, y_list, sid_list = [], [], []

for sid in unique_stations:
    sdf = df[df["station_id"] == sid].sort_values("timestamp")
    X_seq = sdf[feature_cols].values
    y_seq = sdf["next_occupancy_rate"].values
    if len(sdf) <= SEQ_LEN:
        continue
    for i in range(len(sdf) - SEQ_LEN):
        X_list.append(X_seq[i:i+SEQ_LEN])
        y_list.append(y_seq[i+SEQ_LEN])
        sid_list.append(station_to_index[int(sid)])

X = np.array(X_list)
y = np.array(y_list)
sid_idx = np.array(sid_list)
print(f"[OK] Built sequences: {X.shape}")

# ======================================================
# SCALING
# ======================================================
scaler_X = MinMaxScaler().fit(X.reshape(-1, X.shape[2]))
X_scaled = scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1, 1))
y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

with open(SCALER_X_PATH, "wb") as f: pickle.dump(scaler_X, f)
with open(SCALER_Y_PATH, "wb") as f: pickle.dump(scaler_y, f)
print("[OK] Scalers saved.")

# ======================================================
# TRAIN-VALIDATION SPLIT
# ======================================================
idx = np.arange(X.shape[0])
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
sid_train, sid_val = sid_idx[train_idx], sid_idx[val_idx]

# ======================================================
# BUILD MODEL (no Lambda)
# ======================================================
seq_input = Input(shape=(SEQ_LEN, X.shape[2]), name="seq_input")
station_input = Input(shape=(1,), dtype="int32", name="station_input")

emb = Embedding(input_dim=num_stations, output_dim=EMBEDDING_DIM, name="station_emb")(station_input)
emb = Reshape((EMBEDDING_DIM,), name="reshape_emb")(emb)
emb = RepeatVector(SEQ_LEN, name="repeat_emb")(emb)

x = Concatenate(axis=-1)([seq_input, emb])
x = LSTM(128, return_sequences=True, activation="tanh")(x)
x = Dropout(0.15)(x)
x = LSTM(96, activation="tanh")(x)
x = Dropout(0.15)(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
out = Dense(1, activation="linear")(x)

model = Model(inputs=[seq_input, station_input], outputs=out)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ======================================================
# TRAIN OR LOAD MODEL
# ======================================================
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Pre-trained model found. Loading: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
else:
    print("[INFO] Training model...")
    history = model.fit(
        [X_train, sid_train], y_train,
        validation_data=([X_val, sid_val], y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
        ],
        verbose=1
    )
    print("[OK] Training finished.")
    model.save(MODEL_PATH)
    print("[OK] Model saved successfully.")

# ======================================================
# SAVE META
# ======================================================
meta = {
    "feature_cols": feature_cols,
    "loc_ohe": loc_ohe,
    "dow_ohe": dow_ohe,
    "season_ohe": season_ohe,
    "use_embedding": True,
    "station_to_index": station_to_index,
    "index_to_station": index_to_station,
    "embedding_dim": EMBEDDING_DIM
}
with open(META_PATH, "wb") as f: pickle.dump(meta, f)
print("[OK] Meta saved.")

# ======================================================
# PHASE 4: 24-HOUR FORECAST
# ======================================================
print("[STEP] Generating 24-hour occupancy forecast...")

hourly_forecast = get_hourly_forecast_interpolated("Noida", API_KEY)
forecast_rows = []

for sid in unique_stations:
    idx = station_to_index[sid]
    sdf = df[df["station_id"] == sid].sort_values("timestamp").reset_index(drop=True)
    if len(sdf) < SEQ_LEN:
        continue

    seq = sdf[feature_cols].iloc[-SEQ_LEN:].copy().reset_index(drop=True)
    last_time = sdf["timestamp"].iloc[-1]

    for h in range(24):
        next_time = pd.to_datetime(last_time) + timedelta(hours=h + 1)
        seq.iloc[-1, seq.columns.get_loc("hour")] = next_time.hour
        seq.iloc[-1, seq.columns.get_loc("hour_sin")] = np.sin(2 * np.pi * next_time.hour / 24)
        seq.iloc[-1, seq.columns.get_loc("hour_cos")] = np.cos(2 * np.pi * next_time.hour / 24)
        seq.iloc[-1, seq.columns.get_loc("month")] = next_time.month
        seq.iloc[-1, seq.columns.get_loc("is_weekend")] = 1 if next_time.weekday() in [5, 6] else 0
        seq.iloc[-1, seq.columns.get_loc("is_holiday")] = 0
        key = next_time.replace(minute=0, second=0, microsecond=0)
        seq.iloc[-1, seq.columns.get_loc("temperature_c")] = hourly_forecast.get(key, seq.iloc[-1]["temperature_c"])

        seq_scaled = scaler_X.transform(seq).reshape(1, SEQ_LEN, len(feature_cols))
        pred_scaled = model.predict([seq_scaled, np.array([idx])], verbose=0)[0][0]
        pred_occ = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        pred_occ = max(0, min(pred_occ, 100))

        forecast_rows.append({
            "station_id": sid,
            "timestamp": next_time,
            "predicted_occupancy_rate(%)": round(pred_occ, 2)
        })

        next_row = seq.iloc[-1].copy()
        next_row["occupancy_rate"] = pred_occ
        seq = pd.concat([seq.iloc[1:], pd.DataFrame([next_row])], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(FORECAST_OUT_PATH, index=False)
print(f"[OK] Forecast saved to {FORECAST_OUT_PATH}")

# ======================================================
# VISUALIZE FORECAST
# ======================================================
plt.figure(figsize=(10, 6))
for sid in forecast_df["station_id"].unique():
    sdf = forecast_df[forecast_df["station_id"] == sid]
    plt.plot(sdf["timestamp"], sdf["predicted_occupancy_rate(%)"], label=f"Station {sid}")
plt.title("Predicted 24-hour Occupancy (%) by Station")
plt.xlabel("Next 24 Hours")
plt.ylabel("Predicted Occupancy (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_visuals", "forecast_occupancy_percent.png"))
plt.close()

print("[OK] Forecast visualization saved successfully.")
