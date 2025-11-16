# ======================================================
# PHASE 1: DATA PREPARATION & EXPLORATORY ANALYSIS
# (UPDATED FOR ONE-HOT ENCODING + META EXPORT)
# ======================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ------------------------------------------------------
# PATH SETUP (same style as before)
# ------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # notebooks/
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))     # EV_Charging_Prediction
DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "synthetic_ev_data_time_aware.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "ev_cleaned.csv")
EDA_VISUALS_PATH = os.path.join(PROJECT_ROOT, "outputs", "eda_visuals")
META_PATH = os.path.join(PROJECT_ROOT, "outputs", "meta.pkl")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(EDA_VISUALS_PATH, exist_ok=True)

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print(f"[OK] Loaded dataset with shape: {df.shape}\n")

print(df.info())
print("\nSample rows:\n", df.head())

# ------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------
print("\nChecking for missing values...")
print(df.isna().sum())

if df["avg_battery_capacity_kwh"].isna().sum() > 0:
    df["avg_battery_capacity_kwh"].fillna(df["avg_battery_capacity_kwh"].mean(), inplace=True)

df = df[(df["occupancy_rate"] >= 0) & (df["occupancy_rate"] <= 100)]
df = df[df["vacant_spots"] >= 0]
df = df[(df["temperature_c"] > -10) & (df["temperature_c"] < 50)]

# ------------------------------------------------------
# FEATURE ENGINEERING (same, expanded)
# ------------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week_num"] = df["timestamp"].dt.weekday
df["is_weekend"] = df["day_of_week_num"].isin([5, 6]).astype(int)

# season
df["season"] = df["month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Monsoon", 10: "Monsoon", 11: "Autumn"
})

# hour cyclic
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ------------------------------------------------------
# ONE-HOT ENCODING (new, required for LSTM + Streamlit)
# ------------------------------------------------------
ohe_location = pd.get_dummies(df["location_type"], prefix="loc")
ohe_dow = pd.get_dummies(df["day_of_week"], prefix="dow")
ohe_season = pd.get_dummies(df["season"], prefix="season")

df = pd.concat([df, ohe_location, ohe_dow, ohe_season], axis=1)

# ------------------------------------------------------
# FINAL FEATURE LIST (needed for training)
# ------------------------------------------------------
numeric_features = [
    "hour", "month", "temperature_c", "is_weekend", "is_holiday",
    "avg_vehicle_age_years", "avg_battery_capacity_kwh",
    "station_capacity_spots", "occupancy_rate", "vacant_spots",
    "arrivals_this_hour", "demand_this_hour",
    "hour_sin", "hour_cos"
]

feature_cols = (
    numeric_features
    + list(ohe_location.columns)
    + list(ohe_dow.columns)
    + list(ohe_season.columns)
)

print("\nTotal features after OHE:", len(feature_cols))
print("Feature columns used for training saved in meta.pkl")

# ------------------------------------------------------
# SAVE CLEANED DATASET
# ------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print("\nCleaned dataset saved to:", OUTPUT_PATH)

# ------------------------------------------------------
# SAVE META (for Phase-3 + Streamlit)
# ------------------------------------------------------
meta = {
    "feature_cols": feature_cols,
    "loc_ohe": list(ohe_location.columns),
    "dow_ohe": list(ohe_dow.columns),
    "season_ohe": list(ohe_season.columns),
}
with open(META_PATH, "wb") as f:
    pickle.dump(meta, f)

print("meta.pkl saved with feature information.")

# ------------------------------------------------------
# EDA VISUALS (unchanged)
# ------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")

# 1. Average hourly arrivals
plt.figure(figsize=(8, 4))
df.groupby("hour")["arrivals_this_hour"].mean().plot()
plt.title("Average Hourly Arrivals Across Stations")
plt.xlabel("Hour")
plt.ylabel("Avg Arrivals")
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, "hourly_arrivals.png"))
plt.close()

# 2. Occupancy by location
plt.figure(figsize=(8, 4))
df.groupby("location_type")["occupancy_rate"].mean().sort_values().plot(kind="bar")
plt.title("Average Occupancy by Location Type")
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, "occupancy_by_location.png"))
plt.close()

# 3. Temperature vs Arrivals
plt.figure(figsize=(6, 4))
plt.scatter(df["temperature_c"], df["arrivals_this_hour"], alpha=0.3)
plt.xlabel("Temperature (C)")
plt.ylabel("Arrivals")
plt.title("Temperature vs Charging Demand")
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, "temperature_vs_demand.png"))
plt.close()

# 4. Monthly demand
plt.figure(figsize=(8, 4))
df.groupby("month")["arrivals_this_hour"].mean().plot(marker="o")
plt.title("Monthly Avg Charging Demand")
plt.xlabel("Month")
plt.ylabel("Avg Arrivals")
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, "monthly_demand.png"))
plt.close()

# 5. Correlation heatmap
corr = df[[
    "arrivals_this_hour", "temperature_c", "hour", "is_weekend",
    "avg_vehicle_age_years", "avg_battery_capacity_kwh", "occupancy_rate"
]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, "correlation_heatmap.png"))
plt.close()

print("\nEDA visuals saved to:", EDA_VISUALS_PATH)
print("\nPhase 1 completed successfully.")
