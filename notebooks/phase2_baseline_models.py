# ======================================================
# PHASE 2: BASELINE & CLASSICAL ML MODELS
# PART 1: ROBUST PATH SETUP & DATA LOADING
# ======================================================

import os
import pandas as pd
import numpy as np

def find_project_root(start_path=None, markers=('data', 'outputs', 'EV_Charging_Prediction')):
    """
    Walk upward from start_path until we find a folder that contains any marker directory/name.
    Returns the path to that folder or None if not found.
    """
    if start_path is None:
        start_path = os.getcwd()
    cur = os.path.abspath(start_path)

    # safety: avoid infinite loop; stop at filesystem root
    while True:
        # check for presence of marker directories or folder name
        for m in markers:
            # if marker folder exists inside cur OR cur folder name matches marker
            if os.path.isdir(os.path.join(cur, m)) or os.path.basename(cur) == m:
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            return None
        cur = parent

# Try to discover project root automatically
PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    # fallback: assume one level up from notebooks
    PROJECT_ROOT = os.path.dirname(os.getcwd())

print("[INFO] Project root detected as:", PROJECT_ROOT)

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'ev_cleaned.csv')  # cleaned file from Phase 1
# if not present in outputs, try 'data' folder as a fallback
if not os.path.exists(DATA_PATH):
    alt = os.path.join(PROJECT_ROOT, 'data', 'synthetic_ev_data_time_aware.csv')
    if os.path.exists(alt):
        DATA_PATH = alt

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find ev_cleaned.csv or the original CSV. "
                            f"Checked: {os.path.join(PROJECT_ROOT, 'outputs', 'ev_cleaned.csv')} "
                            f"and {os.path.join(PROJECT_ROOT, 'data', 'synthetic_ev_data_time_aware.csv')}")

# Load
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
print(f"[OK] Loaded cleaned dataset from: {DATA_PATH}")
print(f"[OK] Dataset shape: {df.shape}")

# choose target
target_col = 'arrivals_this_hour'
print(f"[INFO] Target variable selected: {target_col}")
# ======================================================
# PART 2: FEATURE SELECTION & TRAIN-TEST SPLIT
# ======================================================

from sklearn.model_selection import train_test_split

# --- 1. Select numerical and categorical features ---
# We'll use mainly time, station, and environmental factors
feature_cols = [
    'hour', 'month', 'temperature_c',
    'is_weekend', 'is_holiday',
    'avg_vehicle_age_years', 'avg_battery_capacity_kwh',
    'station_capacity_spots', 'occupancy_rate', 'vacant_spots'
]

# Check which of these exist in the dataframe (safe check)
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df[target_col]

print(f"[OK] Selected {len(feature_cols)} features: {feature_cols}")
print(f"[INFO] Feature matrix shape: {X.shape}, Target shape: {y.shape}")

# --- 2. Split into training & test sets ---
# We'll keep last 20% for testing (time-aware split could also be used later)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[OK] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ======================================================
# PART 3: BASELINE MACHINE-LEARNING MODELS
# ======================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- Helper: Evaluation function ---
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model, make predictions, and print performance metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n===== {name} RESULTS =====")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

# --- 1️⃣ Linear Regression (Baseline) ---
lin_reg = LinearRegression()
results = []
results.append(evaluate_model("Linear Regression", lin_reg, X_train, X_test, y_train, y_test))

# --- 2️⃣ Random Forest Regressor ---
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
results.append(evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test))

# --- 3️⃣ XGBoost Regressor ---
xgb = XGBRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
results.append(evaluate_model("XGBoost", xgb, X_train, X_test, y_train, y_test))

# --- Summarize all results ---
import pandas as pd
res_df = pd.DataFrame(results).sort_values(by='RMSE')
print("\n==============================")
print("MODEL PERFORMANCE SUMMARY")
print("==============================")
print(res_df.to_string(index=False))

# --- Optional: Save results ---
res_path = os.path.join(PROJECT_ROOT, 'outputs', 'baseline_results.csv')
res_df.to_csv(res_path, index=False)
print(f"\n[OK] Baseline model results saved to {res_path}")

# ======================================================
# PART 4: MODEL VISUALIZATION & FEATURE IMPORTANCE
# ======================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Visualization Path ===
EDA_VISUALS_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'eda_visuals')
os.makedirs(EDA_VISUALS_PATH, exist_ok=True)

# === 1️⃣ Model Performance Comparison ===
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='RMSE', data=res_df, palette='Blues_d')
plt.title('Model Comparison: RMSE (Lower is Better)')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, 'baseline_model_rmse.png'))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='R2', data=res_df, palette='Greens_d')
plt.title('Model Comparison: R² (Higher is Better)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.tight_layout()
plt.savefig(os.path.join(EDA_VISUALS_PATH, 'baseline_model_r2.png'))
plt.close()

print("[OK] Model performance charts saved to outputs/eda_visuals/")

# === 2️⃣ Feature Importance: Random Forest ===
try:
    importances_rf = rf.feature_importances_
    sorted_idx_rf = np.argsort(importances_rf)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances_rf[sorted_idx_rf],
        y=np.array(feature_cols)[sorted_idx_rf],
        palette='viridis'
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_VISUALS_PATH, 'feature_importance_rf.png'))
    plt.close()

    print("[OK] Random Forest feature importance plot saved.")

except Exception as e:
    print("[WARN] Could not compute Random Forest feature importance:", e)

# === 3️⃣ Feature Importance: XGBoost ===
try:
    importances_xgb = xgb.feature_importances_
    sorted_idx_xgb = np.argsort(importances_xgb)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances_xgb[sorted_idx_xgb],
        y=np.array(feature_cols)[sorted_idx_xgb],
        palette='mako'
    )
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_VISUALS_PATH, 'feature_importance_xgb.png'))
    plt.close()

    print("[OK] XGBoost feature importance plot saved.")

except Exception as e:
    print("[WARN] Could not compute XGBoost feature importance:", e)
