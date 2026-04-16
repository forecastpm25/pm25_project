from flask import Flask, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

app = Flask(__name__)

# ===== FIREBASE =====
cred = credentials.Certificate("pm25-29c12-firebase-adminsdk-fbsvc-23a0e50ede.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ===== CONFIG =====
LOOKBACK = 48
HORIZONS = list(range(1,25)) + [48,72]

STATIONS = [
    {"name": "station_5030", "collection": "pm25_station_5030"},
    {"name": "station_3295", "collection": "pm25_station_3295"}
]

FEATURES = [
    "pm2.5",
    "is_missing",
    "lag_1","lag_24",
    "rolling_mean_3","rolling_mean_6",
    "rolling_std_6",
    "diff_1","diff_24",
    "hour_sin","hour_cos",
    "dow_sin","dow_cos",
    "month_sin","month_cos"
]

# ===== LOAD =====
MODELS = {}
for s in STATIONS:
    path = f"models/{s['name']}"
    MODELS[s["name"]] = {
        "model": load_model(f"{path}/lstm_model.h5"),
        "x_scaler": joblib.load(f"{path}/x_scaler.pkl"),
        "y_scaler": joblib.load(f"{path}/y_scaler.pkl")
    }

# ===== PREPROCESS =====
def preprocess(df):
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"pm25": "pm2.5"})
    df["pm2.5"] = pd.to_numeric(df["pm2.5"], errors="coerce")

    df = df.sort_values("datetime")
    df = df[["datetime","pm2.5"]]

    df = df.set_index("datetime").resample("1h").mean()

    df["is_missing"] = df["pm2.5"].isna().astype(int)
    df["pm2.5"] = df["pm2.5"].interpolate(method="time", limit=6)

    df = df.dropna().reset_index()

    df["hour"] = df["datetime"].dt.hour
    df["dow"]  = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    df["lag_1"] = df["pm2.5"].shift(1)
    df["lag_24"] = df["pm2.5"].shift(24)

    df["rolling_mean_3"] = df["pm2.5"].rolling(3).mean()
    df["rolling_mean_6"] = df["pm2.5"].rolling(6).mean()
    df["rolling_std_6"]  = df["pm2.5"].rolling(6).std()

    df["diff_1"] = df["pm2.5"].diff()
    df["diff_24"] = df["pm2.5"].diff(24)

    df = df.dropna()

    return df

# ===== PREDICT =====
def predict_station(s):
    docs = db.collection(s["collection"]).stream()
    data = [d.to_dict() for d in docs]

    if len(data) == 0:
        return {"station": s["name"], "error": "no data"}

    df = preprocess(pd.DataFrame(data))

    if len(df) < LOOKBACK:
        return {"station": s["name"], "error": "not enough data"}

    last = df.tail(LOOKBACK)

    model_pack = MODELS[s["name"]]

    x = model_pack["x_scaler"].transform(last[FEATURES])
    x = np.expand_dims(x, axis=0)

    pred = model_pack["model"].predict(x, verbose=0)[0]

    # inverse scale
    pred = model_pack["y_scaler"].inverse_transform(pred.reshape(-1,1)).flatten()

    # 🔥 inverse log
    pred = np.expm1(pred)

    # 🔥 clamp
    pred = np.clip(pred, 0, 300)

    result = {f"t+{h}": float(v) for h, v in zip(HORIZONS, pred)}
    result["station"] = s["name"]
    result["created_at"] = datetime.now().isoformat()

    db.collection("pm25_prediction").add(result)

    return result

@app.route("/predict")
def predict():
    return jsonify([predict_station(s) for s in STATIONS])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)