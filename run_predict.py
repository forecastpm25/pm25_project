import numpy as np
import pandas as pd
import joblib
import os
import gdown
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model

import firebase_admin
from firebase_admin import credentials, firestore

# ==============================
# 🔥 CONFIG
# ==============================
LOOKBACK = 48
HORIZONS = list(range(1,25)) + [48,72]

STATIONS = [
    {"name": "station_5030", "collection": "pm25_station_5030"},
    {"name": "station_3295", "collection": "pm25_station_3295"}
]

FEATURES = [
    "pm2.5","is_missing",
    "lag_1","lag_24",
    "rolling_mean_3","rolling_mean_6",
    "rolling_std_6",
    "diff_1","diff_24",
    "hour_sin","hour_cos",
    "dow_sin","dow_cos",
    "month_sin","month_cos"
]

# ==============================
# 🔥 GOOGLE DRIVE LINKS (ใส่ของคุณ)
# ==============================
DRIVE_LINKS = {
    "station_5030": {
        "lstm": "https://drive.google.com/file/d/uc?id=1dd2ghsr42Ri9kYxaQXy7dJ6VvNAXrh-O",
        "x_scaler": "https://drive.google.com/file/d/uc?id=1An9OyJWwXG_U-1PDQo91HFINVa16PKYG",
        "y_scaler": "https://drive.google.com/file/d/uc?id=14oRLg7kWvGk50v5cWboZwkhY-OfikrNh",
        "arima": "https://drive.google.com/file/d/uc?id=1-s74diG8a-1J3xrIO3Y15HPEdqBu6oO-"
    },
    "station_3295": {
        "lstm": "https://drive.google.com/file/d/uc?id=1St9YNrJCppsRQHJ42Bt0tuCGjTCYbFNa",
        "x_scaler": "https://drive.google.com/file/d/uc?id=1UHxW__2h4g_dC7K-Nc_pggYWWcwbjqxw",
        "y_scaler": "https://drive.google.com/file/d/uc?id=1CXji3AgLPJnbcMbXGgm6vLfL6YhusI8o",
        "arima": "https://drive.google.com/file/d/uc?id=1xwnFmFM6Tmp8-Z5O4roYgf0TJEWT26H_"
    }
}

# ==============================
# 🔥 DOWNLOAD MODEL
# ==============================
def download_models():
    for s in STATIONS:
        name = s["name"]
        path = f"models/{name}"
        os.makedirs(path, exist_ok=True)

        links = DRIVE_LINKS[name]

        files = {
            "lstm_model.h5": links["lstm"],
            "x_scaler.pkl": links["x_scaler"],
            "y_scaler.pkl": links["y_scaler"],
            "arima.pkl": links["arima"]
        }

        for fname, url in files.items():
            fpath = f"{path}/{fname}"
            if not os.path.exists(fpath):
                print(f"⬇️ downloading {fname}")
                gdown.download(url, fpath, quiet=False, fuzzy=True)
# ==============================
# 🔥 FIREBASE
# ==============================
def init_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    return firestore.client()

# ==============================
# 🔥 PREPROCESS
# ==============================
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

    return df.dropna()

# ==============================
# 🔥 LOAD MODEL
# ==============================
def load_models():
    MODELS = {}

    for s in STATIONS:
        name = s["name"]
        path = f"models/{name}"

        MODELS[name] = {
            "model": load_model(f"{path}/lstm_model.h5"),
            "x_scaler": joblib.load(f"{path}/x_scaler.pkl"),
            "y_scaler": joblib.load(f"{path}/y_scaler.pkl"),
            "arima": joblib.load(f"{path}/arima.pkl")
        }

    return MODELS

# ==============================
# 🔥 RUN
# ==============================
def run():
    print("🚀 START")

    download_models()
    db = init_firebase()
    MODELS = load_models()

    for s in STATIONS:
        print("🔍", s["name"])

        docs = db.collection(s["collection"]).stream()
        data = [d.to_dict() for d in docs]

        if len(data) == 0:
            print("❌ no data")
            continue

        df = preprocess(pd.DataFrame(data))

        if len(df) < LOOKBACK:
            print("❌ not enough data")
            continue

        last = df.tail(LOOKBACK)
        pack = MODELS[s["name"]]

        x = pack["x_scaler"].transform(last[FEATURES])
        x = np.expand_dims(x, axis=0)

        # ===== LSTM =====
        lstm_pred = pack["model"].predict(x, verbose=0)[0]

        # ===== ARIMA =====
        residual_pred = pack["arima"].forecast(steps=len(HORIZONS))

        # ===== COMBINE =====
        pred = lstm_pred + residual_pred

        pred = pack["y_scaler"].inverse_transform(pred.reshape(-1,1)).flatten()
        pred = np.expm1(pred)
        pred = np.clip(pred, 0, 300)

        result = {f"t+{h}": round(float(v), 2) for h, v in zip(HORIZONS, pred)}
        result["station"] = s["name"]
        tz = pytz.timezone("Asia/Bangkok")
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        result["created_at"] = now

        db.collection("pm25_prediction").add(result)

        print("✅ done", s["name"])

    print("🎉 FINISH")

# ==============================
if __name__ == "__main__":
    run()