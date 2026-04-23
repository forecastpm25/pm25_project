import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.arima.model import ARIMA

# ===== CONFIG =====
LOOKBACK = 48
HORIZONS = list(range(1,25)) + [48,72]

STATIONS = [
    {"name": "station_5030", "file": "data/kaho_data.csv"},
    {"name": "station_3295", "file": "data/taksin_data.csv"}
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

def preprocess(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
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

def create_dataset(x, y, lookback, horizons):
    X, Y = [], []
    max_h = max(horizons)

    for i in range(len(x) - lookback - max_h):
        X.append(x[i:i+lookback])
        Y.append([y[i+lookback+h-1, 0] for h in horizons])

    return np.array(X), np.array(Y)

def build_model(output_dim):
    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(output_dim)
    ])
    model.compile(optimizer="adam", loss="huber")
    return model

# ===== TRAIN =====
for s in STATIONS:
    print(f"\n===== TRAIN {s['name']} =====")

    df = pd.read_csv(s["file"])
    df = preprocess(df)

    df["pm2.5_log"] = np.log1p(df["pm2.5"])

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    train_x = x_scaler.fit_transform(train_df[FEATURES])
    train_y = y_scaler.fit_transform(train_df[["pm2.5_log"]])

    X_train, y_train = create_dataset(train_x, train_y, LOOKBACK, HORIZONS)

    model = build_model(len(HORIZONS))

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # ===== RESIDUAL =====
    preds = model.predict(X_train, verbose=0)
    residual = y_train[:,0] - preds[:,0]

    # ===== ARIMA =====
    arima_model = ARIMA(residual, order=(1,0,1)).fit()

    save_path = f"models/{s['name']}"
    os.makedirs(save_path, exist_ok=True)

    model.save(f"{save_path}/lstm_model.h5")
    joblib.dump(x_scaler, f"{save_path}/x_scaler.pkl")
    joblib.dump(y_scaler, f"{save_path}/y_scaler.pkl")
    joblib.dump(arima_model, f"{save_path}/arima.pkl")

    print(f"✅ Saved {s['name']}")

print("🎉 Done")