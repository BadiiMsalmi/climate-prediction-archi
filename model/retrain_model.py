import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------
# Config 
# -------------------
PGUSER = os.getenv("PGUSER", "airuser")
PGPASS = os.getenv("PGPASS", "airpass")
PGHOST = os.getenv("PGHOST", "postgres")
PGPORT = os.getenv("PGPORT", "5432")
PGDB   = os.getenv("PGDB", "air_quality")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/saved_models")  
HISTORY_HOURS = int(os.getenv("HISTORY_HOURS", "24*365*5"))  
INPUT_STEPS = int(os.getenv("INPUT_STEPS", "24"))
HORIZON = int(os.getenv("HORIZON", "1"))

os.makedirs(MODEL_DIR, exist_ok=True)

engine = create_engine(f"postgresql://{PGUSER}:{PGPASS}@{PGHOST}:{PGPORT}/{PGDB}")

def load_data(recent_hours=None):
    # Load timestamp-ordered data; you can restrict to last N hours if needed
    query = "SELECT timestamp, temperature, humidity FROM temperature_hourly ORDER BY timestamp"
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    if df.empty:
        raise ValueError("No data found in table temperature_hourly.")
    df = df.set_index('timestamp').resample('H').mean().ffill().bfill()
    if recent_hours:
        cutoff = df.index.max() - pd.Timedelta(hours=recent_hours)
        df = df[df.index >= cutoff]
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    data = df[['temperature', 'humidity', 'hour', 'dow']].astype('float32')
    return data, df.index

def make_windows(arr, input_steps=24, horizon=1):
    X, y = [], []
    T = arr.shape[0]
    for i in range(T - input_steps - horizon + 1):
        X.append(arr[i:i+input_steps])
        y.append(arr[i+input_steps:i+input_steps+horizon, 0])
    return np.array(X), np.array(y)

def build_base_model(input_shape, lr=1e-3):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(HORIZON)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model

def save_model_atomic(model, scaler_X, scaler_y, model_dir=MODEL_DIR):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = os.path.join(model_dir, f"model_{ts}")
    os.makedirs(target, exist_ok=True)
    model_path = os.path.join(target, "model.h5")
    model.save(model_path)
    joblib.dump(scaler_X, os.path.join(target, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(target, "scaler_y.pkl"))
    pointer = {"path": target, "saved_at": ts}
    with open(os.path.join(model_dir, "current_model.json"), "w") as f:
        json.dump(pointer, f)
    print("Saved model to:", target)
    return target

def evaluate_and_log(y_true, y_pred, save_path):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"VAL MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    try:
        dfm = pd.DataFrame([{
            "saved_at": datetime.utcnow(),
            "path": save_path,
            "mae": float(mae),
            "rmse": float(rmse)
        }])
        dfm.to_sql("model_metrics", engine, if_exists="append", index=False)
    except Exception as e:
        print("Warning: couldn't write metrics to DB:", e)
    return mae, rmse

def main():
    print("Loading data...")
    data, idx = load_data(recent_hours=None)  
    n = len(data)
    if n < (INPUT_STEPS + HORIZON + 10):
        raise ValueError("Not enough data to make windows. Need more rows.")
    train_size = int(n * 0.8)
    train_df = data.iloc[:train_size]
    val_df = data.iloc[train_size:]

    # Fit scalers on train set 
    scaler_X = StandardScaler().fit(train_df)
    scaler_y = StandardScaler().fit(train_df[['temperature']])

    X_train_s = scaler_X.transform(train_df)
    X_val_s = scaler_X.transform(val_df)

    X_train, y_train = make_windows(X_train_s, INPUT_STEPS, HORIZON)
    X_val, y_val = make_windows(X_val_s, INPUT_STEPS, HORIZON)
    print("Window shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # Load existing model if available -> fine-tune. Otherwise build new.
    current_pointer = os.path.join(MODEL_DIR, "current_model.json")
    if os.path.exists(current_pointer):
        try:
            with open(current_pointer, "r") as f:
                pointer = json.load(f)
            existing_model_path = os.path.join(pointer["path"], "model.h5")
            print("Loading existing model for fine-tuning:", existing_model_path)
            model = load_model(existing_model_path)
            model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
        except Exception as e:
            print("Could not load existing model, building new. Error:", e)
            model = build_base_model(input_shape=(INPUT_STEPS, X_train.shape[2]), lr=1e-3)
    else:
        print("No existing model found. Building a new model.")
        model = build_base_model(input_shape=(INPUT_STEPS, X_train.shape[2]), lr=1e-3)

    # Callbacks
    tmp_checkpoint = os.path.join(MODEL_DIR, "tmp_best.h5")
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(tmp_checkpoint, save_best_only=True)
    ]

    # Fine-tune
    EPOCHS = 5
    BATCH = 64
    print("Starting training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=1)

    # evaluate on val
    y_pred_s = model.predict(X_val)
    # inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_s)
    y_true = scaler_y.inverse_transform(y_val)
    # log and save
    save_path = save_model_atomic(model, scaler_X, scaler_y)
    evaluate_and_log(y_true, y_pred, save_path)
    print("Retrain complete.")

if __name__ == "__main__":
    main()
