# initial_train.py
import os
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Config --------
PGUSER = os.getenv("PGUSER", "airuser")
PGPASS = os.getenv("PGPASS", "airpass")
PGHOST = os.getenv("PGHOST", "postgres")
PGPORT = os.getenv("PGPORT", "5432")
PGDB   = os.getenv("PGDB", "air_quality")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/saved_models")
INPUT_STEPS = int(os.getenv("INPUT_STEPS", "24"))
HORIZON = int(os.getenv("HORIZON", "1"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH = int(os.getenv("BATCH", "64"))

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Connecting to Postgres at {PGHOST}:{PGPORT}/{PGDB} ...")
engine = create_engine(f"postgresql://{PGUSER}:{PGPASS}@{PGHOST}:{PGPORT}/{PGDB}")

# -------- Load & preprocess --------
print("Loading data from temperature_hourly table...")
query = "SELECT timestamp, temperature, humidity FROM temperature_hourly ORDER BY timestamp"
df = pd.read_sql(query, engine, parse_dates=['timestamp'])
if df.empty:
    raise ValueError("No data found in table temperature_hourly.")

df = df.set_index('timestamp').resample('H').mean().ffill().bfill()
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek
data = df[['temperature', 'humidity', 'hour', 'dow']].astype('float32')
print(f" Data shape after preprocessing: {data.shape}")

# -------- Split and scale --------
n = len(data)
train_size = int(n * 0.8)
train_df = data.iloc[:train_size]
val_df = data.iloc[train_size:]

scaler_X = StandardScaler().fit(train_df)
scaler_y = StandardScaler().fit(train_df[['temperature']])

X_train_s = scaler_X.transform(train_df)
X_val_s   = scaler_X.transform(val_df)

# -------- Window function --------
def make_windows(arr, input_steps=24, horizon=1):
    X, y = [], []
    T = arr.shape[0]
    for i in range(T - input_steps - horizon + 1):
        X.append(arr[i:i+input_steps])
        y.append(arr[i+input_steps:i+input_steps+horizon, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(X_train_s, INPUT_STEPS, HORIZON)
X_val, y_val = make_windows(X_val_s, INPUT_STEPS, HORIZON)
print(f"Windowed training data shape: X_train={X_train.shape}, y_train={y_train.shape}")

# -------- Build model --------
model = Sequential([
    LSTM(64, input_shape=(INPUT_STEPS, X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(HORIZON)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# -------- Callbacks and Train --------
tmp_checkpoint = os.path.join(MODEL_DIR, "tmp_best.h5")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(tmp_checkpoint, save_best_only=True)
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# -------- Eval and plot --------
print("Evaluating model...")
y_pred_s = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = scaler_y.inverse_transform(y_val)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"VAL MAE: {mae:.3f}, RMSE: {rmse:.3f}")

plt.figure(figsize=(10, 4))
plt.plot(y_true[:200], label='True')
plt.plot(y_pred[:200], label='Predicted')
plt.title("Temperature Prediction (Validation)")
plt.legend()
plt.tight_layout()

# -------- Save versioned model + scalers atomically --------
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
target = os.path.join(MODEL_DIR, f"model_{ts}")
os.makedirs(target, exist_ok=True)
model_path = os.path.join(target, "model.h5")
model.save(model_path)
joblib.dump(scaler_X, os.path.join(target, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(target, "scaler_y.pkl"))
plt.savefig(os.path.join(target, "val_predictions.png"))

# update pointer file
pointer = {"path": target, "saved_at": ts}
with open(os.path.join(MODEL_DIR, "current_model.json"), "w") as f:
    json.dump(pointer, f)

# log metrics to DB 
try:
    dfm = pd.DataFrame([{
        "saved_at": datetime.utcnow(),
        "path": target,
        "mae": float(mae),
        "rmse": float(rmse)
    }])
    dfm.to_sql("model_metrics", engine, if_exists="append", index=False)
except Exception as e:
    print("Warning: couldn't write metrics to DB:", e)

print("Saved model:", target)
print("Initial training complete.")
