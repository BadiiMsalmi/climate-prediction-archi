# model_api.py
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine

PGUSER = os.getenv("PGUSER", "airuser")
PGPASS = os.getenv("PGPASS", "airpass")
PGHOST = os.getenv("PGHOST", "postgres")
PGPORT = os.getenv("PGPORT", "5432")
PGDB   = os.getenv("PGDB", "air_quality")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/saved_models")

engine = create_engine(f"postgresql://{PGUSER}:{PGPASS}@{PGHOST}:{PGPORT}/{PGDB}")

app = Flask(__name__)

# in-memory cache
_cached = {"pointer_path": None, "model": None, "scaler_X": None, "scaler_y": None}

def load_current():
    pointer_file = os.path.join(MODEL_DIR, "current_model.json")
    if not os.path.exists(pointer_file):
        raise FileNotFoundError("current_model.json not found in MODEL_DIR")
    with open(pointer_file, "r") as f:
        pointer = json.load(f)
    model_path = os.path.join(pointer["path"], "model.h5")
    if _cached["pointer_path"] != pointer["path"]:
        print("Loading new model from", pointer["path"])
        _cached["model"] = load_model(model_path)
        _cached["scaler_X"] = joblib.load(os.path.join(pointer["path"], "scaler_X.pkl"))
        _cached["scaler_y"] = joblib.load(os.path.join(pointer["path"], "scaler_y.pkl"))
        _cached["pointer_path"] = pointer["path"]
    return _cached["model"], _cached["scaler_X"], _cached["scaler_y"]

@app.route("/predict", methods=["GET"])
def predict_next_hour():
    # load last INPUT_STEPS hours from DB
    INPUT_STEPS = int(os.getenv("INPUT_STEPS", "24"))
    df = pd.read_sql("SELECT timestamp, temperature, humidity FROM temperature_hourly ORDER BY timestamp DESC LIMIT {}".format(INPUT_STEPS), engine, parse_dates=['timestamp'])
    if df.empty or len(df) < INPUT_STEPS:
        return jsonify({"error": "not enough data"}), 400
    df = df.sort_values("timestamp")
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    X = df[['temperature', 'humidity', 'hour', 'dow']].astype('float32')

    model, scaler_X, scaler_y = load_current()
    X_s = scaler_X.transform(X)
    X_in = np.expand_dims(X_s, axis=0)
    pred_s = model.predict(X_in)
    pred = scaler_y.inverse_transform(pred_s)[0][0]

    # optional: save prediction to DB table predictions (timestamp, predicted, actual=NULL initially)
    try:
        dfp = pd.DataFrame([{
            "pred_ts": pd.Timestamp.utcnow(),
            "predicted_temperature": float(pred)
        }])
        dfp.to_sql("predictions", engine, if_exists="append", index=False)
    except Exception as e:
        print("Could not write prediction to DB:", e)

    return jsonify({"predicted_temperature_next_hour": round(float(pred), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
