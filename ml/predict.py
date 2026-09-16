from pathlib import Path
import subprocess
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ml" / "artifacts" / "ecoroute_model.joblib"
FEATURES = ["mode", "distance_km", "passengers", "occupancy", "traffic_index", "temperature_c", "rain_index", "weekend"]


def predict_one(payload: dict) -> float:
    if payload["mode"] in {"Bicycle", "Walking"}:
        return 0.0
    if not MODEL.exists():
        subprocess.run([sys.executable, str(ROOT / "ml" / "train.py")], check=True)
    model = joblib.load(MODEL)
    row = {key: payload[key] for key in FEATURES}
    return max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
