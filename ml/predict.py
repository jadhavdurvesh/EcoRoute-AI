from pathlib import Path
import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parent / "artifacts" / "ecoroute_model.joblib"
FEATURES = ["mode", "distance_km", "passengers", "occupancy", "traffic_index", "temperature_c", "rain_index", "weekend"]

def predict_one(payload: dict) -> float:
    if payload["mode"] in {"Bicycle", "Walking"}:
        return 0.0
    if not MODEL.exists():
        raise FileNotFoundError("Model artifact not found. Run python ml/train.py first.")
    model = joblib.load(MODEL)
    row = {key: payload[key] for key in FEATURES}
    return max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
