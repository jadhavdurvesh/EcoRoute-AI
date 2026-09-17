import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model():
    rng = np.random.default_rng(42)
    modes = np.array(["Car", "Motorcycle", "Bus", "Metro"])
    factors = {"Car": 0.192, "Motorcycle": 0.103, "Bus": 0.089, "Metro": 0.041}

    rows = []
    for _ in range(1800):
        mode = str(rng.choice(modes))
        distance = float(rng.uniform(0.5, 50.0))
        passengers = int(rng.integers(1, 6)) if mode == "Car" else 1
        occupancy = float(passengers if mode == "Car" else (18 if mode == "Bus" else 120 if mode == "Metro" else 1))
        traffic = float(rng.uniform(0.0, 1.0))
        temperature = float(rng.uniform(10.0, 40.0))
        rain = float(rng.uniform(0.0, 1.0))
        weekend = int(rng.integers(0, 2))

        emissions = factors[mode] * distance
        if mode == "Car":
            emissions /= max(passengers, 1)
        elif mode == "Bus":
            emissions *= 18.0 / occupancy
        elif mode == "Metro":
            emissions *= 120.0 / occupancy

        emissions *= 1.0 + 0.20 * traffic
        emissions *= 1.0 + 0.06 * rain
        emissions += float(rng.normal(0.0, max(0.008, emissions * 0.03)))

        rows.append({
            "mode": mode,
            "distance_km": distance,
            "passengers": passengers,
            "occupancy": occupancy,
            "traffic_index": traffic,
            "temperature_c": temperature,
            "rain_index": rain,
            "weekend": weekend,
            "co2e_kg": max(0.0, emissions),
        })

    df = pd.DataFrame(rows)
    X = df.drop(columns=["co2e_kg"])
    y = df["co2e_kg"]

    categorical = ["mode"]
    numerical = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical),
    ])

    model = Pipeline([
        ("preprocess", pre),
        ("regressor", RandomForestRegressor(
            n_estimators=35,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=3,
        )),
    ])
    model.fit(X, y)
    return model
