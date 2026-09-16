from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
VEHICLES = {
    "Car": 0.192,
    "Motorcycle": 0.103,
    "Bus": 0.089,
    "Metro": 0.041,
    "Bicycle": 0.0,
    "Walking": 0.0,
}

rows = []
for _ in range(12000):
    mode = RNG.choice(list(VEHICLES))
    distance = float(np.clip(RNG.gamma(2.2, 5.5), 0.5, 80))
    passengers = 1 if mode in {"Motorcycle", "Bicycle", "Walking", "Metro", "Bus"} else int(RNG.integers(1, 6))
    occupancy = passengers if mode == "Car" else float(np.clip(RNG.normal(18 if mode == "Bus" else 120, 8), 2, 180))
    traffic = float(RNG.uniform(0, 1))
    temperature = float(RNG.uniform(10, 40))
    rain = float(RNG.uniform(0, 1))
    weekend = int(RNG.integers(0, 2))

    base = VEHICLES[mode] * distance
    if mode == "Car":
        base /= max(passengers, 1)
    elif mode == "Bus":
        base *= 18 / max(occupancy, 2)
    elif mode == "Metro":
        base *= 120 / max(occupancy, 20)

    congestion_multiplier = 1 + 0.20 * traffic if mode in {"Car", "Motorcycle", "Bus"} else 1.0
    weather_multiplier = 1 + 0.06 * rain if mode in {"Car", "Motorcycle", "Bus"} else 1.0
    noise = RNG.normal(0, max(0.01, base * 0.035))
    co2e = max(0, base * congestion_multiplier * weather_multiplier + noise)

    rows.append({
        "mode": mode,
        "distance_km": round(distance, 3),
        "passengers": passengers,
        "occupancy": round(occupancy, 2),
        "traffic_index": round(traffic, 3),
        "temperature_c": round(temperature, 2),
        "rain_index": round(rain, 3),
        "weekend": weekend,
        "co2e_kg": round(co2e, 5),
    })

out = Path(__file__).resolve().parent / "synthetic_mobility.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows):,} rows to {out}")
