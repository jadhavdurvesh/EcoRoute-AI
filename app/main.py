import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.model import build_model

st.set_page_config(page_title="EcoRoute AI", page_icon="🌱", layout="wide")

st.title("🌱 EcoRoute AI")
st.caption("AI-powered mobility decisions for lower-carbon trips")

MODES = ["Car", "Motorcycle", "Bus", "Metro", "Bicycle", "Walking"]
DEFAULT_OCC = {"Car": 1, "Motorcycle": 1, "Bus": 18, "Metro": 120, "Bicycle": 1, "Walking": 1}
FEATURES = [
    "mode",
    "distance_km",
    "passengers",
    "occupancy",
    "traffic_index",
    "temperature_c",
    "rain_index",
    "weekend",
]


@st.cache_resource

def get_model():
    return build_model()


# Build the model once. Show a real error instead of leaving the page blank.
try:
    model = get_model()
except Exception as exc:
    st.error("EcoRoute AI could not initialize the model.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Trip scenario")
    distance = st.slider("Distance (km)", 0.5, 80.0, 8.0, 0.5)
    passengers = st.slider("Passengers", 1, 6, 1)
    traffic = st.slider("Traffic intensity", 0.0, 1.0, 0.45, 0.05)
    temperature = st.slider("Temperature (°C)", 0.0, 45.0, 27.0, 1.0)
    rain = st.slider("Rain intensity", 0.0, 1.0, 0.1, 0.05)
    weekend = st.checkbox("Weekend", value=False)
    analyze = st.button("Analyze trip", type="primary", use_container_width=True)


# Keep the latest successful analysis in session state so the UI stays stable.
if "ecoroute_results" not in st.session_state:
    st.session_state.ecoroute_results = None

if analyze or st.session_state.ecoroute_results is None:
    try:
        rows = []
        for mode in MODES:
            occupancy = passengers if mode == "Car" else DEFAULT_OCC[mode]
            payload = {
                "mode": mode,
                "distance_km": distance,
                "passengers": passengers if mode == "Car" else 1,
                "occupancy": occupancy,
                "traffic_index": traffic,
                "temperature_c": temperature,
                "rain_index": rain,
                "weekend": int(weekend),
            }

            if mode in {"Bicycle", "Walking"}:
                value = 0.0
            else:
                frame = pd.DataFrame([{key: payload[key] for key in FEATURES}])
                value = float(model.predict(frame)[0])

            rows.append({
                "Mode": mode,
                "Predicted CO2e (kg)": max(0.0, value),
            })

        results = pd.DataFrame(rows).sort_values("Predicted CO2e (kg)").reset_index(drop=True)
        results["Predicted CO2e (kg)"] = results["Predicted CO2e (kg)"].round(3)
        st.session_state.ecoroute_results = results
    except Exception as exc:
        st.error("The trip analysis failed.")
        st.exception(exc)
        st.stop()

results = st.session_state.ecoroute_results
recommended = results.iloc[0]
car = float(results.loc[results["Mode"] == "Car", "Predicted CO2e (kg)"].iloc[0])
saved = max(0.0, car - float(recommended["Predicted CO2e (kg)"]))

c1, c2, c3 = st.columns(3)
c1.metric("Recommended option", str(recommended["Mode"]))
c2.metric("Predicted emissions", f"{float(recommended['Predicted CO2e (kg)']):.3f} kg CO₂e")
c3.metric("Avoided vs car", f"{saved:.3f} kg CO₂e")

st.divider()
st.subheader("Trip comparison")

for _, row in results.iterrows():
    mode = row["Mode"]
    value = float(row["Predicted CO2e (kg)"])
    st.write(f"**{mode}** — {value:.3f} kg CO₂e")

st.success(
    f"For this scenario, **{recommended['Mode']}** has the lowest predicted emissions among the modeled options."
)

with st.expander("Scenario details"):
    st.write(
        f"Distance: {distance:g} km · Passengers: {passengers} · "
        f"Traffic: {traffic:.2f} · Temperature: {temperature:g} °C · "
        f"Rain: {rain:.2f} · Weekend: {'Yes' if weekend else 'No'}"
    )

st.info(
    "Prototype note: the training data included with this project is synthetic. "
    "For real-world deployment, retrain the model with validated local mobility and emissions data."
)
