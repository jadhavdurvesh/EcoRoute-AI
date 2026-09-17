import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.model import build_model

st.set_page_config(page_title="EcoRoute AI", page_icon="🌱", layout="wide")
st.markdown("# 🌱 EcoRoute AI")
st.caption("AI-powered mobility decisions for lower-carbon trips")

model = build_model()

with st.sidebar:
    st.header("Trip scenario")
    distance = st.slider("Distance (km)", 0.5, 80.0, 8.0, 0.5)
    passengers = st.slider("Passengers", 1, 6, 1)
    traffic = st.slider("Traffic intensity", 0.0, 1.0, 0.45, 0.05)
    temperature = st.slider("Temperature (°C)", 0.0, 45.0, 27.0, 1.0)
    rain = st.slider("Rain intensity", 0.0, 1.0, 0.1, 0.05)
    weekend = st.checkbox("Weekend", value=False)

MODES = ["Car", "Motorcycle", "Bus", "Metro", "Bicycle", "Walking"]
DEFAULT_OCC = {"Car": 1, "Motorcycle": 1, "Bus": 18, "Metro": 120, "Bicycle": 1, "Walking": 1}
FEATURES = ["mode", "distance_km", "passengers", "occupancy", "traffic_index", "temperature_c", "rain_index", "weekend"]

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
    value = 0.0 if mode in {"Bicycle", "Walking"} else float(model.predict(pd.DataFrame([{k: payload[k] for k in FEATURES}]))[0])
    rows.append({"Mode": mode, "Predicted CO2e (kg)": max(0.0, value)})

results = pd.DataFrame(rows).sort_values("Predicted CO2e (kg)").reset_index(drop=True)
results["Predicted CO2e (kg)"] = results["Predicted CO2e (kg)"].round(3)
recommended = results.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Recommended option", recommended["Mode"])
c2.metric("Predicted emissions", f"{recommended['Predicted CO2e (kg)']:.3f} kg CO₂e")
car = float(results.loc[results["Mode"] == "Car", "Predicted CO2e (kg)"].iloc[0])
saved = max(0.0, car - float(recommended["Predicted CO2e (kg)"]))
c3.metric("Avoided vs car", f"{saved:.3f} kg CO₂e")

st.divider()
left, right = st.columns([1.4, 1])
with left:
    fig = px.bar(results, x="Mode", y="Predicted CO2e (kg)", title="AI-predicted trip emissions", text_auto=".3f")
    fig.update_layout(yaxis_title="kg CO₂e", xaxis_title="")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Recommendation")
    st.success(f"For this scenario, **{recommended['Mode']}** has the lowest predicted emissions among the modeled options.")
    st.write("The prediction considers transport mode, trip distance, occupancy, traffic, temperature, rainfall and weekend context.")
    with st.expander("Why these features matter"):
        st.write("Distance and transport mode drive most of the baseline emissions. Occupancy can reduce per-passenger emissions for shared transport, while congestion and weather can increase road-transport energy use.")

st.subheader("Compare scenarios")
st.dataframe(results, width="stretch", hide_index=True)
st.download_button("Download comparison CSV", results.to_csv(index=False), "ecoroute_comparison.csv", "text/csv")
st.info("Prototype note: the training data included with this project is synthetic. For deployment in a real city, retrain the model with validated local mobility and emissions data.")
