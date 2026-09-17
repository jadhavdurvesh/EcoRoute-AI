from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@st.cache_resource(show_spinner="Training EcoRoute AI model once for this app session...")
def build_model():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "synthetic_mobility.csv"

    if not data_path.exists():
        raise FileNotFoundError("Prototype dataset not found: data/synthetic_mobility.csv")

    df = pd.read_csv(data_path)
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
            n_estimators=60,
            max_depth=10,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        )),
    ])

    model.fit(X, y)
    return model
