from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_mobility.csv"
MODEL_DIR = ROOT / "ml" / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
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
    ("regressor", RandomForestRegressor(n_estimators=240, random_state=42, n_jobs=-1, min_samples_leaf=2)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {"mae_kg": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)), "rows": int(len(df))}
joblib.dump(model, MODEL_DIR / "ecoroute_model.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
