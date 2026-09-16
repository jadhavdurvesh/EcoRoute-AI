# EcoRoute AI

AI-powered mobility sustainability analysis for the 1M1B AI for Sustainability Virtual Internship.

## What it does
- Predicts trip-level CO2e emissions with a machine-learning model.
- Compares common transport modes.
- Recommends lower-emission alternatives.
- Estimates potential emissions avoided versus car travel.
- Provides an interactive Streamlit interface.

## Sustainability alignment
- SDG 11: Sustainable Cities and Communities
- SDG 13: Climate Action

## Project status
Prototype / educational project prepared for the 1M1B AI for Sustainability Virtual Internship.

## Repository structure
- `app/` — Streamlit application
- `src/` — model and feature logic
- `data/` — prototype dataset generation/input
- `models/` — trained model artifacts
- `notebooks/` — experiments and validation
- `docs/` — submission and demo materials
- `tests/` — automated checks

## Run locally
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model note
Reported validation metrics are based on the project's prototype/synthetic dataset and should not be interpreted as real-world emissions accuracy.
