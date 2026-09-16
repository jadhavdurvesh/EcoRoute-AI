# EcoRoute AI

AI-powered sustainable mobility decision support for comparing transport options, estimating trip emissions, and identifying lower-emission alternatives.

## 1M1B AI for Sustainability
- **Primary SDGs:** SDG 11 – Sustainable Cities and Communities; SDG 13 – Climate Action
- **Core AI:** Random Forest regression trained on mobility features to estimate CO2e per trip.
- **Product:** Interactive Streamlit dashboard with scenario comparison, AI prediction, and recommendation logic.

## Features
- AI-based CO2e prediction for a trip
- Compare car, motorcycle, bus, metro, bicycle, and walking
- Distance, passengers, occupancy, traffic, and weather inputs
- Scenario ranking by predicted emissions
- Estimated emissions avoided by switching options
- Explainable feature-impact panel
- Downloadable comparison CSV

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python ml/train.py
streamlit run app/main.py
```

## Project structure
```text
ecoroute-ai/
├── app/main.py
├── ml/train.py
├── ml/predict.py
├── data/generate_dataset.py
├── docs/project_summary.md
├── requirements.txt
└── README.md
```

## Note on the dataset
The included training dataset is synthetic and is intended for a transparent prototype. Production deployment should replace it with validated local mobility/emissions data.
