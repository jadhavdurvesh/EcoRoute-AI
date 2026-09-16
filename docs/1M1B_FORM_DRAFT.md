# 1M1B Submission Draft — EcoRoute AI

## Title of the Project
EcoRoute AI — AI-Powered Sustainable Mobility Decision Support

## SDGs Aligned
- SDG 11: Sustainable Cities and Communities
- SDG 13: Climate Action

## Technologies Used
Python, Streamlit, scikit-learn, Random Forest regression, pandas, NumPy, Plotly, joblib

## Problem Statement (75–150 words)
Urban mobility choices are often made without a simple way to compare the environmental impact of different transport options for the same journey. People may understand that public transport, walking, cycling, or shared mobility can reduce emissions, but they lack an accessible scenario-specific estimate that considers factors such as distance, occupancy and traffic. This creates a gap between sustainability awareness and practical decision-making. EcoRoute AI addresses this problem by providing an interactive way to estimate and compare trip-level CO2e emissions across multiple transport modes. The project is designed as an AI-based decision-support prototype that can later be retrained using validated local mobility and emissions data to support city-specific use cases.

## Solution Including AI Elements
EcoRoute AI is an interactive dashboard that takes trip conditions such as distance, transport mode, passenger count, occupancy, traffic intensity, temperature, rainfall and weekend context as inputs. A Random Forest regression model predicts the expected CO2e emissions for each transport option. The system compares the predictions, identifies the lowest-emission modeled option, and estimates emissions avoided relative to a car trip. The prototype uses a transparent synthetic dataset so that the complete AI workflow can run without a paid external API. The architecture is designed for future retraining on validated local datasets.

## Target Users
Students, commuters, sustainability teams, educational institutions and urban-mobility teams.

## Anticipated / Actual Impact
The prototype makes the environmental trade-offs between transport choices visible at the point of decision. It can help users understand how distance, occupancy and travel conditions affect trip emissions and identify lower-emission alternatives. With validated local data, the system could be adapted for city-specific awareness, educational and mobility-planning use cases.

## Links
GitHub: [add final repository URL]
Demo: [add deployed Streamlit URL]
Video: [add 60–90 second demo URL]

## Additional Information
The project was designed as an AI-for-sustainability prototype with an emphasis on transparency and reproducibility. The included training data is synthetic and should be replaced by validated local mobility/emissions data before operational deployment.
