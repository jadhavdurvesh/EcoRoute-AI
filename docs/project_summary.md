# EcoRoute AI — Project Summary

## Problem
Urban mobility decisions are often made without a simple way to compare the environmental impact of different transport options for the same trip. People may know that some modes are greener than others, but they lack an accessible, scenario-specific estimate that accounts for distance, occupancy and travel conditions.

## Solution
EcoRoute AI is an AI-powered decision-support dashboard. A user describes a trip, and a machine-learning model predicts CO2e emissions for several transport modes. The system compares the modeled outcomes, highlights the lowest-emission option, and estimates potential emissions avoided relative to a car trip.

## AI
The prototype uses a Random Forest regression model. Inputs include transport mode, distance, passenger count, occupancy, traffic intensity, temperature, rainfall and weekend context. The model is trained on a transparent synthetic dataset so the prototype can run without paid APIs. The architecture is designed so validated local mobility data can later replace the prototype dataset.

## Target users
Students, commuters, sustainability teams, educational institutions and urban-mobility teams can use the prototype for awareness, scenario analysis and early-stage decision support.

## Intended impact
The project is designed to make transport-emissions trade-offs visible at the point of decision. In a production implementation, city-specific data could support more accurate local estimates and help users identify lower-emission travel choices.

## SDGs
- SDG 11: Sustainable Cities and Communities
- SDG 13: Climate Action
