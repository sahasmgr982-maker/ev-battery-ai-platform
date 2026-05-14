# EV Battery AI Platform

An AI-driven platform for assessing and classifying end-of-life electric vehicle batteries.

**Final Year Project — BSc (Hons) Computer Science**
**University of West London**

- **Author:** Sahas Budha (32132820)
- **Supervisor:** Dr Fateme Dinmohammadi
- **Submission:** May 2026

---

## Overview

This project implements the data-driven core of an AI platform for managing end-of-life decisions for EV lithium-ion batteries. A Random Forest model trained on the NASA Battery Aging Dataset predicts each battery's State of Health (SOH), and a Streamlit web application maps that prediction to one of three recommended pathways:

- **Reuse** (SOH ≥ 80%) — second-life ready
- **Repurpose** (60–80%) — stationary energy storage
- **Recycle** (< 60%) — material recovery

## Model Performance

| Metric | Value |
|---|---|
| Test R² | 0.9967 |
| MAE | 0.29% SOH |
| RMSE | 0.73% SOH |
| Predictions within 5% | 99.6% |

## Quick Start

1. Clone the repository:

       git clone https://github.com/sahasmgr982-maker/ev-battery-ai-platform.git
       cd ev-battery-ai-platform

2. Install dependencies:

       pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

3. Run the web platform:

       streamlit run app.py

4. The platform will open in your browser at http://localhost:8501

## Project Structure

- app.py — Streamlit web application
- .streamlit/config.toml — Theme configuration
- notebooks/01_explore_data.ipynb — Data exploration and SOH calculation
- notebooks/02_train_model.ipynb — Model training and evaluation
- data/battery_features.csv — Cleaned dataset
- models/battery_soh_model.pkl — Trained Random Forest model
- models/model_info.json — Model metadata and performance metrics
- figures/ — Charts used in the report

## Dataset

The raw NASA Battery Aging Dataset is publicly available at:
https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

The cleaned battery_features.csv in this repository is derived from that source.

## Citation

Budha, S. (2026) Designing an AI Platform for Battery Recycling in Electric Vehicles. BSc Final Year Project, University of West London.