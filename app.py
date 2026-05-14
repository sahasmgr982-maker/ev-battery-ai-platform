"""
EV Battery AI Platform — Streamlit web interface
Final Year Project: Designing an AI Platform for Battery Recycling in Electric Vehicles
Author: Sahas Budha (32132820)
Supervisor: Dr Fateme Dinmohammadi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

# ---------------------------------------------------------
# Page configuration — this must be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="EV Battery AI Platform",
    page_icon="🔋",
    layout="wide"
)

# ---------------------------------------------------------
# Load model and data (cached so it doesn't reload on every interaction)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained Random Forest model from disk."""
    model = joblib.load("models/battery_soh_model.pkl")
    with open("models/model_info.json") as f:
        info = json.load(f)
    return model, info

@st.cache_data
def load_data():
    """Load the processed battery features dataset."""
    df = pd.read_csv("data/battery_features.csv")
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df = df[df['type'] == 'discharge'].dropna(subset=['Capacity']).copy()

    # Parse start_time from string representation of array to datetime
    def parse_start_time(s):
        parts = s.strip('[]').split()
        year, month, day, hour, minute, second = map(float, parts)
        return pd.Timestamp(int(year), int(month), int(day), int(hour), int(minute), int(second))

    df['start_time'] = df['start_time'].apply(parse_start_time)

    max_capacities = df.groupby('battery_id')['Capacity'].max()
    df['reference_capacity'] = df['battery_id'].map(max_capacities)
    df['SOH'] = (df['Capacity'] / df['reference_capacity']) * 100
    df = df.sort_values(['battery_id', 'start_time']).reset_index(drop=True)
    df['cycle_number'] = df.groupby('battery_id').cumcount() + 1
    return df

model, model_info = load_model()
df = load_data()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def classify_battery(soh):
    """Return recycling recommendation based on SOH."""
    if soh >= 80:
        return "♻️ Reuse — Second-life ready", "green"
    elif soh >= 60:
        return "🔄 Repurpose — Stationary energy storage", "orange"
    else:
        return "⚙️ Recycle — Material recovery", "red"

# ---------------------------------------------------------
# Sidebar — navigation (NEW: polished card + metric boxes)
# ---------------------------------------------------------
st.sidebar.markdown("""
    <div style="
        background: linear-gradient(180deg, #1f4e79 0%, #2d5d8f 100%);
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        text-align: center;
    ">
        <div style="font-size: 38px;">🔋</div>
        <h3 style="color: white; margin: 5px 0 0 0;">Battery AI Platform</h3>
        <p style="color: #c8d4e0; font-size: 11px; margin: 4px 0 0 0;">Final Year Project</p>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Battery Assessment", "📊 Fleet Dashboard", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.metric("Test R²", f"{model_info['metrics']['R2_test']:.4f}")
st.sidebar.metric("MAE", f"{model_info['metrics']['MAE']:.2f}% SOH")

st.sidebar.markdown("---")
st.sidebar.caption("**Dataset:** NASA Battery Aging")
st.sidebar.caption("**Model:** Random Forest")
st.sidebar.caption("**University of West London**")

# =========================================================
# PAGE 1 — HOME (NEW: banner)
# =========================================================
if page == "🏠 Home":
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #1f4e79 0%, #4a7ab3 100%);
            padding: 30px 40px;
            border-radius: 12px;
            margin-bottom: 25px;
        ">
            <h1 style="color: white; margin: 0; font-size: 36px;">🔋 EV Battery AI Platform</h1>
            <p style="color: #e0e8f0; margin: 8px 0 0 0; font-size: 16px;">
                AI-driven Assessment, Classification &amp; Recycling Recommendations for End-of-Life EV Batteries
            </p>
            <p style="color: #b8c8d8; margin: 12px 0 0 0; font-size: 13px;">
                Final Year Project — Sahas Budha (32132820) — Supervised by Dr Fateme Dinmohammadi
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Batteries in dataset", df['battery_id'].nunique())
    with col2:
        st.metric("Total discharge cycles", f"{len(df):,}")
    with col3:
        st.metric("Model test R²", f"{model_info['metrics']['R2_test']:.4f}")

    st.markdown("---")
    st.markdown("""
    ### Project overview
    This platform implements **Objectives 1 and 2** of the proposed AI-driven battery recycling system:

    1. **Battery Health Assessment** — predicting the State of Health (SOH) of a battery from operational
       data using a Random Forest regressor trained on the NASA Battery Aging dataset.
    2. **Classification & Sorting** — categorising batteries into three recommended pathways:
       *Reuse*, *Repurpose*, or *Recycle*.

    Objectives 3–5 (Disassembly optimisation, Material recovery, Sustainability monitoring) are
    addressed at the **conceptual design** level in the accompanying report.

    ### How to use
    - **🔍 Battery Assessment** — get a prediction and recommendation for a specific battery cycle
    - **📊 Fleet Dashboard** — view aggregate statistics across the full battery fleet
    - **ℹ️ About** — model details, methodology, and acknowledgements
    """)

# =========================================================
# PAGE 2 — BATTERY ASSESSMENT (with manual input mode)
# =========================================================
elif page == "🔍 Battery Assessment":
    st.title("🔍 Battery Health Assessment")
    st.markdown("Predict the State of Health of a battery and receive a recycling recommendation.")

    mode = st.radio(
        "Input mode",
        ["📂 Use existing NASA battery", "✏️ Enter battery readings manually"],
        horizontal=True
    )

    # MODE A — pick from the NASA dataset
    if mode == "📂 Use existing NASA battery":
        col1, col2 = st.columns(2)
        with col1:
            battery_id = st.selectbox("Select battery", sorted(df['battery_id'].unique()))
        with col2:
            battery_subset = df[df['battery_id'] == battery_id]
            cycle = st.slider(
                "Cycle number",
                min_value=int(battery_subset['cycle_number'].min()),
                max_value=int(battery_subset['cycle_number'].max()),
                value=int(battery_subset['cycle_number'].min())
            )

        row = battery_subset[battery_subset['cycle_number'] == cycle].iloc[0]
        X_input = pd.DataFrame([{
            'cycle_number': row['cycle_number'],
            'ambient_temperature': pd.to_numeric(row['ambient_temperature'], errors='coerce'),
            'Capacity': row['Capacity'],
            'reference_capacity': row['reference_capacity']
        }])
        predicted_soh = float(model.predict(X_input)[0])
        actual_soh = float(row['SOH'])
        show_actual = True

    # MODE B — manual input
    else:
        st.info("Enter battery operational readings below. The model will predict the battery's current State of Health.")

        col1, col2 = st.columns(2)
        with col1:
            cycle_in = st.number_input(
                "Cycle number",
                min_value=1, max_value=1000, value=100, step=1,
                help="How many discharge cycles the battery has been through."
            )
            capacity_in = st.number_input(
                "Current discharge capacity (Ah)",
                min_value=0.1, max_value=3.0, value=1.5, step=0.01, format="%.2f",
                help="The capacity measured during the most recent discharge cycle."
            )
        with col2:
            temp_in = st.number_input(
                "Ambient temperature (°C)",
                min_value=-10.0, max_value=60.0, value=24.0, step=0.5, format="%.1f",
                help="Temperature during battery operation."
            )
            ref_capacity_in = st.number_input(
                "Reference (peak) capacity (Ah)",
                min_value=0.5, max_value=3.0, value=1.85, step=0.01, format="%.2f",
                help="The battery's maximum observed capacity (when new or near-new)."
            )

        if capacity_in > ref_capacity_in:
            st.warning("⚠ Current capacity is higher than reference capacity. Did you swap the values?")

        X_input = pd.DataFrame([{
            'cycle_number': cycle_in,
            'ambient_temperature': temp_in,
            'Capacity': capacity_in,
            'reference_capacity': ref_capacity_in
        }])
        predicted_soh = float(model.predict(X_input)[0])
        actual_soh = (capacity_in / ref_capacity_in) * 100
        show_actual = False
        battery_id = None

    # SHARED OUTPUT
    st.markdown("---")
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("Predicted SOH", f"{predicted_soh:.2f}%")
    if show_actual:
        rcol2.metric("Actual SOH (NASA)", f"{actual_soh:.2f}%")
        rcol3.metric("Prediction error", f"{abs(predicted_soh - actual_soh):.2f}%")
    else:
        rcol2.metric("Naive SOH (capacity ratio)", f"{actual_soh:.2f}%")
        rcol3.metric("Difference", f"{abs(predicted_soh - actual_soh):.2f}%")

    recommendation, colour = classify_battery(predicted_soh)
    if colour == "green":
        st.success(f"### Recommendation: {recommendation}")
        st.info("✓ This battery retains sufficient capacity for direct reuse in another EV or as a spare module.")
    elif colour == "orange":
        st.warning(f"### Recommendation: {recommendation}")
        st.info("⚠ Capacity is below EV operating standards but suitable for second-life applications such as residential or grid-scale energy storage.")
    else:
        st.error(f"### Recommendation: {recommendation}")
        st.info("✗ Capacity is below practical use thresholds. Battery should be processed for material recovery (lithium, cobalt, nickel).")

    # Degradation chart (dataset mode only)
    if mode == "📂 Use existing NASA battery" and battery_id is not None:
        st.markdown("---")
        st.subheader(f"Capacity-degradation profile — Battery {battery_id}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(battery_subset['cycle_number'], battery_subset['SOH'],
                color='#1f4e79', linewidth=2, label='SOH over cycles')
        ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, label='80% threshold (Reuse → Repurpose)')
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1, label='60% threshold (Repurpose → Recycle)')
        ax.axvline(x=cycle, color='black', linestyle=':', alpha=0.6, label=f'Selected cycle ({cycle})')
        ax.scatter([cycle], [predicted_soh], color='red', s=80, zorder=5, label=f'Prediction ({predicted_soh:.1f}%)')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('State of Health (%)')
        ax.set_title(f'Battery {battery_id} — Lifetime SOH Trajectory')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# =========================================================
# PAGE 3 — FLEET DASHBOARD
# =========================================================
elif page == "📊 Fleet Dashboard":
    st.title("📊 Fleet Sustainability Dashboard")
    st.markdown("Aggregate view across all batteries — useful for fleet operators and recycling planners.")

    latest_per_battery = df.sort_values('cycle_number').groupby('battery_id').tail(1).copy()
    latest_per_battery['recommendation'] = latest_per_battery['SOH'].apply(
        lambda x: 'Reuse' if x >= 80 else ('Repurpose' if x >= 60 else 'Recycle')
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total batteries", len(latest_per_battery))
    k2.metric("Avg final SOH", f"{latest_per_battery['SOH'].mean():.1f}%")
    k3.metric("Reuse-ready", (latest_per_battery['recommendation'] == 'Reuse').sum())
    k4.metric("Needs recycling", (latest_per_battery['recommendation'] == 'Recycle').sum())

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Recommendation breakdown")
        counts = latest_per_battery['recommendation'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        colours = {'Reuse': '#4caf50', 'Repurpose': '#ff9800', 'Recycle': '#f44336'}
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=[colours.get(c, '#888888') for c in counts.index],
               startangle=90)
        ax.set_title('Recycling Recommendations (Fleet End-State)')
        st.pyplot(fig)

    with col_b:
        st.subheader("SOH distribution at end-of-life")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(latest_per_battery['SOH'], bins=15, color='#1f4e79', edgecolor='white')
        ax.axvline(80, color='orange', linestyle='--', label='80% threshold')
        ax.axvline(60, color='red', linestyle='--', label='60% threshold')
        ax.set_xlabel('Final SOH (%)')
        ax.set_ylabel('Number of batteries')
        ax.set_title('Distribution of Final State of Health')
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("♻️ Estimated material recovery (if recycled today)")
    n_recycle = (latest_per_battery['recommendation'] == 'Recycle').sum()
    kg_lithium_per_battery   = 1.2
    kg_cobalt_per_battery    = 1.8
    kg_nickel_per_battery    = 3.5
    co2_avoided_per_battery  = 50

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lithium recoverable", f"{n_recycle * kg_lithium_per_battery:.1f} kg")
    m2.metric("Cobalt recoverable", f"{n_recycle * kg_cobalt_per_battery:.1f} kg")
    m3.metric("Nickel recoverable", f"{n_recycle * kg_nickel_per_battery:.1f} kg")
    m4.metric("CO₂ emissions avoided", f"{n_recycle * co2_avoided_per_battery:.0f} kg")

    st.caption("Values are illustrative estimates based on typical Li-ion battery composition (Harper et al., 2019). Actual yields depend on battery chemistry and recycling process.")

# =========================================================
# PAGE 4 — ABOUT
# =========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About this platform")

    st.markdown("""
    ### Project information
    - **Title:** Designing an AI Platform for Battery Recycling in Electric Vehicles
    - **Student:** Sahas Budha (32132820)
    - **Supervisor:** Dr Fateme Dinmohammadi
    - **Course:** Computer Science, University of West London

    ### Methodology
    - **Dataset:** NASA Battery Aging Dataset (Saha & Goebel, 2007)
    - **Model:** Random Forest Regressor (100 estimators, max depth 15)
    - **Features:** Cycle number, ambient temperature, current capacity, reference capacity
    - **Target:** State of Health (SOH) — percentage of peak observed capacity

    ### Model performance
    """)
    metrics = model_info['metrics']
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Test R²", f"{metrics['R2_test']:.4f}")
    mc2.metric("MAE", f"{metrics['MAE']:.3f}% SOH")
    mc3.metric("RMSE", f"{metrics['RMSE']:.3f}% SOH")

    st.markdown("""
    ### Project scope
    This platform implements **Objectives 1 and 2** of the original proposal (Battery Health Assessment
    and Classification & Sorting). Objectives 3–5 (Disassembly Optimisation, Material Recovery,
    Sustainability Monitoring) are addressed conceptually in the accompanying report, consistent with
    the scope-management approach declared in the project risk register.

    ### Limitations
    - The NASA dataset uses controlled laboratory conditions; real-world EV batteries face more varied
      stress profiles.
    - Ambient temperature contributed only ~1% to predictions in this dataset, but is known to be
      a significant aging factor in deployed batteries — richer datasets would improve generalisation.
    - The recycling recommendations use industry-standard 80% and 60% thresholds; actual decisions
      would require additional safety and economic analysis.
    """)