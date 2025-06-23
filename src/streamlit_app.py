import streamlit as st
import os

# Define paths
dl_mean_path = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/plots/ablations/1750681434"
dl_var_path = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/plots/ablations/1750681370"
baselines_mean_path = (
    "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/plots/ablations/1750681524"
)
baselines_var_path = (
    "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/plots/ablations/1750681626"
)

# Load HTML file names (ensure sorted for consistent pairing)
dl_mean_files = sorted(
    [f"{dl_mean_path}/{f}" for f in os.listdir(dl_mean_path) if f.endswith(".html")]
)
dl_var_files = sorted(
    [f"{dl_var_path}/{f}" for f in os.listdir(dl_var_path) if f.endswith(".html")]
)
baselines_mean_files = sorted(
    [
        f"{baselines_mean_path}/{f}"
        for f in os.listdir(baselines_mean_path)
        if f.endswith(".html")
    ]
)
baselines_var_files = sorted(
    [
        f"{baselines_var_path}/{f}"
        for f in os.listdir(baselines_var_path)
        if f.endswith(".html")
    ]
)

st.set_page_config(layout="wide")

st.title("DL vs Baseline Forecasting Comparison")

# Assume all lists have 3 plots
for i in range(3):
    st.subheader(f"Prediction Window {i + 1}")

    col1, col2 = st.columns(2)

    H = 800
    W = 1600

    with col1:
        st.markdown("**DL: Mean Prediction**")
        with open(dl_mean_files[i], "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=H, width=W, scrolling=True)

        st.markdown("**Baseline: Mean Prediction**")
        with open(baselines_mean_files[i], "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=H, width=W, scrolling=True)

    with col2:
        st.markdown("**DL: Variance Prediction**")
        with open(dl_var_files[i], "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=H, width=W, scrolling=True)

        st.markdown("**Baseline: Variance Prediction**")
        with open(baselines_var_files[i], "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=H, width=W, scrolling=True)

    st.markdown("---")  # Divider between sets
