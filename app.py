from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "broiler_weight_prediction_pipeline.joblib"
METADATA_PATH = APP_DIR / "broiler_model_metadata.json"

FEATURES = [
    "Bird Age (days)",
    "Feed Intake (g/bird/day)",
    "Water Intake (mL/bird/day)",
    "Temperature (°C)",
    "Humidity (%)",
    "Stocking Density (birds/m²)",
    "Breed Type",
]
NUMERIC_FEATURES = FEATURES[:-1]
BREEDS = ["Ross 308", "Cobb 500", "Arbor Acres Plus"]

st.set_page_config(page_title="Broiler Weight Predictor", page_icon="🐔", layout="wide")
st.title("Broiler Weight Predictor")
st.caption("A decision-support prototype for estimating average flock live weight from routine farm records.")


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_resource(show_spinner="Loading the trained Random Forest model…")
def load_model(path: str):
    return joblib.load(path)


metadata = load_metadata()
model_file = metadata.get("model_file", MODEL_PATH.name)
if model_file != MODEL_PATH.name:
    st.error(f"Metadata expects `{model_file}`, but this app is configured for `{MODEL_PATH.name}`.")
    st.stop()

if not MODEL_PATH.exists():
    st.warning(
        "The trained model artifact is not present yet. Add "
        "`broiler_weight_prediction_pipeline.joblib` and "
        "`broiler_model_metadata.json` to this app directory before running predictions."
    )
    st.info(
        "The attached report selects Random Forest using leave-one-batch-out validation "
        "(mean R² 0.9839; mean MAE 0.0847 kg). This page does not train a substitute model "
        "or generate estimates until the report's fitted pipeline is supplied."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

with st.sidebar:
    st.header("Farm measurements")
    with st.form("prediction_form"):
        age = st.number_input("Bird age (days)", min_value=1, max_value=49, value=35, step=1)
        feed = st.number_input("Feed intake (g/bird/day)", min_value=0.0, value=260.0, step=1.0)
        water = st.number_input("Water intake (mL/bird/day)", min_value=0.0, value=650.0, step=10.0)
        temperature = st.number_input("House temperature (°C)", min_value=10.0, max_value=50.0, value=31.0, step=0.5)
        humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
        density = st.number_input("Stocking density (birds/m²)", min_value=0.1, value=12.5, step=0.5)
        breed = st.selectbox("Breed type", BREEDS)
        submitted = st.form_submit_button("Estimate average weight", type="primary", use_container_width=True)

st.subheader("Estimated flock-average live weight")
if submitted:
    input_values = {
        "Bird Age (days)": float(age),
        "Feed Intake (g/bird/day)": float(feed),
        "Water Intake (mL/bird/day)": float(water),
        "Temperature (°C)": float(temperature),
        "Humidity (%)": float(humidity),
        "Stocking Density (birds/m²)": float(density),
        "Breed Type": breed,
    }
    row = pd.DataFrame([input_values], columns=FEATURES)
    try:
        prediction = float(model.predict(row)[0])
        if not pd.notna(prediction) or prediction < 0:
            raise ValueError("The model returned an invalid weight.")
        st.metric("Predicted average body weight", f"{prediction:.3f} kg")

        observed_ranges = metadata.get("observed_ranges", {})
        warnings = []
        for feature in NUMERIC_FEATURES:
            bounds = observed_ranges.get(feature)
            if bounds and len(bounds) == 2:
                low, high = map(float, bounds)
                value = input_values[feature]
                if value < low or value > high:
                    warnings.append(f"{feature}: {value:g} is outside the training-data range [{low:g}, {high:g}].")
        for warning in warnings:
            st.warning(warning)
        if not warnings:
            st.caption("Inputs fall within the training ranges recorded in model metadata, if available.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.write("Enter the seven farm measurements in the sidebar and select **Estimate average weight**.")

with st.expander("Model and validation context"):
    st.write(
        "**Selected model:** Random Forest Regressor. The report selected it because it led "
        "leave-one-batch-out validation, a more relevant estimate of performance on a new "
        "production sequence than a random row split."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Grouped mean R²", "0.9839")
    c2.metric("Grouped mean MAE", "0.0847 kg")
    c3.metric("Grouped pooled OOF RMSE", "0.1354 kg")
    st.caption(
        "These are reported internal validation results from 11 reconstructed batches across "
        "two Abuja farms—not a guarantee for each prediction or evidence of external validation. "
        "Use as decision support alongside direct weighing and farm-management judgement."
    )

st.divider()
st.caption("Prediction target: average flock body weight (kg), not total flock weight.")
