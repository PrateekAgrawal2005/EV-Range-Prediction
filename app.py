"""Streamlit interface for the final EV range SVR pipeline.

The model is loaded as-is.  This app never fits, changes, or saves the
pipeline; it only sends the exact 17 training features to ``predict``.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("ev_range_svr_pipeline.joblib")

# Keep this list in the exact training order. Do not add leakage/target fields.
FEATURE_COLUMNS = [
    "top_speed_kmh",
    "battery_capacity_kWh",
    "number_of_cells",
    "torque_nm",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm",
    "brand",
    "fast_charge_port",
    "drivetrain",
    "segment",
    "car_body_type",
]

CATEGORICAL_COLUMNS = ["brand", "fast_charge_port", "drivetrain", "segment", "car_body_type"]

# Used only if an older pipeline cannot expose its fitted encoder categories.
# These are documented values from the project dataset, not new categories.
FALLBACK_OPTIONS = {
    "fast_charge_port": ["CCS", "CHAdeMO"],
    "drivetrain": ["AWD", "FWD", "RWD"],
    "car_body_type": [
        "SUV", "Sedan", "Hatchback", "Liftback Sedan", "Station/Estate",
        "Cabriolet", "Coupe", "Small Passenger Van",
    ],
}


@st.cache_resource(show_spinner="Loading prediction model…")
def load_pipeline():
    """Load the supplied, already-trained pipeline without modifying it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH.name}. Place it beside app.py."
        )
    return joblib.load(MODEL_PATH)


def fitted_categories(pipeline):
    """Read categories from the fitted encoder, regardless of pipeline step name."""
    found = {}
    seen = set()

    def visit(obj, input_columns=None):
        if id(obj) in seen:
            return
        seen.add(id(obj))

        # OneHotEncoder has categories_ after fitting. Match it to input columns.
        if hasattr(obj, "categories_"):
            # Newer encoders retain feature_names_in_; older fitted encoders
            # get their column names from their ColumnTransformer selector.
            names = list(getattr(obj, "feature_names_in_", input_columns or []))
            for name, values in zip(names, obj.categories_):
                if name in CATEGORICAL_COLUMNS:
                    found[name] = sorted(str(value) for value in values)

        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                visit(step, input_columns)
        if hasattr(obj, "transformers_"):
            for _, transformer, columns in obj.transformers_:
                if transformer not in ("drop", "passthrough"):
                    visit(transformer, list(columns) if not isinstance(columns, str) else [columns])

    visit(pipeline)
    return found


def options_for(column, categories):
    """Prefer fitted training categories, with safe documented fallbacks."""
    options = categories.get(column) or FALLBACK_OPTIONS.get(column, [])
    if not options:
        # Brand and segment must come from this model's dataset; do not guess them.
        raise ValueError(
            f"Could not read valid '{column}' values from the saved pipeline. "
            "Please use a pipeline with a fitted categorical encoder."
        )
    return options


def select_default(options, preferred=None):
    return options.index(preferred) if preferred in options else 0


st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1120px; padding-top: 2.5rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: #f4f8f5; border: 1px solid #d8e8dc; border-radius: 14px;
        padding: 1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
    </style>""",
    unsafe_allow_html=True,
)

st.title("EV Range Predictor")
st.caption("Predict the driving range of an electric vehicle from its specifications.")
st.write("")

try:
    pipeline = load_pipeline()
    category_options = fitted_categories(pipeline)
    # Resolve these before rendering widgets so configuration failures are clear.
    brand_options = options_for("brand", category_options)
    port_options = options_for("fast_charge_port", category_options)
    drivetrain_options = options_for("drivetrain", category_options)
    segment_options = options_for("segment", category_options)
    body_options = options_for("car_body_type", category_options)
except Exception as exc:
    st.error(f"Unable to load the prediction model: {exc}")
    st.info("Add ev_range_svr_pipeline.joblib to this folder, then refresh the app.")
    st.stop()

with st.container(border=True):
    st.subheader("Vehicle Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_speed_kmh = st.number_input("Top speed (km/h)", min_value=1.0, value=180.0, step=1.0)
    with col2:
        torque_nm = st.number_input("Torque (Nm)", min_value=0.0, value=350.0, step=1.0)
    with col3:
        acceleration_0_100_s = st.number_input("0–100 km/h (seconds)", min_value=0.1, value=7.5, step=0.1)

with st.container(border=True):
    st.subheader("Battery & Charging")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        battery_capacity_kWh = st.number_input("Battery capacity (kWh)", min_value=0.1, value=75.0, step=0.1)
    with col2:
        number_of_cells = st.number_input("Number of cells", min_value=1, value=288, step=1)
    with col3:
        fast_charging_power_kw_dc = st.number_input("DC fast charging power (kW)", min_value=0.0, value=150.0, step=1.0)
    with col4:
        fast_charge_port = st.selectbox("Fast-charge port", port_options, index=select_default(port_options, "CCS"))

with st.container(border=True):
    st.subheader("Vehicle Dimensions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        length_mm = st.number_input("Length (mm)", min_value=1.0, value=4700.0, step=10.0)
    with col2:
        width_mm = st.number_input("Width (mm)", min_value=1.0, value=1900.0, step=10.0)
    with col3:
        height_mm = st.number_input("Height (mm)", min_value=1.0, value=1650.0, step=10.0)
    with col4:
        cargo_volume_l = st.number_input("Cargo volume (L)", min_value=0.0, value=500.0, step=10.0)
    st.divider()
    towing_capacity_kg = st.number_input(
        "Towing capacity (kg)", min_value=0.0, value=1000.0, step=50.0,
        help="Practical vehicle specification used by the model.",
    )

with st.container(border=True):
    st.subheader("Vehicle Configuration")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        brand = st.selectbox("Brand", brand_options)
    with col2:
        drivetrain = st.selectbox("Drivetrain", drivetrain_options, index=select_default(drivetrain_options, "RWD"))
    with col3:
        segment = st.selectbox("Segment", segment_options)
    with col4:
        car_body_type = st.selectbox("Body type", body_options, index=select_default(body_options, "SUV"))
    with col5:
        seats = st.number_input("Seats", min_value=1, value=5, step=1)

if st.button("Predict Range", type="primary", use_container_width=True):
    try:
        # A one-row DataFrame preserves model feature names and their exact order.
        features = pd.DataFrame([{
            "top_speed_kmh": top_speed_kmh,
            "battery_capacity_kWh": battery_capacity_kWh,
            "number_of_cells": number_of_cells,
            "torque_nm": torque_nm,
            "acceleration_0_100_s": acceleration_0_100_s,
            "fast_charging_power_kw_dc": fast_charging_power_kw_dc,
            "towing_capacity_kg": towing_capacity_kg,
            "cargo_volume_l": cargo_volume_l,
            "seats": seats,
            "length_mm": length_mm,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "brand": brand,
            "fast_charge_port": fast_charge_port,
            "drivetrain": drivetrain,
            "segment": segment,
            "car_body_type": car_body_type,
        }], columns=FEATURE_COLUMNS)
        prediction = float(pipeline.predict(features)[0])
        # Keep the result focused: a new EV has no known actual range yet.
        with st.container(border=True):
            st.metric("Estimated Driving Range", f"{prediction:,.0f} km")
        st.caption("This prediction is based on vehicle specifications and should be treated as an estimate.")
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

with st.expander("About the Model"):
    st.markdown("**Model:** Support Vector Regression (SVR)")
    a, b, c = st.columns(3)
    a.metric("MAE", "10.31 km")
    b.metric("RMSE", "13.65 km")
    c.metric("R²", "0.9824")
    st.caption("Performance reported on the held-out test set.")

with st.expander("Features used"):
    numerical_features = FEATURE_COLUMNS[:12]
    categorical_features = FEATURE_COLUMNS[12:]
    left, right = st.columns(2)
    with left:
        st.markdown("**Numerical Features**")
        st.write(", ".join(f"`{column}`" for column in numerical_features))
    with right:
        st.markdown("**Categorical Features**")
        st.write(", ".join(f"`{column}`" for column in categorical_features))
