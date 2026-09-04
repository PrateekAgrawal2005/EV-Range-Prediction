# EV Range Prediction

A Streamlit application that estimates an electric vehicle's driving range from vehicle specifications. The app loads the final, saved Support Vector Regression (SVR) pipeline directly; it does not retrain, alter, or save the model.

## Problem statement

Driving range is an important consideration when comparing electric vehicles. This project predicts an EV's estimated range in kilometres from its performance, battery, charging, dimensions, and configuration specifications.

## Final model and performance

- Final model: Support Vector Regression (SVR)
- MAE: 10.309 km
- RMSE: 13.646 km
- R²: 0.9824

Metrics were reported on the held-out test set.

## Features used

Numerical features:

- `top_speed_kmh`
- `battery_capacity_kWh`
- `number_of_cells`
- `torque_nm`
- `acceleration_0_100_s`
- `fast_charging_power_kw_dc`
- `towing_capacity_kg`
- `cargo_volume_l`
- `seats`
- `length_mm`
- `width_mm`
- `height_mm`

Categorical features:

- `brand`
- `fast_charge_port`
- `drivetrain`
- `segment`
- `car_body_type`

`efficiency_wh_per_km` is intentionally excluded from the final model because it causes target leakage. `range_km` is the target, not an application input.

## How the app works

`app.py` loads `ev_range_svr_pipeline.joblib` from the repository root using a relative file path. It collects the 17 features above in a one-row pandas DataFrame and sends that DataFrame directly to the saved pipeline for prediction. The pipeline already contains preprocessing, categorical encoding, scaling, and the SVR model.

## Run locally

1. Install Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Start Streamlit from the project root:

   ```bash
   streamlit run app.py
   ```

## Project structure

```text
EV/
├── app.py
├── ev_range_svr_pipeline.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Streamlit Community Cloud deployment

Push `app.py`, `ev_range_svr_pipeline.joblib`, `requirements.txt`, `README.md`, and `.gitignore` to the GitHub repository. In Streamlit Community Cloud, select that repository and set the main file path to `app.py`.

The requirements pin the verified package versions, including `scikit-learn==1.6.1`, which is required to load this saved pipeline reliably. The model file must remain in the repository root with the exact lowercase filename `ev_range_svr_pipeline.joblib`.

Live Demo:
https://ev-range-prediction-5quiycdpj8khgu6pdba4w3.streamlit.app/
