# Broiler Weight Predictor (Streamlit)

A Streamlit front end for the report's selected Random Forest model. The model was selected using leave-one-batch-out validation (mean R² 0.9839; mean MAE 0.0847 kg), not the row-level test leaderboard. The app accepts the report's seven input variables and predicts average flock body weight in kilograms.

## Model artifact and compatibility

Place these files beside `app.py`:

- `broiler_weight_prediction_pipeline.joblib` — the supplied fitted scikit-learn preprocessing-and-Random-Forest pipeline.
- `broiler_model_metadata.json` — metadata matching that artifact. Start from `broiler_model_metadata.example.json`; include numeric `observed_ranges` from the training dataframe if available, so the app can flag out-of-range values.

The supplied artifact was serialized with **scikit-learn 1.6.1**. This project pins that exact scikit-learn version in `requirements.txt`; do not change it without verifying model loading and predictions. `joblib` files are Python pickle-based; only load an artifact from a trusted source.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp broiler_model_metadata.example.json broiler_model_metadata.json
# The supplied model artifact should be named broiler_weight_prediction_pipeline.joblib here.
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Put `app.py`, `requirements.txt`, `broiler_model_metadata.json`, and the model artifact in a GitHub repository accessible to Streamlit Community Cloud.
2. In Streamlit Community Cloud, create an app from that repository and set the main file path to `app.py` (or `broiler_streamlit/app.py` if this folder is nested).
3. Deploy and verify a prediction against a known worked example from the report/notebook.

The repository will contain the serialized model, so confirm that sharing it with the chosen repository/deployment audience is acceptable. Do not upload farm records unless they are explicitly needed and authorized.

## Scope and caveats

The report's grouped metrics are internal validation results from reconstructed batches on two Abuja farms. They do not establish external validity on other farms or guarantee the error for an individual prediction. Treat outputs as decision support alongside direct weighing and professional farm judgement. The target is flock-average body weight, not total flock weight.
