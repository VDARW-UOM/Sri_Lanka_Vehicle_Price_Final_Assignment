"""Streamlit interface for the trained Sri Lankan advertised vehicle price estimator."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "vehicle_ml_outputs"
MODEL_PATH = OUTPUTS / "trained_vehicle_price_model.pkl"
DATA_PATH = OUTPUTS / "cleaned_vehicle_data.csv"
METRICS_PATH = OUTPUTS / "test_metrics.csv"
META_PATH = OUTPUTS / "model_metadata.json"

st.set_page_config(page_title="Lanka Vehicle Value | ML Study", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:2rem; padding-bottom:4rem; max-width:1160px;}
.hero {padding:1.7rem 1.9rem; border-radius:17px; background:linear-gradient(110deg,#0b243b,#174768); color:#f8fafc; margin-bottom:1rem;}
.hero h1 {font-size:2rem; font-weight:750; color:#fff; margin:0 0 .35rem; letter-spacing:-.035em;}
.hero p {color:#d5e3ef; margin:0; font-size:.98rem;}
.eyebrow {color:#91c4dd; text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; padding-bottom:.5rem;}
.result {border:1px solid #b5d1df; border-radius:14px; background:#f1f7fa; color:#15314c; padding:1.5rem; margin:.5rem 0;}
.result .amount {font-size:2.35rem; font-weight:800; letter-spacing:-.03em;}
.smallnote {font-size:.88rem; color:#526477;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class='hero'><div class='eyebrow'>MSc Machine Learning · Sri Lanka</div>
<h1>Used Vehicle Price Estimation</h1><p>A regression study of advertised vehicle prices using ensemble machine learning.</p></div>""", unsafe_allow_html=True)

missing = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error("The project requires Notebook 01 and Notebook 02 to be executed before launching the app. Missing: " + ", ".join(missing))
    st.stop()

@st.cache_resource(show_spinner="Loading the fitted regression pipeline…")
def get_pipeline(path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def get_data(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def get_metadata(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

model = get_pipeline(MODEL_PATH)
records = get_data(DATA_PATH)
meta = get_metadata(META_PATH)
required_cat = ["Brand", "Model", "Edition", "Transmission", "FuelType", "BodyType", "Negotiable"]
required_num = ["ManufactureYear", "EngineCC", "MileageKM"]
required = required_cat + required_num
if not set(required).issubset(records.columns):
    st.error("Cleaned dataset does not match the model's expected feature schema.")
    st.stop()

for column in required_cat:
    records[column] = records[column].fillna("Unknown").astype(str)


def options(col, subset=None):
    table = records if subset is None or subset.empty else subset
    vals = sorted(x for x in table[col].dropna().unique().tolist() if str(x).strip())
    return vals or ["Unknown"]


def index_of(values, preferred):
    return values.index(preferred) if preferred in values else 0


def safe_number(value, default):
    return float(value) if pd.notna(value) and np.isfinite(value) else default


def format_lkr(amount):
    return "Rs. {:,.0f}".format(float(amount))

predict_tab, study_tab, about_tab = st.tabs(["Price estimate", "Model evaluation", "Study & limitations"])
with predict_tab:
    st.subheader("Vehicle specifications")
    st.caption("Specify an advertised vehicle. The model predicts an indicative asking price, not a guaranteed selling price.")
    left, right = st.columns(2, gap="large")
    with left:
        brands = options("Brand")
        brand = st.selectbox("Brand", brands, index=index_of(brands, "Toyota"))
        brand_df = records[records.Brand == brand]
        models = options("Model", brand_df)
        vehicle_model = st.selectbox("Model", models)
        make_model_df = brand_df[brand_df.Model == vehicle_model]
        edition = st.selectbox("Edition", options("Edition", make_model_df))
        year = st.number_input("Year of manufacture", min_value=1950, max_value=2026,
                               value=2018, step=1)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=800000,
                                  value=75000, step=1000)
    with right:
        fuel = st.selectbox("Fuel type", options("FuelType", make_model_df))
        transmission = st.selectbox("Transmission", options("Transmission", make_model_df))
        body = st.selectbox("Body type", options("BodyType", make_model_df))
        negotiable = st.selectbox("Negotiable", options("Negotiable"))
        typical_cc = safe_number(make_model_df.EngineCC.median(), 1500.0)
        engine_cc = st.number_input("Engine capacity (cc)", min_value=0, max_value=8000,
                                    value=int(typical_cc), step=100,
                                    help="Set to 0 for an electric or unspecified engine capacity.")
    submitted = st.button("Estimate asking price", use_container_width=True, type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Brand":brand, "Model":vehicle_model, "Edition":edition,
            "ManufactureYear":year, "Transmission":transmission,
            "FuelType":fuel, "BodyType":body, "EngineCC":np.nan if engine_cc == 0 else engine_cc,
            "MileageKM":mileage, "Negotiable":negotiable,
        }])[required]
        try:
            prediction = max(0.0, float(model.predict(row)[0]))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.markdown(f"<div class='result'><div>Estimated advertised price</div>"
                        f"<div class='amount'>{format_lkr(prediction)}</div>"
                        f"<div class='smallnote'>Model: {meta.get('model', 'saved estimator')} · LKR · Indicative only</div></div>",
                        unsafe_allow_html=True)
            if prediction > 0:
                st.caption(f"Approximately {prediction/1_000_000:,.2f} million LKR. Listing conditions, date, negotiation and inspections are not modelled in full.")

with study_tab:
    st.subheader("Held-out model comparison")
    if METRICS_PATH.exists():
        results = pd.read_csv(METRICS_PATH)
        show_cols = [x for x in ["Model", "MAE_LKR", "RMSE_LKR", "R2", "Median_APE_pct"] if x in results]
        st.dataframe(results[show_cols].style.format({"MAE_LKR":"{:,.0f}",
            "RMSE_LKR":"{:,.0f}","R2":"{:.3f}","Median_APE_pct":"{:.1f}"}),
            use_container_width=True,hide_index=True)
        if "MAE_LKR" in results:
            st.bar_chart(results.set_index("Model")[["MAE_LKR"]])
        winner = results[results.Model == meta.get("model", "")]
        if not winner.empty:
            a,b,c = st.columns(3)
            best=winner.iloc[0]
            a.metric("Selected-model MAE",format_lkr(best.MAE_LKR))
            b.metric("Selected-model RMSE",format_lkr(best.RMSE_LKR))
            c.metric("Selected-model R²",f"{best.R2:.3f}")
    else:
        st.info("Run Notebook 02 to generate the model comparison table.")
    st.caption("Lower MAE/RMSE and higher R² are preferable. R² is not a percentage of exactly correct price predictions.")

with about_tab:
    st.subheader("Research context")
    st.markdown("""
    **Data source:** Sri Lankan vehicle advertisements collected from Ikman.lk through the public
    [KDU CarPricePredictor project](https://github.com/MarcWije/CarPricePredictor).

    **Target:** seller-advertised asking price in Sri Lankan rupees, not confirmed transaction value.

    **Method:** categorical one-hot encoding, median numeric imputation, robust numeric scaling,
    log-target transformation and ensemble regression. Models were selected using validation MAE,
    with a group-disjoint held-out test for reporting.

    **Cautions:** listing dates and unique advertisement identifiers are absent; modified duplicate
    listings may remain. Some low prices may be deposits, while uncommon luxury cars have sparse training
    coverage. Reported estimates should not be used as a professional valuation or a financial decision
    without independent confirmation.
    """)

st.divider()
st.caption("Academic ML prototype · Built for the MSc Machine Learning final assignment · Not a live valuation service")
