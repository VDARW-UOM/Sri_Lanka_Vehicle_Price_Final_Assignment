# Sri Lankan Used Vehicle Price Prediction
### MSc Machine Learning — Final Assignment

An end-to-end tabular regression study predicting **advertised asking prices in Sri Lankan rupees (LKR)** from Sri Lankan vehicle listing attributes. Includes two executed, academic-style Jupyter notebooks and an inference interface built using Streamlit.

## Contents

| File | Role |
|---|---|
| `01_Vehicle_Data_Cleaning_EDA_Submission.ipynb` | Raw-data audit, type normalization, cleaning, data-quality visualizations, export |
| `02_Vehicle_Price_Regression_Submission.ipynb` | Leakage-aware experimental partitioning, Decision Tree / Random Forest / XGBoost / LightGBM, group-aware tuning, comparison, diagnostics, model serialization |
| `ikman-scrapes.csv` | Original downloaded Ikman listing records (unchanged) |
| `vehicle_ml_outputs/` | Generated cleaned records, cleaning audit, CV and test scores, figures, saved model |
| `app.py` | Streamlit predictor + held-out model results |
| `requirements.txt` | Python dependencies |

Both computational research documents are **native `.ipynb` notebooks** containing executable cells, concise academic Markdown, in-notebook graphics and saved outputs—not exported `.py` scripts. `app.py` is a conventional Python module because Streamlit requires a Python application entrypoint; it is not an analysis notebook.

## Reproduce in Google Colab

1. Upload the **entire** project folder to **My Drive**. The two notebooks automatically look for a directory named `Sri_Lanka_Vehicle_Price_Jupyter_Notebooks` or `Sri_Lanka_Vehicle_Price_Final_Assignment`. If your Drive folder has a different name, edit `PROJECT_NAME` or the candidate path in the first path-resolution cell of each notebook.
2. Open `01_Vehicle_Data_Cleaning_EDA_Submission.ipynb` in Colab, connect the runtime and use **Runtime → Run all**. The notebook mounts Drive if needed and produces `vehicle_ml_outputs/cleaned_vehicle_data.csv` and `cleaning_report.json`.
3. Open `02_Vehicle_Price_Regression_Submission.ipynb` and use **Runtime → Run all**. It reads the cleaned CSV, fits/compares regression methods, and generates evaluation artifacts and `trained_vehicle_price_model.pkl`. Execution of cross-validation and tuning takes longer than the earlier activity notebook; CPU is sufficient.
4. Save the executed notebooks in Drive. **Do not upload only the `.ipynb` files:** the initial CSV must remain alongside them.

If a Colab dependency is unavailable, run the following command in a separate code cell **once** before running all:

```python
%pip install -q lightgbm xgboost scikit-learn seaborn
```

Runtime package versions may alter small numeric results. Keep the same versions when unpickling a trained model in a different environment.

## Run the web application locally

The app runs in a terminal on your own computer (or can be deployed on Streamlit Community Cloud). Google Colab's notebook Run All is not a hosting service for the Streamlit app.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run Notebook 01 and Notebook 02 first, so the cleaned data and serialized pipeline exist. The app reads its inputs from `vehicle_ml_outputs` and offers vehicle detail inputs, an asking-price estimate, model-comparison metrics and study limitations.

## Methodology

- **Regression target:** `PriceLKR`, a seller's advertised asking price—not a confirmed transaction or an inflation-adjusted valuation.
- **Predictors:** brand, model, edition, manufacturing year, fuel, transmission, body type, engine cc, mileage and negotiability.
- **Data controls:** validated numeric conversion, normalised text, duplicate removal, audited target exclusions, conservative outlier policy and retention of unresolved missing predictors.
- **Partitioning:** `GroupShuffleSplit` groups identical predictor rows to reduce repeat-listing leakage, with an independent test holdout. Modified or relisted advertisements may still leak without listing IDs; there is no time-based backtest because scrape dates are unavailable.
- **Model selection:** compare fitted candidates on validation MAE; run a small 3-fold group-aware search for two strong ensemble families using **training data only**; use validation to choose the saved estimator; reserve holdout scores for reporting.
- **Metrics:** MAE and RMSE in LKR, R², median absolute percentage error, price-band error and permutation importance. R² is **not** classification accuracy.
- **Persistence:** export the full sklearn pipeline including its imputers and encoders, verified by a reload-and-predict test.

## Related studies / provenance

- Original KDU scraping project and dataset: [MarcWije/CarPricePredictor](https://github.com/MarcWije/CarPricePredictor). The project documents collecting vehicle listings from Sri Lankan marketplaces and evaluating ensemble regressors. **This assignment retrains models independently** and does not claim the KDU project's performance as its own.
- Basnayaka, D. (2023). *Prediction of Market Price for Used Cars in Sri Lanka Based on Machine Learning Approaches*. [IIT research repository](https://dlib.iit.ac.lk/xmlui/handle/123456789/1634).
- Jayadeera, T. P. & Jayamanne, D. J. (2019). *Fair Price Prediction System for Used Cars in Sri Lanka Using Machine Learning and Robotic Process Automation*. [NSBM](https://www.nsbm.ac.lk/research_publication/icobi-2019_fair-price-prediction-system-for-used-cars-in-sri-lanka-using-machine-learning-and-robotic-process-automation/).

Metrics from other papers are not numerically interchangeable with this project's results due to differences in datasets and validation methods.

## Study limitations and responsible use

- Source observations are **advertised listings**, which may contain seller mistakes or down-payment amounts.
- Advert timestamps and IDs are unavailable; duplicate cleaning does not guarantee independence of reposted ads.
- Rare brands/luxury price ranges have fewer samples and larger absolute errors.
- The study does not incorporate imported vehicle market changes, taxes, exchange rates or real-time market information.
- The application is an academic research prototype, not a professional valuation tool.

## Submission checklist

- [ ] Enter your student name/index number and module identifier in the header of both notebooks (if required by your lecturer).
- [ ] Verify the latest Colab Run All completed on both notebooks.
- [ ] Include executed outputs inside notebooks, original CSV and final `.pkl` model.
- [ ] Test Streamlit inference with at least one vehicle.
- [ ] Confirm Moodle submission requirements and upload/GitHub sharing permissions for the source dataset.
