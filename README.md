# Integrated Retail Analytics for Store Optimization and Demand Forecasting

End-to-end machine-learning project on weekly sales from 45 retail stores (~81 departments, Feb-2010 → Oct-2012):
**anomaly detection → feature engineering → store segmentation → inferred market-basket analysis → demand forecasting → strategy**.

> **Author:** Ameya · **Contribution:** Individual
> **Notebook (Colab):** [ADD LINK] · **Documentation (Google Doc):** [ADD LINK] · **Video walkthrough:** [ADD LINK]

---

## 1. Business problem

Retail margins are thin: over-stocking ties up capital, under-stocking loses sales. The chain needs to know
*where sales behave abnormally*, *which stores behave alike* (so they can be planned as a group), and
*how much each store-department will sell next week* given holidays, promotions and the local economy.

## 2. Data

| File | Grain | Rows | Key columns |
|---|---|---|---|
| `sales_data-set.csv` | store × dept × week | 421,570 | `Weekly_Sales` (target), `IsHoliday` |
| `Features_data_set.csv` | store × week | 8,190 | `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment` |
| `stores_data-set.csv` | store | 45 | `Type` (A/B/C), `Size` (sq ft) |

Notes: dates are `dd/mm/yyyy`; `MarkDown1-5` are missing before Nov-2011 (programme start, structural gap);
`Weekly_Sales` contains negative values (net returns).

## 3. Approach

| Component | Method | Evaluation |
|---|---|---|
| Anomaly detection | Per-series robust z-score & IQR · Isolation Forest (multivariate) · residual analysis after seasonal decomposition (time-based) | Flag rate, cross-method overlap, % of flags explained by holidays/markdowns |
| Preprocessing & features | Zero-imputation + indicator for markdowns; per-series winsorisation; calendar, named-holiday, weeks-to-Christmas, lag (1/2/4/8/52) and rolling features; economic deltas | — |
| Store segmentation | StandardScaler → PCA → K-Means (k by silhouette/elbow/DBI); Ward hierarchical cross-check | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Market basket (inferred) | Department co-movement correlation; Apriori on "above-median week" baskets | Support, confidence, lift |
| Demand forecasting | Linear/Ridge baseline → Random Forest → XGBoost/LightGBM (tuned with `TimeSeriesSplit`); SARIMA for long-horizon chain forecast | WMAE (holiday ×5), MAE, RMSE, R², MAPE |
| External factors | Feature importance, ablation (with/without), partial dependence, cluster-level sensitivity | Δ WMAE |
| Explainability | SHAP on final model | — |

Evaluation uses a **chronological split** (last 12 weeks held out) — no random shuffling, no lag leakage.

## 4. Results

> Fill in after running the notebook.

| Model | WMAE | MAE | RMSE | R² |
|---|---|---|---|---|
| Naive – last week | | | | |
| Naive – same week last year | | | | |
| Ridge (tuned) | | | | |
| Random Forest (tuned) | | | | |
| **Gradient Boosting (tuned) – final** | | | | |

* Segmentation: k = [K], silhouette = [X]
* Anomalies flagged: [X]% of rows; [X]% explained by holidays / markdowns
* Top cross-selling department pairs: [list]

## 5. Key insights & strategy (summary)

* Holidays are not interchangeable — Thanksgiving/Christmas drive large spikes, Labor Day barely moves sales → holiday-specific pre-positioning.
* Recency (`lag_1`, rolling means) and yearly seasonality (`lag_52`) dominate demand; economic factors are secondary but consistent, strongest in [cluster].
* [K] store segments beyond the A/B/C labels → segment-specific safety stock, markdown budgets and messaging.
* Positively co-moving departments → adjacency / bundle promotions; negatively correlated ones → avoid simultaneous discounts.

Full strategy and implementation challenges are in notebook §9 and the Google Doc.

## 6. Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── sales_data-set.csv
│   ├── Features_data_set.csv
│   └── stores_data-set.csv
├── notebooks/
│   └── Integrated_Retail_Analytics_Store_Optimization.ipynb   # main deliverable (template-based)
├── models/                      # produced by the notebook
│   ├── retail_forecast_bundle.joblib
│   ├── store_segments.csv
│   └── anomaly_report.csv
├── visualizations/              # exported charts used in the doc / video
└── docs/
    └── Project_Documentation.pdf  # export of the Google Doc
```

## 7. How to run

```bash
git clone https://github.com/<your-username>/retail-store-optimization-forecasting.git
cd retail-store-optimization-forecasting
pip install -r requirements.txt
jupyter notebook notebooks/Integrated_Retail_Analytics_Store_Optimization.ipynb
```

* Set `DATA_DIR` in the *Dataset Loading* cell (default `data`; on Colab e.g. `/content/drive/MyDrive/.../data`).
* The notebook runs top-to-bottom without errors. Optional packages (`xgboost`, `lightgbm`, `mlxtend`, `pmdarima`, `shap`) are auto-installed if missing and skipped gracefully if unavailable.
* Runtime: ~15–30 min on Colab CPU. `SUBSAMPLE` in ML Model-2 controls Random Forest training size.

## 8. Requirements

See `requirements.txt` (pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib, seaborn, joblib, xgboost, lightgbm, mlxtend, pmdarima, shap).

## 9. Future work

Transaction-level data for true market-basket analysis and customer personalisation · hierarchical forecast reconciliation ·
quantile (probabilistic) forecasts for safety-stock · store-cluster recommendation layer · automated retraining and anomaly alerting.
