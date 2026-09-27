from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

SALES_FILE = DATA_DIR / "sales_data-set.csv"
FEATURES_FILE = DATA_DIR / "Features_data_set.csv"
STORES_FILE = DATA_DIR / "stores_data-set.csv"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Final evaluation values produced by the project notebook.
# These are dashboard/reporting values, not re-trained when Flask starts.
MODEL_RESULTS = [
    {"model": "Linear Regression (log target)", "WMAE": 32342.365, "MAE": 34163.669, "RMSE": 388724.852, "R2": -314.128},
    {"model": "Ridge (tuned, TimeSeriesCV)", "WMAE": 32342.365, "MAE": 34163.668, "RMSE": 388724.845, "R2": -314.128},
    {"model": "Random Forest", "WMAE": 1484.786, "MAE": 1386.061, "RMSE": 2951.686, "R2": 0.982},
    {"model": "Random Forest (tuned)", "WMAE": 1502.148, "MAE": 1402.562, "RMSE": 3001.092, "R2": 0.981},
    {"model": "XGBoost", "WMAE": 1342.670, "MAE": 1267.040, "RMSE": 2621.429, "R2": 0.986},
    {"model": "XGBoost (tuned)", "WMAE": 1309.426, "MAE": 1231.836, "RMSE": 2565.665, "R2": 0.986},
]

NOTEBOOK_SUMMARY = {
    "best_model": "XGBoost (tuned)",
    "best_wmae": 1309.426,
    "best_mae": 1231.836,
    "best_rmse": 2565.665,
    "best_r2": 0.986272,
    "ml_chain_mape": 0.0137,
    "sarima_chain_mape": 0.0296,
    "external_importance_share": 0.022,
    "train_cutoff": "2012-08-10",
    "train_weeks": 131,
    "test_weeks": 12,
}

HOLIDAYS = {
    "SuperBowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "LaborDay": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}

# Fallback is the final two-cluster assignment reported by the notebook.
# If models/store_segments.csv exists, the app uses that file instead.
FALLBACK_CLUSTER_1 = {3, 30, 33, 36, 37, 38, 42, 43, 44}


def _assert_files() -> None:
    missing = [str(p) for p in (SALES_FILE, FEATURES_FILE, STORES_FILE) if not p.exists()]
    if missing:
        joined = "\n - ".join(missing)
        raise FileNotFoundError(
            "The dashboard could not find the project CSV files. Expected:\n - " + joined +
            "\n\nKeep app.py beside a data/ folder, or set DATA_DIR to the folder containing the CSVs."
        )


def _clean_float(value: Any, digits: int = 3) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for c in out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records")


@lru_cache(maxsize=1)
def load_project_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the same three source tables used by the notebook and build dashboard fields."""
    _assert_files()

    sales = pd.read_csv(SALES_FILE, parse_dates=["Date"], dayfirst=True)
    features = pd.read_csv(FEATURES_FILE, parse_dates=["Date"], dayfirst=True)
    stores = pd.read_csv(STORES_FILE)

    df = (
        sales.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
        .merge(stores, on="Store", how="left")
        .sort_values(["Store", "Dept", "Date"])
        .reset_index(drop=True)
    )

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    df["Holiday_Name"] = "None"
    for holiday, dates in HOLIDAYS.items():
        mask = df["Date"].isin(pd.to_datetime(dates))
        df.loc[mask, "Holiday_Name"] = holiday

    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df["Total_MarkDown"] = df[markdown_cols].sum(axis=1, min_count=1)
    df["Has_MarkDown"] = df["Total_MarkDown"].fillna(0).gt(0).astype(int)

    # Notebook's statistical anomaly layer: robust MAD z-score per store-department series.
    group_keys = [df["Store"], df["Dept"]]
    median = df["Weekly_Sales"].groupby(group_keys).transform("median")
    abs_dev = (df["Weekly_Sales"] - median).abs()
    mad = abs_dev.groupby(group_keys).transform("median")
    safe_mad = mad.mask(mad == 0, np.nan)
    df["robust_z"] = 0.6745 * (df["Weekly_Sales"] - median) / safe_mad
    df["anom_z"] = df["robust_z"].abs().gt(3.5).fillna(False).astype(int)

    return df, stores


def load_cluster_map(stores: pd.DataFrame) -> dict[int, int]:
    segment_file = MODEL_DIR / "store_segments.csv"
    if segment_file.exists():
        seg = pd.read_csv(segment_file)
        if "Store" not in seg.columns:
            first = seg.columns[0]
            seg = seg.rename(columns={first: "Store"})
        if "cluster_km" in seg.columns:
            return {
                int(row.Store): int(row.cluster_km)
                for row in seg[["Store", "cluster_km"]].dropna().itertuples(index=False)
            }

    return {
        int(store): (1 if int(store) in FALLBACK_CLUSTER_1 else 0)
        for store in stores["Store"].tolist()
    }


@lru_cache(maxsize=1)
def build_dashboard_payload() -> dict[str, Any]:
    df, stores = load_project_frame()
    cluster_map = load_cluster_map(stores)

    chain = df.groupby("Date", as_index=False)["Weekly_Sales"].sum().sort_values("Date")
    chain["Date"] = chain["Date"].dt.strftime("%Y-%m-%d")

    store_avg = (
        df.groupby(["Store", "Type"], observed=True)["Weekly_Sales"]
        .mean()
        .reset_index()
        .sort_values("Weekly_Sales", ascending=False)
    )
    store_avg["Type"] = store_avg["Type"].astype(str)
    store_avg["Cluster"] = store_avg["Store"].map(cluster_map).fillna(-1).astype(int)
    store_avg = store_avg.merge(stores[["Store", "Size"]], on="Store", how="left")

    dept_avg = (
        df.groupby("Dept", as_index=False)["Weekly_Sales"]
        .mean()
        .sort_values("Weekly_Sales", ascending=False)
    )

    holiday_mean = (
        df.groupby("Holiday_Name")["Weekly_Sales"]
        .mean()
        .reindex(["None", "SuperBowl", "LaborDay", "Thanksgiving", "Christmas"])
    )
    base = float(holiday_mean.loc["None"])
    holiday_rows = [
        {
            "holiday": name,
            "mean_sales": _clean_float(value, 2),
            "uplift_pct": _clean_float((value / base - 1) * 100, 2),
        }
        for name, value in holiday_mean.items()
    ]

    stat_anom = df[df["anom_z"] == 1].copy()
    anom_timeline = stat_anom.groupby("Date").size().rename("count").reset_index()
    anom_timeline["Date"] = anom_timeline["Date"].dt.strftime("%Y-%m-%d")

    anom_pivot = stat_anom.pivot_table(
        index="Dept", columns="Store", values="anom_z", aggfunc="sum", fill_value=0
    ).sort_index()

    # Department co-movement from week-over-week changes, matching the notebook logic.
    dept_wk = df.pivot_table(
        index=["Store", "Date"], columns="Dept", values="Weekly_Sales", aggfunc="sum"
    )
    dept_wk = dept_wk.loc[:, dept_wk.notna().mean() > 0.8]
    chg = dept_wk.groupby(level=0).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    chg = chg.clip(-5, 5)
    dept_corr = chg.corr(min_periods=500)

    mask = np.triu(np.ones(dept_corr.shape, dtype=bool), k=1)
    pairs = dept_corr.where(mask).stack().sort_values(ascending=False)
    pairs.index = pairs.index.set_names(["Dept_A", "Dept_B"])
    pairs_df = pairs.rename("corr").reset_index()

    top_pairs = pairs_df.head(15).copy()
    negative_pairs = pairs_df.sort_values("corr").head(8).copy()
    top_pairs["corr"] = top_pairs["corr"].round(3)
    negative_pairs["corr"] = negative_pairs["corr"].round(3)

    corr_obj = dept_corr.round(3).astype(object).where(dept_corr.notna(), None)

    segment_summary = (
        store_avg.groupby("Cluster")
        .agg(
            stores=("Store", "count"),
            avg_weekly_sales=("Weekly_Sales", "mean"),
            avg_size=("Size", "mean"),
        )
        .reset_index()
    )

    best = min(MODEL_RESULTS, key=lambda r: r["WMAE"])
    top10_share = (
        df[df["Dept"].isin(dept_avg.head(10)["Dept"])]["Weekly_Sales"].sum()
        / df["Weekly_Sales"].sum()
    )

    payload = {
        "meta": {
            "date_min": df["Date"].min().strftime("%Y-%m-%d"),
            "date_max": df["Date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(df)),
            "stores": int(df["Store"].nunique()),
            "departments": int(df["Dept"].nunique()),
            "total_sales": round(float(df["Weekly_Sales"].sum()), 2),
            "avg_chain_weekly_sales": round(float(chain["Weekly_Sales"].mean()), 2),
            "anomaly_count": int(stat_anom.shape[0]),
            "anomaly_rate": round(float(stat_anom.shape[0] / len(df)), 5),
            "top10_department_share": round(float(top10_share), 4),
            "best_model": best["model"],
            "best_wmae": best["WMAE"],
        },
        "filters": {
            "stores": [int(x) for x in sorted(df["Store"].dropna().unique())],
            "departments": [int(x) for x in sorted(df["Dept"].dropna().unique())],
        },
        "chain": _records(chain),
        "store_avg": _records(store_avg),
        "dept_avg": _records(dept_avg),
        "holidays": holiday_rows,
        "anomaly_timeline": _records(anom_timeline),
        "anomaly_heatmap": {
            "stores": [int(x) for x in anom_pivot.columns.tolist()],
            "departments": [int(x) for x in anom_pivot.index.tolist()],
            "z": anom_pivot.astype(int).values.tolist(),
        },
        "dept_corr": {
            "departments": [int(x) for x in dept_corr.columns.tolist()],
            "z": corr_obj.values.tolist(),
        },
        "top_pairs": _records(top_pairs),
        "negative_pairs": _records(negative_pairs),
        "segment_summary": _records(segment_summary),
        "model_results": MODEL_RESULTS,
        "notebook_summary": NOTEBOOK_SUMMARY,
    }
    return payload


def filtered_series(store: str, dept: str) -> dict[str, Any]:
    df, _ = load_project_frame()
    frame = df
    title_bits: list[str] = []

    if store != "all":
        store_i = int(store)
        frame = frame[frame["Store"] == store_i]
        title_bits.append(f"Store {store_i}")
    else:
        title_bits.append("All stores")

    if dept != "all":
        dept_i = int(dept)
        frame = frame[frame["Dept"] == dept_i]
        title_bits.append(f"Dept {dept_i}")
    else:
        title_bits.append("all departments")

    if frame.empty:
        return {
            "title": " / ".join(title_bits),
            "dates": [],
            "sales": [],
            "anomaly_dates": [],
            "anomaly_sales": [],
            "kpis": {"total_sales": 0, "avg_weekly_sales": 0, "peak_week_sales": 0, "anomalies": 0},
        }

    weekly = frame.groupby("Date", as_index=False)["Weekly_Sales"].sum().sort_values("Date")

    # For filtered views, a week is highlighted when at least one underlying store-dept row was flagged.
    flagged_dates = frame.loc[frame["anom_z"] == 1, "Date"].drop_duplicates()
    flagged = weekly[weekly["Date"].isin(flagged_dates)]

    return {
        "title": " / ".join(title_bits),
        "dates": weekly["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "sales": [round(float(x), 2) for x in weekly["Weekly_Sales"]],
        "anomaly_dates": flagged["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "anomaly_sales": [round(float(x), 2) for x in flagged["Weekly_Sales"]],
        "kpis": {
            "total_sales": round(float(frame["Weekly_Sales"].sum()), 2),
            "avg_weekly_sales": round(float(weekly["Weekly_Sales"].mean()), 2),
            "peak_week_sales": round(float(weekly["Weekly_Sales"].max()), 2),
            "anomalies": int(frame["anom_z"].sum()),
        },
    }


@app.route("/")
def index():
    try:
        dashboard = build_dashboard_payload()
        return render_template("index.html", dashboard=dashboard, startup_error=None)
    except Exception as exc:
        return render_template("index.html", dashboard=None, startup_error=str(exc)), 500


@app.route("/api/filter")
def api_filter():
    store = request.args.get("store", "all").strip().lower()
    dept = request.args.get("dept", "all").strip().lower()

    try:
        if store != "all":
            int(store)
        if dept != "all":
            int(dept)
        return jsonify(filtered_series(store, dept))
    except (TypeError, ValueError):
        return jsonify({"error": "store and dept must be integers or 'all'"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    try:
        _assert_files()
        return jsonify({"status": "ok", "data_dir": str(DATA_DIR)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
